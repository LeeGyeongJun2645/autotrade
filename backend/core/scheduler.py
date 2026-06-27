"""APScheduler 기반 자동매매 스케줄러.

KIS(주식): 평일 09:00~15:29 매분 실행 | 15:20 변동성 돌파 강제 매도
Upbit(코인): 24/7 5분마다 실행

포지션 상태는 인메모리(_positions)에 보관.
7단계(models/trade.py) 완성 후 DB 기록 추가 예정.
"""

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.api import kis, telegram, upbit
from backend.config import settings
from backend.core.daily_report import send_daily_report
from backend.core.risk_manager import Position, RiskManager
from backend.core import sim_log
from backend.models.trade import delete_position, insert_trade, load_all_positions, upsert_position
from backend.strategies import StrategyResult
from backend.strategies import moving_average as ma_strategy
from backend.strategies import rsi as rsi_strategy
from backend.strategies import volatility_breakout as vb_strategy

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
_OHLCV_SEM = asyncio.Semaphore(10)  # 업비트/KIS OHLCV 동시 요청 최대 10개


class TradingScheduler:
    """자동매매 스케줄러.

    add_kis_symbol / add_upbit_ticker 로 감시 종목을 등록하고
    start() 를 호출하면 자동매매가 시작된다.
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler(timezone=KST)
        self._positions: dict[str, Position] = {}
        self._kis_symbols: list[str] = []
        self._upbit_tickers: list[str] = []
        self._lock = asyncio.Lock()
        self._ml_signals: dict[str, dict] = {}  # {symbol: {"signal": ..., "buy_prob": ..., "news_score": ...}}

    # ── 종목 관리 ─────────────────────────────────────────────────

    def add_kis_symbol(self, symbol: str) -> None:
        if symbol not in self._kis_symbols:
            self._kis_symbols.append(symbol)
            logger.info("KIS 감시 종목 추가: %s", symbol)

    def remove_kis_symbol(self, symbol: str) -> None:
        self._kis_symbols = [s for s in self._kis_symbols if s != symbol]

    def add_upbit_ticker(self, ticker: str) -> None:
        if ticker not in self._upbit_tickers:
            self._upbit_tickers.append(ticker)
            logger.info("Upbit 감시 티커 추가: %s", ticker)

    def remove_upbit_ticker(self, ticker: str) -> None:
        self._upbit_tickers = [t for t in self._upbit_tickers if t != ticker]

    def get_positions(self) -> dict[str, Position]:
        """현재 보유 포지션 스냅샷 반환."""
        return dict(self._positions)

    def get_ml_signals(self) -> dict[str, dict]:
        """최근 ML 신호 스냅샷 반환."""
        return dict(self._ml_signals)

    # ── 스케줄러 생명주기 ──────────────────────────────────────────

    async def restore_positions(self) -> None:
        """DB에서 포지션 복구 후 거래소 실잔고 비교로 좀비 포지션 제거."""
        saved = await load_all_positions()
        async with self._lock:
            for symbol, (_, pos) in saved.items():
                if symbol not in self._positions:
                    self._positions[symbol] = pos

        # ── 좀비 포지션 검증 (서버 재시작 사이 청산된 포지션 제거) ──
        await self._validate_positions()

    async def _validate_positions(self) -> None:
        """거래소 실잔고와 DB 포지션 비교 — 불일치 포지션(좀비) 자동 제거."""
        zombies: list[str] = []
        try:
            if settings.upbit_access_key:
                bal = await upbit.get_balance()
                holdings = {
                    h["currency"]: float(h.get("balance", 0))
                    for h in bal.get("holdings", [])
                }
                async with self._lock:
                    for symbol, pos in list(self._positions.items()):
                        if not symbol.startswith("KRW-"):
                            continue
                        currency = symbol.replace("KRW-", "")
                        actual_qty = holdings.get(currency, 0.0)
                        if actual_qty < pos.qty * 0.9:  # 10% 이상 차이 → 좀비
                            zombies.append(symbol)
        except Exception:
            logger.warning("[좀비검증] 업비트 잔고 조회 실패, 건너뜀")

        for symbol in zombies:
            logger.warning("[좀비포지션] %s DB에 있으나 실잔고 없음 → 제거", symbol)
            async with self._lock:
                self._positions.pop(symbol, None)
            await delete_position(symbol)
            sim_log.push(symbol, f"좀비 포지션 제거 (서버 재시작 사이 청산된 것으로 추정)", "SELL")

    def start(self) -> None:
        """스케줄러 시작. FastAPI lifespan 에서 호출."""
        # KIS: 평일 09:00~15:59 매분 (15:30 이후는 _kis_tick 내부에서 스킵)
        self._scheduler.add_job(
            self._kis_tick,
            CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*", timezone=KST),
            id="kis_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # KIS: 15:20 변동성 돌파 EOD 강제 매도
        self._scheduler.add_job(
            self._kis_eod_sell,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=20, timezone=KST),
            id="kis_eod",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # Upbit: 24/7 5분마다
        self._scheduler.add_job(
            self._upbit_tick,
            CronTrigger(minute="*/5", timezone=KST),
            id="upbit_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 일일 백테스트 리포트: 매일 08:00 KST
        self._scheduler.add_job(
            self._daily_report,
            CronTrigger(hour=8, minute=0, timezone=KST),
            id="daily_report",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # AI 에이전트 경쟁 시뮬레이션: 5분마다
        self._scheduler.add_job(
            self._agent_tick,
            CronTrigger(minute="*/5", timezone=KST),
            id="agent_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 전 에이전트 일일 재학습: 매일 03:00 (거래 없는 새벽, 최신 데이터 반영)
        self._scheduler.add_job(
            self._daily_retrain,
            CronTrigger(hour=3, minute=0, timezone=KST),
            id="daily_retrain",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info(
            "스케줄러 시작 | KIS: %s | Upbit: %s",
            self._kis_symbols,
            self._upbit_tickers,
        )

    def stop(self) -> None:
        """스케줄러 중지. FastAPI lifespan shutdown 에서 호출."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("스케줄러 중지")

    # ── KIS 틱 ────────────────────────────────────────────────────

    async def _kis_tick(self) -> None:
        now = datetime.now(KST)
        # 장 마감(15:30) 이후 tick 무시 — cron hour="9-15" 이므로 15:30~15:59 구간 방어
        if now.hour == 15 and now.minute >= 30:
            return

        for symbol in list(self._kis_symbols):
            try:
                await self._process_kis_symbol(symbol)
            except Exception:
                logger.exception("[KIS][%s] 틱 처리 중 예외", symbol)

    async def _process_kis_symbol(self, symbol: str) -> None:
        price_data = await kis.get_price(symbol)
        current_price = float(price_data["current_price"])

        async with self._lock:
            position = self._positions.get(symbol)

        # ── 포지션 보유 중 ──
        if position is not None:
            exit_signal = RiskManager.tick(position, current_price)
            if exit_signal.should_exit:
                await self._execute_kis_sell(symbol, position, exit_signal.reason, current_price)
                return

            # MA/RSI 전략 청산 신호 확인 (5분봉)
            if position.strategy in ("moving_average", "rsi"):
                ohlcv_5min = await kis.get_minute_ohlcv(symbol, interval_min=5, count=300)
                sell = self._check_strategy_sell(ohlcv_5min, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_kis_sell(symbol, position, "STRATEGY_SELL", current_price)
            return

        # ── 포지션 없음: 매수 신호 탐색 (5분봉) ──
        ohlcv_5min = await kis.get_minute_ohlcv(symbol, interval_min=5, count=300)
        result, strategy_name = self._run_kis_strategies(ohlcv_5min, current_price)
        if result.is_buy:
            ml_ok = await self._check_ml_gate(symbol, ohlcv_5min, market="stock")
            if ml_ok:
                sim_log.push(symbol, f"{strategy_name}+5분봉ML — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_kis_buy(symbol, current_price, result, strategy_name)
            else:
                sim_log.push(symbol, f"{strategy_name} 신호 있으나 ML 차단 @ {current_price:,.0f}원", "INFO")
        else:
            sim_log.push(symbol, f"분석 완료 — 신호없음 @ {current_price:,.0f}원", "INFO")

    def _run_kis_strategies(
        self,
        ohlcv: list,
        current_price: float,
    ) -> tuple[StrategyResult, str]:
        """전략 우선순위: 변동성 돌파 → MA 크로스 → RSI."""
        result = vb_strategy.analyze(ohlcv, current_price, k=settings.volatility_k)
        if result.is_buy:
            return result, "volatility_breakout"

        result = ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if result.is_buy:
            return result, "moving_average"

        result = rsi_strategy.analyze(
            ohlcv,
            period=settings.rsi_period,
            oversold=settings.rsi_oversold,
            overbought=settings.rsi_overbought,
        )
        return result, "rsi"

    async def _check_ml_gate(self, symbol: str, ohlcv_5min: list, market: str = "coin") -> bool:
        """앙상블 ML 게이트.

        모든 학습된 에이전트가 최근 성과 기반 가중 투표.
        지금 잘 맞히는 전략이 자동으로 발언권 커짐 → 장세 변화 자동 적응.
        학습된 에이전트 없으면 초기 상태이므로 통과.
        """
        from backend.ml.agents import AGENTS, predict_ensemble

        trained = [a for a in AGENTS.values() if a.market == market and a._model is not None]
        if not trained:
            logger.info("[ML Gate][%s] 학습된 에이전트 없음 → 통과", symbol)
            return True

        try:
            signal, prob = await predict_ensemble(ohlcv_5min, ticker=symbol, market=market)
            approved = signal == "buy"
            n = len(trained)
            logger.info(
                "[ML Gate][%s] 앙상블(%d개) %s %.1f%% → %s",
                symbol, n, signal, prob * 100,
                "통과" if approved else "차단",
            )
            return approved
        except Exception:
            logger.warning("[ML Gate][%s] 앙상블 실패, 통과 처리", symbol)
            return True

    def _check_strategy_sell(
        self,
        ohlcv: list,
        strategy: str,
    ) -> StrategyResult | None:
        """보유 포지션의 전략 기반 청산 신호 확인 (MA 데드크로스 / RSI 과매수)."""
        if strategy == "moving_average":
            return ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if strategy == "rsi":
            return rsi_strategy.analyze(
                ohlcv,
                period=settings.rsi_period,
                oversold=settings.rsi_oversold,
                overbought=settings.rsi_overbought,
            )
        return None

    async def _execute_kis_buy(
        self,
        symbol: str,
        current_price: float,
        signal: StrategyResult,
        strategy_name: str,
    ) -> None:
        try:
            # TOCTOU 방지: 주문 전 포지션 재확인 (락 안에서)
            async with self._lock:
                if symbol in self._positions:
                    logger.warning("[KIS][%s] 중복매수 차단 (포지션 이미 존재)", symbol)
                    return

            balance = await kis.get_balance()
            cash = float(balance["cash"])
            qty = int(RiskManager.calculate_qty(cash, current_price))

            ok, reason = RiskManager.validate_order(cash, current_price, float(qty))
            if not ok:
                logger.warning("[KIS][%s] 매수 불가: %s", symbol, reason)
                return

            await kis.place_buy_order(symbol, qty)

            position = RiskManager.open_position(
                symbol=symbol,
                entry_price=current_price,
                qty=float(qty),
                strategy=strategy_name,
                is_crypto=False,
            )
            async with self._lock:
                self._positions[symbol] = position
            await upsert_position(symbol, "KIS", position)
            await insert_trade(symbol, "KIS", "BUY", float(qty), current_price, strategy_name, signal.reason)
            sim_log.push(symbol, f"매수 체결 {qty}주 @ {current_price:,.0f}원 | 손절 {position.stop_loss_price:,.0f}", "BUY")

            logger.info(
                "[KIS 매수] %s %d주 @ %,.0f원 | 전략: %s | %s",
                symbol, qty, current_price, strategy_name, signal.reason,
            )
            await telegram.notify_buy(symbol, float(qty), current_price, strategy_name, signal.reason)
        except Exception:
            logger.exception("[KIS][%s] 매수 실행 중 예외", symbol)

    async def _execute_kis_sell(
        self,
        symbol: str,
        position: Position,
        reason: str,
        current_price: float,
    ) -> None:
        try:
            qty = int(position.qty)
            await kis.place_sell_order(symbol, qty)

            profit_rate = (current_price - position.entry_price) / position.entry_price
            logger.info(
                "[KIS 매도] %s %d주 @ %,.0f원 | 수익률 %.2f%% | 사유: %s",
                symbol, qty, current_price, profit_rate * 100, reason,
            )
            await insert_trade(symbol, "KIS", "SELL", float(qty), current_price, position.strategy, reason, profit_rate)
            sim_log.push(symbol, f"매도 체결 {qty}주 @ {current_price:,.0f}원 | 수익률 {profit_rate*100:+.2f}% | {reason}", "SELL")
            await telegram.notify_sell(symbol, float(qty), current_price, profit_rate, reason)

            async with self._lock:
                self._positions.pop(symbol, None)
            await delete_position(symbol)
        except Exception:
            logger.exception("[KIS][%s] 매도 실행 중 예외", symbol)

    # ── KIS EOD 강제 매도 ────────────────────────────────────────

    async def _kis_eod_sell(self) -> None:
        """15:20 KST - 변동성 돌파 전략 포지션 강제 청산."""
        for symbol in list(self._kis_symbols):
            try:
                async with self._lock:
                    position = self._positions.get(symbol)

                # 변동성 돌파 전략으로 열린 포지션만 EOD 매도 대상
                if position is None or position.strategy != "volatility_breakout":
                    continue

                price_data = await kis.get_price(symbol)
                await self._execute_kis_sell(
                    symbol,
                    position,
                    "EOD_VOLATILITY_BREAKOUT",
                    float(price_data["current_price"]),
                )
            except Exception:
                logger.exception("[KIS][%s] EOD 매도 중 예외", symbol)

    # ── Upbit 틱 ─────────────────────────────────────────────────

    async def _upbit_tick(self) -> None:
        if not settings.upbit_access_key:
            return  # Upbit 키 미설정 시 전체 스킵

        for ticker in list(self._upbit_tickers):
            try:
                await self._process_upbit_ticker(ticker)
            except Exception:
                logger.exception("[Upbit][%s] 틱 처리 중 예외", ticker)

    async def _process_upbit_ticker(self, ticker: str) -> None:
        price_data = await upbit.get_price(ticker)
        current_price = float(price_data["current_price"])

        async with self._lock:
            position = self._positions.get(ticker)

        # ── 포지션 보유 중 ──
        if position is not None:
            exit_signal = RiskManager.tick(position, current_price)
            if exit_signal.should_exit:
                await self._execute_upbit_sell(ticker, position, exit_signal.reason, current_price)
                return

            # MA/RSI 전략 기반 청산 (5분봉)
            if position.strategy in ("moving_average", "rsi"):
                ohlcv_5min = await upbit.get_ohlcv(ticker, interval="minutes/5", count=300)
                sell = self._check_strategy_sell(ohlcv_5min, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_upbit_sell(ticker, position, "STRATEGY_SELL", current_price)
            return

        # ── 포지션 없음: 매수 신호 탐색 (5분봉) ──
        ohlcv_5min = await upbit.get_ohlcv(ticker, interval="minutes/5", count=300)
        result, strategy_name = self._run_upbit_strategies(ohlcv_5min, current_price)
        if result.is_buy:
            ml_ok = await self._check_ml_gate(ticker, ohlcv_5min, market="coin")
            if ml_ok:
                sim_log.push(ticker, f"{strategy_name}+5분봉ML — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_upbit_buy(ticker, current_price, result, strategy_name)
            else:
                sim_log.push(ticker, f"{strategy_name} 신호 있으나 ML 차단 @ {current_price:,.0f}원", "INFO")
        else:
            sim_log.push(ticker, f"분석 완료 — 신호없음 @ {current_price:,.0f}원", "INFO")

    def _run_upbit_strategies(
        self,
        ohlcv: list,
        current_price: float,
    ) -> tuple[StrategyResult, str]:
        """코인: MA 크로스 → RSI.

        변동성 돌파는 EOD 강제 매도가 필요한 전략이라 코인(24/7)에 부적합하므로 제외.
        """
        result = ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if result.is_buy:
            return result, "moving_average"

        result = rsi_strategy.analyze(
            ohlcv,
            period=settings.rsi_period,
            oversold=settings.rsi_oversold,
            overbought=settings.rsi_overbought,
        )
        return result, "rsi"

    async def _execute_upbit_buy(
        self,
        ticker: str,
        current_price: float,
        signal: StrategyResult,
        strategy_name: str,
    ) -> None:
        try:
            # TOCTOU 방지: 주문 전 포지션 재확인 (락 안에서)
            async with self._lock:
                if ticker in self._positions:
                    logger.warning("[Upbit][%s] 중복매수 차단 (포지션 이미 존재)", ticker)
                    return

            balance = await upbit.get_balance()
            krw = float(balance["krw"])
            amount_krw = krw * settings.max_position_ratio

            if amount_krw < 5_000:
                logger.warning("[Upbit][%s] 매수 불가: 잔액 부족 (%.0f원)", ticker, krw)
                return

            await upbit.place_buy_order(ticker, amount_krw)

            # qty는 추정값 (실제 체결 수량은 주문 체결 후 확정됨)
            qty = amount_krw / current_price
            position = RiskManager.open_position(
                symbol=ticker,
                entry_price=current_price,
                qty=qty,
                strategy=strategy_name,
                is_crypto=True,
            )
            async with self._lock:
                self._positions[ticker] = position
            await upsert_position(ticker, "UPBIT", position)
            await insert_trade(ticker, "UPBIT", "BUY", qty, current_price, strategy_name, signal.reason)
            sim_log.push(ticker, f"매수 체결 {qty:.6f} @ {current_price:,.0f}원 | 손절 {position.stop_loss_price:,.0f}", "BUY")

            logger.info(
                "[Upbit 매수] %s %.8f @ %.0f원 | 전략: %s | %s",
                ticker, qty, current_price, strategy_name, signal.reason,
            )
            await telegram.notify_buy(ticker, qty, current_price, strategy_name, signal.reason, is_crypto=True)
        except Exception:
            logger.exception("[Upbit][%s] 매수 실행 중 예외", ticker)

    async def _execute_upbit_sell(
        self,
        ticker: str,
        position: Position,
        reason: str,
        current_price: float,
    ) -> None:
        try:
            await upbit.place_sell_order(ticker, position.qty)

            profit_rate = (current_price - position.entry_price) / position.entry_price
            logger.info(
                "[Upbit 매도] %s %.8f @ %.0f원 | 수익률 %.2f%% | 사유: %s",
                ticker, position.qty, current_price, profit_rate * 100, reason,
            )
            await insert_trade(ticker, "UPBIT", "SELL", position.qty, current_price, position.strategy, reason, profit_rate)
            sim_log.push(ticker, f"매도 체결 {position.qty:.6f} @ {current_price:,.0f}원 | 수익률 {profit_rate*100:+.2f}% | {reason}", "SELL")
            await telegram.notify_sell(ticker, position.qty, current_price, profit_rate, reason, is_crypto=True)

            async with self._lock:
                self._positions.pop(ticker, None)
            await delete_position(ticker)
        except Exception:
            logger.exception("[Upbit][%s] 매도 실행 중 예외", ticker)


    async def _daily_report(self) -> None:
        """매일 08:00 — 주요 종목 백테스트 리포트 텔레그램 발송."""
        try:
            await send_daily_report()
        except Exception:
            logger.exception("[리포트] 일일 리포트 실행 중 예외")

    async def _agent_tick(self) -> None:
        """5분마다 — 20개 에이전트 가상 매매 실행.

        코인: 24/7 — 업비트 거래대금 상위 50개
        주식: 평일 장중(09:00~15:30)에 추가 — KIS 거래량 상위 50개
        """
        from backend.api import upbit as _upbit
        from backend.api import kis as _kis
        from backend.ml.agents import AGENTS
        from backend.db.database import DB_PATH
        import aiosqlite

        now = datetime.now(KST)
        is_market_open = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 0))
            and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
        )

        # 코인 티커 (24/7)
        try:
            coin_tickers = await _upbit.get_top_tickers(50)
        except Exception:
            coin_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]

        # 주식 종목 (장중에만 추가)
        stock_symbols: list[str] = []
        if is_market_open:
            try:
                stock_symbols = await _kis.get_volume_rank(50)
            except Exception:
                stock_symbols = list(self._kis_symbols)

        # ── 코인 인터벌별 OHLCV + 현재가 병렬 프리패치 ────────────
        coin_intervals = list({a.interval_str for a in AGENTS.values() if a.market == "coin"})

        # 보유 포지션 코인이 상위 50개에서 빠져도 반드시 청산 체크
        open_coin_pos = {
            t for a in AGENTS.values() if a.market == "coin"
            for t in a._positions.keys()
        }
        extra_tickers = sorted(open_coin_pos - set(coin_tickers))
        fetch_coin_tickers = coin_tickers + extra_tickers
        if extra_tickers:
            logger.info("[AgentTick] 상위50 이탈 포지션 %d개 추가 조회: %s", len(extra_tickers), extra_tickers)

        async def _fetch_coin_ohlcv(ticker: str, interval: str) -> tuple[str, str, list]:
            async with _OHLCV_SEM:
                try:
                    data = await _upbit.get_ohlcv(ticker, interval=interval, count=200)
                except Exception:
                    data = []
            return ticker, interval, data

        async def _fetch_coin_price(ticker: str) -> tuple[str, float]:
            try:
                pd_ = await _upbit.get_price(ticker)
                return ticker, float(pd_["current_price"])
            except Exception:
                return ticker, 0.0

        # 병렬 OHLCV 요청 (fetch_coin_tickers × 코인 인터벌 수)
        ohlcv_tasks = [
            _fetch_coin_ohlcv(t, iv) for t in fetch_coin_tickers for iv in coin_intervals
        ]
        ohlcv_results = await asyncio.gather(*ohlcv_tasks)
        ohlcv_cache: dict[str, list] = {
            f"{t}:{iv}": data for t, iv, data in ohlcv_results
        }

        # 배치 현재가 (상위50 + 포지션 보유 코인 — 1회 요청으로 429 방지)
        try:
            coin_prices = await _upbit.get_prices_batch(fetch_coin_tickers)
        except Exception:
            coin_prices = {}

        # 주식 OHLCV + 현재가도 병렬 프리패치
        stock_intervals = list({a.interval_min for a in AGENTS.values() if a.market == "stock"})
        stock_ohlcv_cache: dict[str, list] = {}
        stock_prices: dict[str, float] = {}

        if is_market_open and stock_symbols:
            async def _fetch_stock_ohlcv(sym: str, iv_min: int) -> tuple[str, int, list]:
                async with _OHLCV_SEM:
                    try:
                        data = await _kis.get_minute_ohlcv(sym, iv_min, count=200)
                    except Exception:
                        data = []
                return sym, iv_min, data

            async def _fetch_stock_price(sym: str) -> tuple[str, float]:
                try:
                    pd_ = await _kis.get_price(sym)
                    return sym, float(pd_["current_price"])
                except Exception:
                    return sym, 0.0

            s_ohlcv_tasks = [
                _fetch_stock_ohlcv(s, iv) for s in stock_symbols for iv in stock_intervals
            ]
            s_results = await asyncio.gather(*s_ohlcv_tasks)
            stock_ohlcv_cache = {f"{s}:{iv}": data for s, iv, data in s_results}

            sp_results = await asyncio.gather(*[_fetch_stock_price(s) for s in stock_symbols])
            stock_prices = {s: p for s, p in sp_results if p > 0}

        async with aiosqlite.connect(DB_PATH) as db:
            for agent in AGENTS.values():
                try:
                    if agent.market == "coin":
                        # ── 코인 전담 (AI01~AI10): 24/7 ──────────────
                        agent.update_position_values(coin_prices)
                        for ticker in fetch_coin_tickers:
                            ohlcv = ohlcv_cache.get(f"{ticker}:{agent.interval_str}", [])
                            if not ohlcv:
                                continue
                            if agent._model is None and not agent.load_model():
                                # 학습용 1주일치 데이터 별도 조회 (코인: 2016봉 이상)
                                train_count = 2000 if agent.interval_min <= 5 else 700
                                try:
                                    train_ohlcv = await _upbit.get_ohlcv(
                                        ticker, interval=agent.interval_str, count=train_count
                                    )
                                except Exception:
                                    train_ohlcv = ohlcv
                                trained = await asyncio.to_thread(agent.train, train_ohlcv)
                                if not trained:
                                    continue
                            price = coin_prices.get(ticker, 0.0)
                            if price <= 0:
                                continue
                            signal, prob = agent.predict(ohlcv)
                            await self._agent_execute(db, agent, ticker, signal, prob, price)

                    else:
                        # ── 주식 전담 (AI11~AI20): 장중만 ────────────
                        if not is_market_open:
                            continue
                        agent.update_position_values(stock_prices)
                        for symbol in stock_symbols:
                            ohlcv = stock_ohlcv_cache.get(f"{symbol}:{agent.interval_min}", [])
                            if not ohlcv:
                                continue
                            if agent._model is None and not agent.load_model():
                                # 학습용 1주일치 데이터 별도 조회 (주식 5분봉: 390봉/주)
                                train_count = 500 if agent.interval_min <= 5 else 200
                                try:
                                    train_ohlcv = await _kis.get_minute_ohlcv(
                                        symbol, agent.interval_min, count=train_count
                                    )
                                except Exception:
                                    train_ohlcv = ohlcv
                                trained = await asyncio.to_thread(agent.train, train_ohlcv)
                                if not trained:
                                    continue
                            price = stock_prices.get(symbol, 0.0)
                            if price <= 0:
                                continue
                            signal, prob = agent.predict(ohlcv)
                            await self._agent_execute(db, agent, symbol, signal, prob, price)

                    # agent_stats upsert
                    await db.execute(
                        """INSERT INTO agent_stats
                           (agent_id, interval_min, label_threshold, buy_threshold, feature_set,
                            total_trades, win_trades, win_rate, total_return, current_balance, is_champion, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(agent_id) DO UPDATE SET
                             total_trades=excluded.total_trades, win_trades=excluded.win_trades,
                             win_rate=excluded.win_rate, total_return=excluded.total_return,
                             current_balance=excluded.current_balance, is_champion=excluded.is_champion,
                             updated_at=excluded.updated_at""",
                        (
                            agent.agent_id, agent.interval_min, agent.label_threshold,
                            agent.buy_threshold, agent.feature_set,
                            agent.total_trades, agent.win_trades,
                            round(agent.win_rate, 4), round(agent.total_return, 4),
                            round(agent._balance, 2), int(agent.is_champion),
                            now.strftime("%Y-%m-%dT%H:%M:%S"),
                        ),
                    )
                except Exception:
                    logger.exception("[Agent][%s] 틱 처리 실패", agent.agent_id)

            await db.commit()

        from backend.ml.agents import refresh_champion_flags
        refresh_champion_flags()

    async def _agent_execute(self, db, agent, symbol: str, signal: str, prob: float, price: float) -> None:
        """에이전트 단일 종목 가상 매수/매도 실행 + DB 기록."""
        STOP_LOSS = -0.03  # -3% 손절 — 항상 적용

        # ── 보유 포지션 익절/손절 우선 체크 ─────────────────────────
        if symbol in agent._positions:
            pos = agent._positions[symbol]

            # 보유 시간 계산
            try:
                held_min = (datetime.now() - datetime.fromisoformat(pos.entered_at)).total_seconds() / 60
            except Exception:
                held_min = 0

            # 가격 데이터 없이 24시간 이상 묶인 포지션 → 마지막 알려진 가격으로 강제 청산
            if price <= 0:
                if held_min >= 1440:  # 24시간
                    last_price = agent._last_position_values.get(symbol, 0) / pos.qty if pos.qty > 0 else 0
                    if last_price > 0:
                        signal = "sell"
                        price = last_price
                        sim_log.push(agent.agent_id, f"[타임아웃강제청산] {symbol} 24h 가격데이터 없음 @ {price:,.0f}원", "SELL")
                return  # 가격 없으면 나머지 로직 스킵

            unreal = (price - pos.entry_price) / pos.entry_price

            # 시간 경과 하향 ROI: 오래 묶일수록 익절 기준 낮춤 (자금 효율)
            if held_min < 30:
                take_profit = 0.05    # 30분 미만 → +5% 익절
            elif held_min < 60:
                take_profit = 0.03    # 30~60분 → +3% 익절
            else:
                take_profit = 0.015   # 60분 이상 → +1.5% 익절 (자금 묶임 방지)

            if unreal >= take_profit:
                signal = "sell"
                sim_log.push(agent.agent_id, f"[익절] {symbol} +{unreal*100:.1f}% ({held_min:.0f}분) @ {price:,.0f}원", "BUY")
            elif unreal <= STOP_LOSS:
                signal = "sell"
                sim_log.push(agent.agent_id, f"[손절] {symbol} {unreal*100:.1f}% @ {price:,.0f}원", "SELL")

        if signal == "buy":
            trade = agent.virtual_buy(symbol, price)
            if trade:
                await db.execute(
                    "INSERT INTO agent_trades (agent_id, ticker, action, price, qty, entry_price, profit_rate, balance) VALUES (?,?,?,?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty, trade.entry_price, trade.profit_rate, trade.balance),
                )
                sim_log.push(trade.agent_id, f"[가상매수] {symbol} @ {price:,.0f}원 (확률 {prob:.1%})", "BUY")

        elif signal == "sell" and symbol in agent._positions:
            trade = agent.virtual_sell(symbol, price)
            if trade:
                pct = (trade.profit_rate or 0) * 100
                await db.execute(
                    "INSERT INTO agent_trades (agent_id, ticker, action, price, qty, entry_price, profit_rate, balance) VALUES (?,?,?,?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty, trade.entry_price, trade.profit_rate, trade.balance),
                )
                level = "BUY" if (trade.profit_rate or 0) > 0 else "SELL"
                sim_log.push(trade.agent_id, f"[가상매도] {symbol} @ {price:,.0f}원 | {pct:+.2f}%", level)

    async def _daily_retrain(self) -> None:
        """매일 03:00 KST — 전 에이전트 최신 데이터로 재학습.

        코인: BTC·ETH·XRP·SOL 4개 순차 시도, 첫 성공 데이터 사용 (2000봉 = 약 7일치)
        주식: 삼성전자·SK하이닉스·NAVER 3개 순차 시도 (500봉 = 약 2일치)
        에이전트별로 피처셋이 다르므로 같은 데이터로 학습해도 모델이 달라짐.
        """
        from backend.api import upbit as _upbit, kis as _kis
        from backend.ml.agents import AGENTS

        logger.info("[Retrain] 전 에이전트 일일 재학습 시작")

        # ── 학습 데이터 수집 ─────────────────────────────────────
        COIN_TICKERS  = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
        STOCK_SYMBOLS = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER

        coin_ohlcv: list[dict] = []
        for ticker in COIN_TICKERS:
            try:
                coin_ohlcv = await _upbit.get_ohlcv(ticker, interval="minutes/5", count=2000)
                if len(coin_ohlcv) >= 500:
                    logger.info("[Retrain] 코인 학습 데이터: %s (%d봉)", ticker, len(coin_ohlcv))
                    break
            except Exception:
                continue

        stock_ohlcv: list[dict] = []
        for symbol in STOCK_SYMBOLS:
            try:
                stock_ohlcv = await _kis.get_minute_ohlcv(symbol, 5, count=500)
                if len(stock_ohlcv) >= 100:
                    logger.info("[Retrain] 주식 학습 데이터: %s (%d봉)", symbol, len(stock_ohlcv))
                    break
            except Exception:
                continue

        # ── 에이전트별 재학습 ────────────────────────────────────
        success = 0
        for agent in AGENTS.values():
            data = coin_ohlcv if agent.market == "coin" else stock_ohlcv
            if not data:
                logger.warning("[Retrain][%s] 학습 데이터 없음, 스킵", agent.agent_id)
                continue
            try:
                trained = await asyncio.to_thread(agent.train, data)
                if trained:
                    success += 1
                    logger.info("[Retrain][%s] 재학습 완료", agent.agent_id)
                else:
                    logger.warning("[Retrain][%s] 재학습 실패 (데이터 부족 또는 레이블 편향)", agent.agent_id)
            except Exception:
                logger.exception("[Retrain][%s] 재학습 중 예외", agent.agent_id)

        logger.info("[Retrain] 완료: %d/20 에이전트 재학습", success)
        try:
            await telegram.notify_message(
                f"🔄 <b>AI 일일 재학습 완료</b>\n"
                f"성공: {success}/20 에이전트\n"
                f"코인 데이터: {len(coin_ohlcv)}봉 | 주식 데이터: {len(stock_ohlcv)}봉"
            )
        except Exception:
            pass

    def get_agents_snapshot(self) -> list[dict]:
        """에이전트 상태 스냅샷 반환 (챔피언 먼저, 이후 코인/주식 순)."""
        from backend.ml.agents import AGENTS
        agents = list(AGENTS.values())
        coins  = sorted([a for a in agents if a.market == "coin"],  key=lambda a: (not a.is_champion, -a.win_rate))
        stocks = sorted([a for a in agents if a.market == "stock"], key=lambda a: (not a.is_champion, -a.win_rate))
        return [a.to_dict() for a in coins + stocks]


# 싱글톤 인스턴스 — FastAPI main.py 에서 import 해서 사용
scheduler = TradingScheduler()
