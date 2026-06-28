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

# ── 주요 매크로 이벤트 (KST): 발표 ±2시간 코인 신규 매수 차단 ─────
_MACRO_EVENTS_KST = [
    # FOMC 2026 (미 동부 14:00 = KST 03:00/04:00)
    "2026-07-29 04:00", "2026-09-17 04:00", "2026-10-29 04:00", "2026-12-10 04:00",
    # US CPI 2026 (미 동부 08:30 = KST 21:30/22:30)
    "2026-07-15 22:30", "2026-08-12 22:30", "2026-09-11 22:30",
    "2026-10-14 22:30", "2026-11-12 22:30", "2026-12-10 22:30",
    # US NFP 비농업고용 2026 (보통 첫째 금요일)
    "2026-07-02 22:30", "2026-08-07 22:30", "2026-09-04 22:30",
    "2026-10-02 22:30", "2026-11-06 22:30", "2026-12-04 22:30",
]


def _is_high_risk_window() -> bool:
    """현재 KST 시각이 주요 매크로 이벤트 발표 ±2시간 이내인지 확인."""
    now = datetime.now(KST).replace(tzinfo=None)
    for es in _MACRO_EVENTS_KST:
        ev = datetime.strptime(es, "%Y-%m-%d %H:%M")
        if abs((now - ev).total_seconds()) <= 7200:
            return True
    return False
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
        self._use_atr_risk: bool = False  # 에이전트 검증 완료 시 True → 실매매도 ATR 동적 손익 적용
        self._btc_oi_hist:    list[dict] = []  # BTC 선물 OI 히스토리 캐시 (코인 에이전트 전체 공유)
        self._btc_taker_hist: list[dict] = []  # BTC 선물 Taker 비율 히스토리 캐시
        self._btc_ohlcv_cache: list[dict] = []  # BTC 5분봉 OHLCV 캐시 (ML 게이트 BTC 상관관계용)
        # DCA 분할매수 상태: {symbol: {"stage": 1~3, "total_budget": float}}
        # stage 1=1차완료, 2=2차완료, 3=3차완료(전량매수)
        self._dca_state: dict[str, dict] = {}

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
        # ── 에이전트 stats 복구 (잔액·거래수·승률·is_active·buy_threshold) ──
        await self._restore_agent_stats()

    async def _restore_agent_stats(self) -> None:
        """서버 재시작 시 DB에서 에이전트 상태 복구 (잔액·거래수·is_active·buy_threshold)."""
        from backend.ml.agents import AGENTS
        from backend.db.database import DB_PATH
        import aiosqlite

        try:
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM agent_stats") as cur:
                    stats_rows = {r["agent_id"]: dict(r) async for r in cur}

                async with db.execute(
                    "SELECT * FROM agent_positions"
                ) as cur:
                    pos_rows: dict[str, list] = {}
                    async for r in cur:
                        pos_rows.setdefault(r["agent_id"], []).append(dict(r))

                async with db.execute(
                    "SELECT * FROM agent_trades ORDER BY traded_at DESC"
                ) as cur:
                    trade_rows: dict[str, list] = {}
                    async for r in cur:
                        aid = r["agent_id"]
                        if len(trade_rows.get(aid, [])) < 30:
                            trade_rows.setdefault(aid, []).append(dict(r))

            for agent in AGENTS.values():
                stats = stats_rows.get(agent.agent_id)
                if stats:
                    agent.restore_from_db(
                        stats,
                        pos_rows.get(agent.agent_id, []),
                        trade_rows.get(agent.agent_id, []),
                    )
            logger.info("[복구] 에이전트 stats %d개 복구 완료", len(stats_rows))
        except Exception:
            logger.exception("[복구] 에이전트 stats 복구 실패")

        # 재시작 후에도 ATR 전환 조건 재평가 (인메모리 플래그 복구)
        await self._check_atr_upgrade()

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
        # 전 에이전트 일일 재학습: 매일 18:00 (장 마감 후, 코인/주식 모두 안전)
        self._scheduler.add_job(
            self._daily_retrain,
            CronTrigger(hour=18, minute=0, timezone=KST),
            id="daily_retrain",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 일일 현황 레포트: 매일 22:00 텔레그램 전송
        self._scheduler.add_job(
            self._send_daily_agent_report,
            CronTrigger(hour=22, minute=0, timezone=KST),
            id="daily_agent_report",
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

            # DCA 2/3차 추가매수 체크
            dca = self._dca_state.get(symbol)
            if dca:
                drop = (current_price - position.entry_price) / position.entry_price
                if dca["stage"] == 1 and drop <= -0.015:
                    await self._execute_kis_dca_add(symbol, current_price, position, 2, 0.35)
                elif dca["stage"] == 2 and drop <= -0.030:
                    await self._execute_kis_dca_add(symbol, current_price, position, 3, 0.25)
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
            btc_ohlcv  = self._btc_ohlcv_cache if market == "coin" else None
            oi_hist    = self._btc_oi_hist      if market == "coin" else None
            taker_hist = self._btc_taker_hist   if market == "coin" else None
            signal, prob = await predict_ensemble(
                ohlcv_5min, ticker=symbol, market=market,
                btc_ohlcv=btc_ohlcv,
                oi_hist=oi_hist, taker_hist=taker_hist,
            )
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
            total_qty = int(RiskManager.calculate_qty(cash, current_price))
            qty = max(1, int(total_qty * 0.40))  # DCA 1차: 총 수량의 40%

            ok, reason = RiskManager.validate_order(cash, current_price, float(qty))
            if not ok:
                logger.warning("[KIS][%s] 매수 불가: %s", symbol, reason)
                return

            await kis.place_buy_order(symbol, qty)
            self._dca_state[symbol] = {"stage": 1, "total_qty": total_qty, "bought_qty": qty}

            # ATR 기반 동적 손익 (에이전트 검증 완료 후 자동 전환)
            sl_rate = tp_rate = None
            if self._use_atr_risk:
                from backend.ml.agents import AGENTS
                atr_vals = [a._last_atr_pct for a in AGENTS.values() if a.market == "stock" and a._last_atr_pct > 0.001]
                if atr_vals:
                    atr = sum(atr_vals) / len(atr_vals)
                    sl_rate = -(atr * 1.5)
                    tp_rate = atr * 2.0

            position = RiskManager.open_position(
                symbol=symbol,
                entry_price=current_price,
                qty=float(qty),
                strategy=strategy_name,
                is_crypto=False,
                stop_loss_rate=sl_rate,
                take_profit_rate=tp_rate,
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
            self._dca_state.pop(symbol, None)
            await delete_position(symbol)
        except Exception:
            logger.exception("[KIS][%s] 매도 실행 중 예외", symbol)

    async def _execute_kis_dca_add(
        self,
        symbol: str,
        current_price: float,
        position: Position,
        next_stage: int,
        qty_ratio: float,
    ) -> None:
        """KIS DCA 추가매수 (2차/3차)."""
        try:
            dca = self._dca_state.get(symbol)
            if not dca:
                return
            add_qty = max(1, int(dca["total_qty"] * qty_ratio))
            if add_qty < 1:
                return

            await kis.place_buy_order(symbol, add_qty)

            total_cost = position.entry_price * position.qty + current_price * add_qty
            new_qty = position.qty + add_qty
            new_avg = total_cost / new_qty
            position.entry_price = new_avg
            position.qty = new_qty
            position.stop_loss_price = new_avg * (1 + settings.stop_loss_rate)
            position.take_profit_price = new_avg * (1 + settings.take_profit_rate)
            async with self._lock:
                self._positions[symbol] = position
            await upsert_position(symbol, "KIS", position)
            await insert_trade(symbol, "KIS", "BUY", float(add_qty), current_price, position.strategy, f"DCA {next_stage}차")
            dca["stage"] = next_stage
            dca["bought_qty"] = int(dca.get("bought_qty", 0)) + add_qty
            sim_log.push(symbol, f"DCA {next_stage}차 {add_qty}주 추가매수 @ {current_price:,.0f}원 | 평균단가 {new_avg:,.0f}원", "BUY")
            logger.info("[KIS DCA %d차] %s %d주 @ %,.0f원 | 평균단가 %,.0f원", next_stage, symbol, add_qty, current_price, new_avg)
            await telegram.notify_buy(symbol, float(add_qty), current_price, position.strategy, f"DCA {next_stage}차 추가매수")
        except Exception:
            logger.exception("[KIS][%s] DCA %d차 추가매수 예외", symbol, next_stage)

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
        if settings.is_paper:
            return  # PAPER 모드: 업비트 모의투자 API 없음 → 실주문 방지

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

            # MA/RSI 전략 기반 청산 (5분봉) — DCA보다 먼저 체크 (청산 시 DCA 스킵)
            if position.strategy in ("moving_average", "rsi"):
                ohlcv_5min = await upbit.get_ohlcv(ticker, interval="minutes/5", count=300)
                sell = self._check_strategy_sell(ohlcv_5min, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_upbit_sell(ticker, position, "STRATEGY_SELL", current_price)
                    return

            # DCA 2/3차 추가 매수 체크
            dca = self._dca_state.get(ticker)
            if dca:
                drop = (current_price - position.entry_price) / position.entry_price
                if dca["stage"] == 1 and drop <= -0.015:
                    await self._execute_upbit_dca_add(ticker, current_price, position, 2, 0.35)
                elif dca["stage"] == 2 and drop <= -0.030:
                    await self._execute_upbit_dca_add(ticker, current_price, position, 3, 0.25)
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
            total_budget = krw * settings.max_position_ratio
            amount_krw = total_budget * 0.40  # DCA 1차: 총 예산의 40%

            if amount_krw < 5_000:
                logger.warning("[Upbit][%s] 매수 불가: 잔액 부족 (%.0f원)", ticker, krw)
                return

            await upbit.place_buy_order(ticker, amount_krw)

            # qty는 추정값 (실제 체결 수량은 주문 체결 후 확정됨)
            qty = amount_krw / current_price
            self._dca_state[ticker] = {"stage": 1, "total_budget": total_budget}

            # ATR 기반 동적 손익 (에이전트 검증 완료 후 자동 전환)
            sl_rate = tp_rate = None
            if self._use_atr_risk:
                from backend.ml.agents import AGENTS
                atr_vals = [a._last_atr_pct for a in AGENTS.values() if a.market == "coin" and a._last_atr_pct > 0.001]
                if atr_vals:
                    atr = sum(atr_vals) / len(atr_vals)
                    sl_rate = -(atr * 1.5)
                    tp_rate = atr * 2.0

            position = RiskManager.open_position(
                symbol=ticker,
                entry_price=current_price,
                qty=qty,
                strategy=strategy_name,
                is_crypto=True,
                stop_loss_rate=sl_rate,
                take_profit_rate=tp_rate,
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

    async def _execute_upbit_dca_add(
        self,
        ticker: str,
        current_price: float,
        position: Position,
        next_stage: int,
        budget_ratio: float,
    ) -> None:
        """DCA 추가 매수 (2차 35% / 3차 25%). 평균 단가 갱신 후 DB 업데이트."""
        try:
            dca = self._dca_state.get(ticker)
            if not dca:
                return
            add_krw = dca["total_budget"] * budget_ratio
            if add_krw < 5_000:
                return

            await upbit.place_buy_order(ticker, add_krw)
            add_qty = add_krw / current_price

            # 평균 단가·수량 갱신
            total_cost = position.entry_price * position.qty + current_price * add_qty
            new_qty    = position.qty + add_qty
            new_avg    = total_cost / new_qty
            position.entry_price        = new_avg
            position.qty                = new_qty
            position.stop_loss_price    = new_avg * (1 + settings.stop_loss_rate)
            position.take_profit_price  = new_avg * (1 + settings.take_profit_rate)

            async with self._lock:
                self._positions[ticker] = position
            await upsert_position(ticker, "UPBIT", position)
            await insert_trade(ticker, "UPBIT", "BUY", add_qty, current_price, position.strategy, f"DCA {next_stage}차")

            dca["stage"] = next_stage
            sim_log.push(ticker, f"DCA {next_stage}차 {add_qty:.6f} @ {current_price:,.0f}원 | 평균단가 {new_avg:,.0f}원", "BUY")
            await telegram.notify_buy(ticker, add_qty, current_price, position.strategy, f"DCA {next_stage}차 추가매수 | 평균단가 {new_avg:,.0f}원", is_crypto=True)
        except Exception:
            logger.exception("[Upbit][%s] DCA %d차 추가매수 실패", ticker, next_stage)

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
            self._dca_state.pop(ticker, None)
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

        # BTC 선물 OI + Taker 히스토리 (코인 에이전트 전체 공유, 5분 TTL)
        from backend.api.binance import get_open_interest_hist, get_taker_ratio_hist
        try:
            btc_oi_hist, btc_taker_hist = await asyncio.gather(
                get_open_interest_hist("BTCUSDT", period="5m", limit=200),
                get_taker_ratio_hist("BTCUSDT", period="5m", limit=200),
            )
            self._btc_oi_hist    = btc_oi_hist
            self._btc_taker_hist = btc_taker_hist
        except Exception:
            btc_oi_hist    = self._btc_oi_hist
            btc_taker_hist = self._btc_taker_hist

        # 주식 OHLCV + 현재가도 병렬 프리패치
        stock_intervals = list({a.interval_min for a in AGENTS.values() if a.market == "stock"})
        stock_ohlcv_cache: dict[str, list] = {}
        stock_prices: dict[str, float] = {}

        # KOSPI 지수 5분봉 (주식 에이전트 상대강도 피처용)
        kospi_ohlcv: list[dict] = []
        if is_market_open and stock_symbols:
            try:
                kospi_ohlcv = await _kis.get_minute_ohlcv("0001", 5, count=200)
            except Exception:
                kospi_ohlcv = []

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

        # BTC OHLCV 캐시 갱신 (ML 게이트 btc_corr_20 피처용)
        _fresh_btc = ohlcv_cache.get("KRW-BTC:minutes/5", [])
        if _fresh_btc:
            self._btc_ohlcv_cache = _fresh_btc

        async with aiosqlite.connect(DB_PATH) as db:
            for agent in AGENTS.values():
                try:
                    if agent.market == "coin":
                        # ── 코인 전담 (AI01~AI10): 24/7 ──────────────
                        agent.update_position_values(coin_prices)
                        # BTC OHLCV (비BTC 코인의 상관관계 피처용)
                        btc_ohlcv_cache = ohlcv_cache.get("KRW-BTC:minutes/5", [])
                        for ticker in fetch_coin_tickers:
                            ohlcv = ohlcv_cache.get(f"{ticker}:{agent.interval_str}", [])
                            if not ohlcv:
                                continue
                            # BTC 자신은 BTC 상관관계 불필요
                            _btc_ref = btc_ohlcv_cache if ticker != "KRW-BTC" else None
                            if agent._model is None and not agent.load_model():
                                train_count = 2000 if agent.interval_min <= 5 else 700
                                try:
                                    train_ohlcv = await _upbit.get_ohlcv(
                                        ticker, interval=agent.interval_str, count=train_count
                                    )
                                    # 학습용 BTC 데이터도 동일 기간으로 조회
                                    _btc_train = None
                                    if ticker != "KRW-BTC":
                                        try:
                                            _btc_train = await _upbit.get_ohlcv(
                                                "KRW-BTC", interval=agent.interval_str, count=train_count
                                            )
                                        except Exception:
                                            pass
                                except Exception:
                                    train_ohlcv = ohlcv
                                    _btc_train = _btc_ref
                                _oi_ref    = btc_oi_hist    if btc_oi_hist    else None
                                _taker_ref = btc_taker_hist if btc_taker_hist else None
                                trained = await asyncio.to_thread(
                                    agent.train, train_ohlcv,
                                    agent._cached_funding_rates or None,
                                    _btc_train, None,
                                    _oi_ref, _taker_ref,
                                )
                                if not trained:
                                    continue
                            price = coin_prices.get(ticker, 0.0)
                            if price <= 0:
                                continue
                            _oi_ref    = btc_oi_hist    if btc_oi_hist    else None
                            _taker_ref = btc_taker_hist if btc_taker_hist else None
                            signal, prob = agent.predict(ohlcv, btc_ohlcv=_btc_ref, oi_hist=_oi_ref, taker_hist=_taker_ref)
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
                                train_count = 500 if agent.interval_min <= 5 else 200
                                try:
                                    train_ohlcv = await _kis.get_minute_ohlcv(
                                        symbol, agent.interval_min, count=train_count
                                    )
                                    _kospi_train = await _kis.get_minute_ohlcv("0001", 5, count=train_count)
                                except Exception:
                                    train_ohlcv = ohlcv
                                    _kospi_train = kospi_ohlcv
                                trained = await asyncio.to_thread(
                                    agent.train, train_ohlcv, None, None, _kospi_train
                                )
                                if not trained:
                                    continue
                            price = stock_prices.get(symbol, 0.0)
                            if price <= 0:
                                continue
                            signal, prob = agent.predict(ohlcv, kospi_ohlcv=kospi_ohlcv or None)
                            await self._agent_execute(db, agent, symbol, signal, prob, price)

                    # agent_stats upsert
                    await db.execute(
                        """INSERT INTO agent_stats
                           (agent_id, interval_min, label_threshold, buy_threshold, feature_set,
                            total_trades, win_trades, win_rate, total_return, current_balance,
                            is_champion, is_active, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(agent_id) DO UPDATE SET
                             total_trades=excluded.total_trades, win_trades=excluded.win_trades,
                             win_rate=excluded.win_rate, total_return=excluded.total_return,
                             current_balance=excluded.current_balance, is_champion=excluded.is_champion,
                             buy_threshold=excluded.buy_threshold, is_active=excluded.is_active,
                             updated_at=excluded.updated_at""",
                        (
                            agent.agent_id, agent.interval_min, agent.label_threshold,
                            round(agent.buy_threshold, 4), agent.feature_set,
                            agent.total_trades, agent.win_trades,
                            round(agent.win_rate, 4), round(agent.total_return, 4),
                            round(agent._balance, 2), int(agent.is_champion),
                            int(agent.is_active),
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
        # ── ATR 기반 동적 손익 ────────────────────────────────────────
        # ATR이 유효하면(>0.1%) 시장 변동성에 자동 적응, 없으면 고정값 폴백
        atr_pct = agent._last_atr_pct
        if atr_pct > 0.001:
            STOP_LOSS   = -(atr_pct * 1.5)          # ATR × 1.5 손절
            tp_base     = atr_pct * 2.0              # ATR × 2.0 기준 익절
        else:
            STOP_LOSS   = -0.03
            tp_base     = 0.05

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
                    if pos.qty <= 0:
                        return  # qty 0이면 처리 불가 (ZeroDivisionError 방지)
                    last_price = agent._last_position_values.get(symbol, 0) / pos.qty
                    if last_price > 0:
                        signal = "sell"
                        price = last_price
                        sim_log.push(agent.agent_id, f"[타임아웃강제청산] {symbol} 24h 가격데이터 없음 @ {price:,.0f}원", "SELL")
                        # 아래 sell 로직으로 계속 진행 (return 없음)
                    else:
                        return  # 마지막 가격도 없으면 스킵
                else:
                    return  # 24시간 미만이면 스킵

            unreal = (price - pos.entry_price) / pos.entry_price

            # 시간 경과 하향 ROI: ATR 기준으로 스케일, 오래 묶일수록 익절 기준 낮춤
            if held_min < 30:
                take_profit = tp_base
            elif held_min < 60:
                take_profit = max(tp_base * 0.6, 0.015)
            else:
                take_profit = max(tp_base * 0.3, 0.01)

            if unreal >= take_profit:
                signal = "sell"
                sim_log.push(agent.agent_id, f"[익절] {symbol} +{unreal*100:.1f}% ({held_min:.0f}분) ATR손익({tp_base*100:.1f}%/{STOP_LOSS*100:.1f}%) @ {price:,.0f}원", "SELL")
            elif unreal <= STOP_LOSS:
                signal = "sell"
                sim_log.push(agent.agent_id, f"[손절] {symbol} {unreal*100:.1f}% (ATR기준 {STOP_LOSS*100:.1f}%) @ {price:,.0f}원", "SELL")

        if signal == "buy":
            # 코인: 매크로 이벤트 발표 ±2시간은 변동성 폭발 위험 → 신규 매수 차단
            if agent.market == "coin" and _is_high_risk_window():
                sim_log.push(agent.agent_id, f"[이벤트회피] {symbol} 매수 차단 (매크로 발표 ±2시간)", "INFO")
                return
            # DCA: 확률 강도에 비례한 진입 비율 (약한 신호는 50%, 강한 신호는 100%)
            _gap = prob - agent.buy_threshold
            _portion = 1.0 if _gap >= 0.08 else (0.7 if _gap >= 0.04 else 0.5)
            trade = agent.virtual_buy(symbol, price, portion=_portion)
            if trade:
                await db.execute(
                    "INSERT INTO agent_trades (agent_id, ticker, action, price, qty, entry_price, profit_rate, balance) VALUES (?,?,?,?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty, trade.entry_price, trade.profit_rate, trade.balance),
                )
                await db.execute(
                    "INSERT OR REPLACE INTO agent_positions (agent_id, ticker, entry_price, qty, entered_at) VALUES (?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.price, trade.qty, trade.traded_at),
                )
                sim_log.push(trade.agent_id, f"[가상매수] {symbol} @ {price:,.0f}원 (확률 {prob:.1%} / 진입{_portion*100:.0f}%)", "BUY")

        elif signal == "sell" and symbol in agent._positions:
            trade = agent.virtual_sell(symbol, price)
            if trade:
                pct = (trade.profit_rate or 0) * 100
                await db.execute(
                    "INSERT INTO agent_trades (agent_id, ticker, action, price, qty, entry_price, profit_rate, balance) VALUES (?,?,?,?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty, trade.entry_price, trade.profit_rate, trade.balance),
                )
                # 포지션 영속성: 청산 시 제거
                await db.execute(
                    "DELETE FROM agent_positions WHERE agent_id=? AND ticker=?",
                    (trade.agent_id, trade.ticker),
                )
                level = "BUY" if (trade.profit_rate or 0) > 0 else "SELL"
                sim_log.push(trade.agent_id, f"[가상매도] {symbol} @ {price:,.0f}원 | {pct:+.2f}%", level)

    async def _daily_retrain(self) -> None:
        """매일 18:00 KST — 전 에이전트 최신 데이터로 재학습.

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

        # ── 코인 펀딩비 히스토리 수집 (BTC 대표값, 50봉 = 약 16일치) ──
        from backend.api.binance import get_historical_funding_rates
        coin_funding: list[dict] = []
        try:
            coin_funding = await get_historical_funding_rates("BTCUSDT", limit=50)
            logger.info("[Retrain] 펀딩비 히스토리: %d개", len(coin_funding))
        except Exception:
            logger.warning("[Retrain] 펀딩비 히스토리 수집 실패, 펀딩비=0으로 학습")

        # ── BTC 학습용 데이터 (코인 에이전트 상관관계 피처용) ────────
        btc_train_ohlcv: list[dict] = []
        try:
            btc_train_ohlcv = await _upbit.get_ohlcv("KRW-BTC", interval="minutes/5", count=2000)
            logger.info("[Retrain] BTC 학습 데이터: %d봉", len(btc_train_ohlcv))
        except Exception:
            logger.warning("[Retrain] BTC 상관관계 학습 데이터 수집 실패")

        # ── KOSPI 학습용 데이터 (주식 에이전트 상대강도 피처용) ──────
        kospi_train_ohlcv: list[dict] = []
        try:
            kospi_train_ohlcv = await _kis.get_minute_ohlcv("0001", 5, count=500)
            logger.info("[Retrain] KOSPI 학습 데이터: %d봉", len(kospi_train_ohlcv))
        except Exception:
            logger.warning("[Retrain] KOSPI 학습 데이터 수집 실패")

        # ── BTC 선물 OI + Taker 히스토리 (코인 에이전트 학습용) ─────
        from backend.api.binance import get_open_interest_hist, get_taker_ratio_hist
        btc_oi_train:    list[dict] = []
        btc_taker_train: list[dict] = []
        try:
            btc_oi_train, btc_taker_train = await asyncio.gather(
                get_open_interest_hist("BTCUSDT", period="5m", limit=500),
                get_taker_ratio_hist("BTCUSDT", period="5m", limit=500),
            )
            logger.info("[Retrain] BTC OI: %d개, Taker비율: %d개", len(btc_oi_train), len(btc_taker_train))
        except Exception:
            logger.warning("[Retrain] BTC OI/Taker 데이터 수집 실패")

        # ── 에이전트 거래 결과 조회 (수익/손실 패턴 흡수용) ─────────
        import aiosqlite
        from backend.db.database import DB_PATH
        agent_trade_results: dict[str, list[dict]] = {}
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            for agent in AGENTS.values():
                async with db.execute(
                    """
                    SELECT b.traded_at AS buy_at, s.profit_rate
                    FROM agent_trades b
                    INNER JOIN agent_trades s
                        ON  s.agent_id = b.agent_id
                        AND s.ticker   = b.ticker
                        AND s.action   = 'SELL'
                        AND s.id = (
                            SELECT MIN(id) FROM agent_trades
                            WHERE agent_id = b.agent_id
                              AND ticker   = b.ticker
                              AND action   = 'SELL'
                              AND id > b.id
                        )
                    WHERE b.action = 'BUY'
                      AND s.profit_rate IS NOT NULL
                      AND b.agent_id = ?
                    ORDER BY b.id DESC
                    LIMIT 300
                    """,
                    (agent.agent_id,),
                ) as cur:
                    rows = await cur.fetchall()
                agent_trade_results[agent.agent_id] = [dict(r) for r in rows]
                if rows:
                    logger.info(
                        "[Retrain][%s] 거래 결과 %d건 로드 (수익: %d건)",
                        agent.agent_id, len(rows),
                        sum(1 for r in rows if (r["profit_rate"] or 0) > 0),
                    )

        # ── 에이전트별 재학습 ────────────────────────────────────
        success = 0
        for agent in AGENTS.values():
            data = coin_ohlcv if agent.market == "coin" else stock_ohlcv
            fr   = coin_funding if agent.market == "coin" else None
            _btc_ref   = btc_train_ohlcv  if agent.market == "coin"  else None
            _kospi_ref = kospi_train_ohlcv if agent.market == "stock" else None
            _oi_ref    = btc_oi_train      if agent.market == "coin"  else None
            _taker_ref = btc_taker_train   if agent.market == "coin"  else None
            if agent.market == "coin":
                agent._cached_funding_rates = coin_funding
                if btc_oi_train:
                    agent._cached_oi_hist    = btc_oi_train
                if btc_taker_train:
                    agent._cached_taker_hist = btc_taker_train
            if not data:
                logger.warning("[Retrain][%s] 학습 데이터 없음, 스킵", agent.agent_id)
                continue
            _trade_res = agent_trade_results.get(agent.agent_id) or None
            try:
                trained = await asyncio.to_thread(
                    agent.train, data, fr, _btc_ref, _kospi_ref, _oi_ref, _taker_ref, _trade_res
                )
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
                f"🔄 <b>AI 일일 재학습 완료 (18:00)</b>\n"
                f"성공: {success}/20 에이전트\n"
                f"코인 데이터: {len(coin_ohlcv)}봉 | 주식 데이터: {len(stock_ohlcv)}봉"
            )
        except Exception:
            pass

        # 재학습 완료 후 성능 기반 임계값 자동 조정
        await self._auto_adjust_thresholds()
        # 에이전트 검증 충분 시 실매매 ATR 손익 자동 전환 체크
        await self._check_atr_upgrade()

    async def _auto_adjust_thresholds(self) -> None:
        """재학습 후 수익률 기반 임계값 자동 조정 + 텔레그램 알림.

        수익률 < -15%: buy_threshold +0.05 (보수적 매수)
        수익률 < -30%: 비활성화 (매수 완전 스킵)
        수익률 > +5%: 임계값/활성 상태 정상 복구
        """
        from backend.ml.agents import AGENTS, AGENT_CONFIGS
        from backend.db.database import DB_PATH
        import aiosqlite

        default_thresholds = {cfg[0]: cfg[3] for cfg in AGENT_CONFIGS}
        adjustments: list[str] = []

        async with aiosqlite.connect(DB_PATH) as db:
            for agent in AGENTS.values():
                if agent.total_trades < 20:
                    continue  # 거래 수 부족 → 판단 보류

                ret_pct = agent.total_return * 100
                default_thr = default_thresholds.get(agent.agent_id, 0.55)

                if ret_pct < -30 and agent.is_active:
                    agent.is_active = False
                    agent.buy_threshold = min(agent.buy_threshold + 0.05, 0.80)
                    adjustments.append(
                        f"⛔ {agent.agent_id}({agent.feature_set}): 비활성화 (수익률 {ret_pct:.1f}%)"
                    )
                elif ret_pct < -15 and agent.is_active:
                    new_thr = min(round(agent.buy_threshold + 0.05, 4), 0.75)
                    if new_thr > agent.buy_threshold:
                        agent.buy_threshold = new_thr
                        adjustments.append(
                            f"⚠️ {agent.agent_id}({agent.feature_set}): 임계값→{new_thr:.2f} (수익률 {ret_pct:.1f}%)"
                        )
                elif ret_pct > 5 and not agent.is_active:
                    agent.is_active = True
                    agent.buy_threshold = default_thr
                    adjustments.append(
                        f"✅ {agent.agent_id}({agent.feature_set}): 재활성화 (수익률 {ret_pct:.1f}%)"
                    )
                elif ret_pct > 5 and agent.buy_threshold > default_thr + 0.001:
                    agent.buy_threshold = default_thr
                    adjustments.append(
                        f"✅ {agent.agent_id}({agent.feature_set}): 임계값 복구→{default_thr:.2f} (수익률 {ret_pct:.1f}%)"
                    )

                await db.execute(
                    "UPDATE agent_stats SET is_active=?, buy_threshold=? WHERE agent_id=?",
                    (int(agent.is_active), round(agent.buy_threshold, 4), agent.agent_id),
                )

            await db.commit()

        if adjustments:
            msg = "🔧 <b>AI 임계값 자동 조정</b>\n\n" + "\n".join(adjustments)
        else:
            msg = "✅ <b>AI 임계값 점검 완료</b>\n전 에이전트 정상 운영 중 (조정 없음)"

        try:
            await telegram.notify_message(msg)
        except Exception:
            pass

    async def _check_atr_upgrade(self) -> None:
        """에이전트 가상매매 성과 기반 실매매 ATR 손익 자동 전환.

        조건: 활성 에이전트 총 거래 500건 이상 + 평균 수익률 > 0
        충족 시 self._use_atr_risk = True → 이후 실매매 매수 시 ATR 기반 손익 적용.
        """
        if self._use_atr_risk:
            return  # 이미 전환됨

        from backend.ml.agents import AGENTS
        active = [a for a in AGENTS.values() if a.total_trades >= 10 and a.is_active]
        if not active:
            return

        total_trades = sum(a.total_trades for a in active)
        avg_return   = sum(a.total_return for a in active) / len(active)

        if total_trades >= 500 and avg_return > 0:
            self._use_atr_risk = True
            logger.info(
                "[ATR전환] 에이전트 검증 완료 (%d건, 평균수익률 %.2f%%) → 실매매 ATR 손익 적용",
                total_trades, avg_return * 100,
            )
            try:
                await telegram.notify_message(
                    f"✅ <b>실매매 ATR 손익 자동 전환</b>\n"
                    f"에이전트 총 거래: {total_trades}건\n"
                    f"평균 수익률: {avg_return*100:+.2f}%\n\n"
                    f"고정 손절/익절 → ATR×1.5 손절 / ATR×2.0 익절\n"
                    f"(변동성 자동 적응 실매매 적용 시작)"
                )
            except Exception:
                pass
        else:
            logger.info(
                "[ATR전환] 조건 미충족 (거래 %d/500건, 평균수익률 %.2f%%)",
                total_trades, avg_return * 100,
            )

    async def _send_daily_agent_report(self) -> None:
        """매일 22:00 — 에이전트 현황 레포트 텔레그램 전송."""
        from backend.ml.agents import AGENTS, INITIAL_CAPITAL

        now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

        def _market_lines(agents: list) -> tuple[str, str]:
            total_now = sum(a._balance + a.position_value for a in agents)
            total_start = len(agents) * INITIAL_CAPITAL
            ret_pct = (total_now - total_start) / total_start * 100
            all_trades = sum(a.total_trades for a in agents)
            all_wins = sum(a.win_trades for a in agents)
            wr = all_wins / all_trades * 100 if all_trades else 0
            active = sum(1 for a in agents if a.is_active)

            summary = (
                f"총평가: {total_now:,.0f}원 ({ret_pct:+.2f}%)\n"
                f"승률: {wr:.1f}% / {all_trades}건 | 활성: {active}/{len(agents)}"
            )
            lines = []
            for a in sorted(agents, key=lambda x: -x.total_return):
                flag = "⛔" if not a.is_active else ("★" if a.is_champion else "·")
                lines.append(
                    f"{flag} {a.agent_id}({a.feature_set[:3]}): "
                    f"{a.total_return*100:+.1f}% | {a.win_rate*100:.0f}% ({a.total_trades}건)"
                )
            return summary, "\n".join(lines)

        coin_agents  = [a for a in AGENTS.values() if a.market == "coin"]
        stock_agents = [a for a in AGENTS.values() if a.market == "stock"]
        c_summary, c_lines = _market_lines(coin_agents)
        s_summary, s_lines = _market_lines(stock_agents)

        msg = (
            f"📊 <b>AI 에이전트 일일 현황</b>  {now_str}\n"
            f"{'─'*28}\n"
            f"🪙 <b>코인 (AI01~10)</b>\n{c_summary}\n{c_lines}\n"
            f"{'─'*28}\n"
            f"📈 <b>주식 (AI11~20)</b>\n{s_summary}\n{s_lines}"
        )
        try:
            await telegram.notify_message(msg)
        except Exception:
            logger.exception("[레포트] 일일 에이전트 레포트 발송 실패")

    def get_agents_snapshot(self) -> list[dict]:
        """에이전트 상태 스냅샷 반환 (챔피언 먼저, 이후 코인/주식 순)."""
        from backend.ml.agents import AGENTS
        agents = list(AGENTS.values())
        coins  = sorted([a for a in agents if a.market == "coin"],  key=lambda a: (not a.is_champion, -a.win_rate))
        stocks = sorted([a for a in agents if a.market == "stock"], key=lambda a: (not a.is_champion, -a.win_rate))
        return [a.to_dict() for a in coins + stocks]


# 싱글톤 인스턴스 — FastAPI main.py 에서 import 해서 사용
scheduler = TradingScheduler()
