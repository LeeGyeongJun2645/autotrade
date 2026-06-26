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
        """DB에서 포지션 복구. lifespan 시작 시 init_db() 이후 호출."""
        saved = await load_all_positions()
        async with self._lock:
            for symbol, (_, pos) in saved.items():
                if symbol not in self._positions:
                    self._positions[symbol] = pos

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
        # ML 모델 자동 재학습: 매주 월요일 07:00 KST
        self._scheduler.add_job(
            self._weekly_ml_train,
            CronTrigger(day_of_week="mon", hour=7, minute=0, timezone=KST),
            id="weekly_ml_train",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # ML 신호 체크 (코인): 매 정각 24/7
        self._scheduler.add_job(
            self._ml_upbit_tick,
            CronTrigger(minute=0, timezone=KST),
            id="ml_upbit_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # ML 신호 체크 (주식): 평일 장중 30분마다
        self._scheduler.add_job(
            self._ml_kis_tick,
            CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/30", timezone=KST),
            id="ml_kis_tick",
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
        # 챔피언 선정: 매일 00:00
        self._scheduler.add_job(
            self._agent_champion_update,
            CronTrigger(hour=0, minute=0, timezone=KST),
            id="agent_champion",
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

            # MA/RSI 전략은 전략 자체의 청산 신호도 확인 (데드크로스 / RSI 과매수)
            # 변동성 돌파는 EOD 잡(_kis_eod_sell)에서 처리하므로 제외
            if position.strategy in ("moving_average", "rsi"):
                ohlcv = await kis.get_daily_ohlcv(symbol, count=70)
                sell = self._check_strategy_sell(ohlcv, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_kis_sell(symbol, position, "STRATEGY_SELL", current_price)
            return

        # ── 포지션 없음: 매수 신호 탐색 ──
        ohlcv = await kis.get_daily_ohlcv(symbol, count=200)
        result, strategy_name = self._run_kis_strategies(ohlcv, current_price)
        if result.is_buy:
            ml_ok = await self._check_ml_gate(symbol, ohlcv)
            if ml_ok:
                sim_log.push(symbol, f"{strategy_name}+ML 이중확인 — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_kis_buy(symbol, current_price, result, strategy_name)
            else:
                sim_log.push(symbol, f"{strategy_name} 신호 있으나 ML 이중확인 차단 @ {current_price:,.0f}원", "INFO")
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

    async def _check_ml_gate(self, symbol: str, ohlcv: list) -> bool:
        """ML 이중 확인 게이트. buy/strong_buy일 때만 True 반환.

        모델 미학습이면 True(게이트 통과)로 처리해 기존 전략대로 동작.
        """
        try:
            from backend.ml.model import XGBSignalModel
            model = XGBSignalModel(symbol)
            result = await model.predict(ohlcv, settings.cryptopanic_token)
            approved = result.signal in ("buy", "strong_buy")
            logger.info(
                "[ML Gate][%s] %s (확률 %.1f%%) → %s",
                symbol, result.signal, result.buy_prob * 100,
                "통과" if approved else "차단",
            )
            return approved
        except RuntimeError:
            return True  # 모델 없음 → 통과
        except Exception:
            logger.warning("[ML Gate][%s] 체크 실패, 통과 처리", symbol)
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

            # MA/RSI 전략 기반 청산
            if position.strategy in ("moving_average", "rsi"):
                ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=70)
                sell = self._check_strategy_sell(ohlcv, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_upbit_sell(ticker, position, "STRATEGY_SELL", current_price)
            return

        # ── 포지션 없음: 매수 신호 탐색 ──
        ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=200)
        result, strategy_name = self._run_upbit_strategies(ohlcv, current_price)
        if result.is_buy:
            ml_ok = await self._check_ml_gate(ticker, ohlcv)
            if ml_ok:
                sim_log.push(ticker, f"{strategy_name}+ML 이중확인 — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_upbit_buy(ticker, current_price, result, strategy_name)
            else:
                sim_log.push(ticker, f"{strategy_name} 신호 있으나 ML 이중확인 차단 @ {current_price:,.0f}원", "INFO")
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

    async def _weekly_ml_train(self) -> None:
        """매주 월요일 07:00 — 업비트 ML 모델 자동 재학습."""
        try:
            from backend.ml.trainer import train_all
            targets = self._upbit_tickers if self._upbit_tickers else None
            results = await train_all(targets)
            logger.info("[ML] 주간 재학습 완료: %d개 모델", len(results))
            sim_log.push("ML", f"주간 재학습 완료: {len(results)}개 모델", "INFO")
        except Exception:
            logger.exception("[ML] 주간 재학습 중 예외")

    async def _ml_upbit_tick(self) -> None:
        """매 정각 — 감시 중인 업비트 티커 ML 신호 체크 및 텔레그램 알림."""
        from backend.api import upbit
        from backend.ml.model import XGBSignalModel

        tickers = self._upbit_tickers or ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
        for ticker in tickers:
            try:
                ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=200)
                model = XGBSignalModel(ticker)
                result = await model.predict(ohlcv, settings.cryptopanic_token)

                prev = self._ml_signals.get(ticker, {})
                prev_signal = prev.get("signal")
                self._ml_signals[ticker] = {
                    "signal":     result.signal,
                    "buy_prob":   result.buy_prob,
                    "news_score": result.news_score,
                    "checked_at": datetime.now(KST).strftime("%H:%M"),
                }

                signal_changed = prev_signal and prev_signal != result.signal
                is_notable = result.signal in ("buy", "strong_buy", "strong_sell")

                level = "BUY" if "buy" in result.signal else ("SELL" if "sell" in result.signal else "INFO")
                sim_log.push(
                    ticker,
                    f"[ML] {result.signal} | 확률 {result.buy_prob:.1%} | 뉴스 {result.news_score:+.3f}",
                    level,
                )

                if signal_changed or is_notable:
                    await telegram.notify_ml_signal(
                        ticker, result.signal, result.buy_prob,
                        result.news_score, prev_signal if signal_changed else None,
                    )

            except RuntimeError:
                # 모델 미학습 상태 — 조용히 skip
                pass
            except Exception:
                logger.exception("[ML][%s] 업비트 신호 체크 실패", ticker)

    async def _ml_kis_tick(self) -> None:
        """평일 장중 30분마다 — 감시 중인 KIS 종목 ML 신호 체크."""
        from backend.ml.model import XGBSignalModel
        from backend.ml.news import get_sentiment_score

        # KIS는 일봉 데이터 기반, ML 예측은 동일 모델 구조 사용
        # 주식용 모델이 없으면 뉴스 감성만 체크
        for symbol in list(self._kis_symbols):
            try:
                news_score = await get_sentiment_score(symbol)
                prev = self._ml_signals.get(symbol, {})
                prev_score = prev.get("news_score", 0.0)

                self._ml_signals[symbol] = {
                    "signal":     "hold",
                    "buy_prob":   0.5,
                    "news_score": news_score,
                    "checked_at": datetime.now(KST).strftime("%H:%M"),
                }

                # 뉴스 감성이 크게 변했을 때 알림 (±0.3 이상 변화)
                score_shift = abs(news_score - prev_score)
                if score_shift >= 0.3:
                    direction = "긍정적" if news_score > prev_score else "부정적"
                    sim_log.push(
                        symbol,
                        f"[뉴스] 감성 변화 {prev_score:+.3f} → {news_score:+.3f} ({direction})",
                        "WARN",
                    )
                    await telegram.notify_ml_signal(
                        symbol, "hold", 0.5, news_score,
                    )

            except Exception:
                logger.exception("[ML][%s] KIS 신호 체크 실패", symbol)


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

        # OHLCV 캐시 (같은 틱 안에서 재사용)
        ohlcv_cache: dict[str, list] = {}

        async def _coin_ohlcv(ticker: str, interval: str) -> list:
            key = f"{ticker}:{interval}"
            if key not in ohlcv_cache:
                try:
                    ohlcv_cache[key] = await _upbit.get_ohlcv(ticker, interval=interval, count=200)
                except Exception:
                    ohlcv_cache[key] = []
            return ohlcv_cache[key]

        async def _stock_ohlcv(symbol: str, interval_min: int) -> list:
            key = f"KIS:{symbol}:{interval_min}"
            if key not in ohlcv_cache:
                try:
                    ohlcv_cache[key] = await _kis.get_minute_ohlcv(symbol, interval_min, count=200)
                except Exception:
                    ohlcv_cache[key] = []
            return ohlcv_cache[key]

        async with aiosqlite.connect(DB_PATH) as db:
            for agent in AGENTS.values():
                try:
                    # ── 코인 가상 매매 (24/7) ─────────────────────────
                    for ticker in coin_tickers:
                        ohlcv = await _coin_ohlcv(ticker, agent.interval_str)
                        if not ohlcv:
                            continue

                        if agent._model is None and not agent.load_model():
                            trained = await asyncio.to_thread(agent.train, ohlcv)
                            if not trained:
                                continue

                        signal, prob = agent.predict(ohlcv)
                        try:
                            price_data = await _upbit.get_price(ticker)
                            price = float(price_data["current_price"])
                        except Exception:
                            continue

                        await self._agent_execute(db, agent, ticker, signal, prob, price)

                    # ── 주식 가상 매매 (장중만) ───────────────────────
                    if is_market_open:
                        for symbol in stock_symbols:
                            ohlcv = await _stock_ohlcv(symbol, agent.interval_min)
                            if not ohlcv:
                                continue

                            signal, prob = agent.predict(ohlcv)
                            try:
                                price_data = await _kis.get_price(symbol)
                                price = float(price_data["current_price"])
                            except Exception:
                                continue

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

    async def _agent_execute(self, db, agent, symbol: str, signal: str, prob: float, price: float) -> None:
        """에이전트 단일 종목 가상 매수/매도 실행 + DB 기록."""
        from backend.ml.agents import AgentTrade

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

    async def _agent_champion_update(self) -> None:
        """매일 자정 — 챔피언 에이전트 선정 + 텔레그램 알림."""
        from backend.ml.agents import update_champion
        champ_id = update_champion()
        if champ_id:
            logger.info("[Agent] 챔피언 선정: %s", champ_id)
            sim_log.push("AGENT", f"챔피언 선정: {champ_id}", "INFO")
            await telegram.notify_message(f"🏆 <b>AI 챔피언 선정</b>\n에이전트: <code>{champ_id}</code>")

    def get_agents_snapshot(self) -> list[dict]:
        """현재 20개 에이전트 상태 스냅샷 반환."""
        from backend.ml.agents import AGENTS
        return [a.to_dict() for a in AGENTS.values()]


# 싱글톤 인스턴스 — FastAPI main.py 에서 import 해서 사용
scheduler = TradingScheduler()
