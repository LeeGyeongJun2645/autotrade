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
from backend.core.risk_manager import Position, RiskManager
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
        ohlcv = await kis.get_daily_ohlcv(symbol, count=70)
        result, strategy_name = self._run_kis_strategies(ohlcv, current_price)
        if result.is_buy:
            await self._execute_kis_buy(symbol, current_price, result, strategy_name)

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
        ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=70)
        result, strategy_name = self._run_upbit_strategies(ohlcv, current_price)
        if result.is_buy:
            await self._execute_upbit_buy(ticker, current_price, result, strategy_name)

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
            await telegram.notify_sell(ticker, position.qty, current_price, profit_rate, reason, is_crypto=True)

            async with self._lock:
                self._positions.pop(ticker, None)
            await delete_position(ticker)
        except Exception:
            logger.exception("[Upbit][%s] 매도 실행 중 예외", ticker)


# 싱글톤 인스턴스 — FastAPI main.py 에서 import 해서 사용
scheduler = TradingScheduler()
