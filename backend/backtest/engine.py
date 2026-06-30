"""Backtrader 기반 백테스트 엔진.

지원 전략: volatility_breakout | moving_average | rsi
입력:  OHLCV 리스트 (KIS/Upbit API 반환 형식, 최신 날짜가 앞)
출력:  BacktestResult (총수익률, 승률, 최대낙폭, 거래횟수 등)

사용 예시:
    from backend.backtest.engine import run_backtest
    result = run_backtest(ohlcv_list, "rsi", initial_cash=10_000_000)
"""

import math
from dataclasses import dataclass
from typing import Any

import backtrader as bt
import pandas as pd

from backend.config import settings


@dataclass
class BacktestResult:
    strategy: str
    initial_cash: float
    final_value: float
    total_return_pct: float  # 총수익률 (%)
    trade_count: int         # 완료된 거래 횟수
    win_count: int           # 수익 거래 횟수
    loss_count: int          # 손실 거래 횟수
    win_rate: float          # 승률 (0~100)
    max_drawdown_pct: float  # 최대낙폭 (%)
    sharpe_ratio: float      # 연환산 샤프 비율


# ── 데이터 변환 ──────────────────────────────────────────────────

def _ohlcv_to_feed(ohlcv_list: list[dict[str, Any]]) -> bt.feeds.PandasData:
    """API OHLCV 리스트 → Backtrader PandasData 피드 변환.

    API는 최신 날짜가 앞(내림차순) → reversed 후 인덱스 정렬.
    KIS 날짜: 'YYYYMMDD' / Upbit 날짜: 'YYYY-MM-DDTHH:MM:SS' 모두 처리.
    """
    df = pd.DataFrame(list(reversed(ohlcv_list)))
    # str[:10]으로 자르면 5분봉 등 장중 데이터에서 하루 1봉만 남으므로 전체 파싱
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return bt.feeds.PandasData(dataname=df)


# ── 공통 베이스 전략 ─────────────────────────────────────────────

class _BaseStrategy(bt.Strategy):
    """공통 파라미터 + 손절 + 백테스트 종료 시 청산 처리."""

    params = (
        ("stop_loss_rate", -0.02),
        ("max_position_ratio", 0.20),
        ("is_crypto", False),
    )

    def __init__(self) -> None:
        self._order: Any = None
        self._entry_price: float = 0.0

    def notify_order(self, order: bt.Order) -> None:
        if order.status == order.Completed:
            if order.isbuy():
                self._entry_price = order.executed.price
            self._order = None
        elif order.status in (order.Cancelled, order.Rejected, order.Margin):
            self._order = None

    def _calc_size(self, price: float) -> float:
        """가용 현금과 max_position_ratio 기반 주문 수량 계산.

        코인은 소수점 수량 허용, 주식은 정수(주) 단위.
        """
        if price <= 0:
            return 0
        cash = self.broker.getcash()
        raw = cash * self.params.max_position_ratio / price
        return raw if self.params.is_crypto else float(max(math.floor(raw), 0))

    def _is_stop_loss(self) -> bool:
        """현재 종가가 손절가 이하인지 확인."""
        if self._entry_price <= 0:
            return False
        return self.data.close[0] <= self._entry_price * (1 + self.params.stop_loss_rate)

    def stop(self) -> None:
        """백테스트 종료 시 잔여 포지션 강제 청산."""
        if self.position and not self._order:
            self.close()


# ── 변동성 돌파 전략 ──────────────────────────────────────────────

class _VBStrategy(_BaseStrategy):
    """변동성 돌파 전략 (래리 윌리엄스).

    목표가 = 전일 종가 + 전일 레인지 × K (전일 종가를 당일 시가 근사치로 사용).
    당일 고가가 목표가를 초과하면 매수 → 다음 봉 시작(시가)에 청산.
    """

    params = (
        ("k", 0.5),
    )

    def next(self) -> None:
        if self._order:
            return

        if self.position:
            if self._is_stop_loss():
                self._order = self.close()
                return
            # 다음 봉 시가에 청산 (1일 보유 원칙)
            self._order = self.close()
        else:
            prev_range = self.data.high[-1] - self.data.low[-1]
            if prev_range <= 0:
                return

            # 당일 시가 + 전일 레인지 × K = 당일 목표가 (라이브 전략과 동일)
            target = self.data.open[0] + prev_range * self.params.k

            if self.data.high[0] >= target:
                size = self._calc_size(self.data.close[0])
                if size > 0:
                    self._order = self.buy(size=size)


# ── 이동평균 크로스 전략 ─────────────────────────────────────────

class _MAStrategy(_BaseStrategy):
    """이동평균 크로스 전략.

    골든크로스(단기 > 장기 역전) → 매수.
    데드크로스(단기 < 장기 역전) 또는 손절 → 매도.
    """

    params = (
        ("ma_short", 20),
        ("ma_long", 60),
    )

    def __init__(self) -> None:
        super().__init__()
        short_ma = bt.indicators.SMA(self.data.close, period=self.params.ma_short)
        long_ma = bt.indicators.SMA(self.data.close, period=self.params.ma_long)
        # CrossOver: +1 = 골든크로스, -1 = 데드크로스
        self._cross = bt.indicators.CrossOver(short_ma, long_ma)

    def next(self) -> None:
        if self._order:
            return

        if self.position:
            if self._is_stop_loss() or self._cross[0] < 0:
                self._order = self.close()
        else:
            if self._cross[0] > 0:
                size = self._calc_size(self.data.close[0])
                if size > 0:
                    self._order = self.buy(size=size)


# ── RSI 역추세 전략 ──────────────────────────────────────────────

class _RSIStrategy(_BaseStrategy):
    """RSI 역추세 전략 (Wilder's Smoothing).

    RSI ≤ oversold → 매수, RSI ≥ overbought 또는 손절 → 매도.
    """

    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30.0),
        ("rsi_overbought", 70.0),
    )

    def __init__(self) -> None:
        super().__init__()
        # Backtrader RSI는 기본적으로 Wilder's Smoothing 사용 (라이브와 동일)
        self._rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period,
            safediv=True,
        )

    def next(self) -> None:
        if self._order:
            return

        if self.position:
            if self._is_stop_loss() or self._rsi[0] >= self.params.rsi_overbought:
                self._order = self.close()
        else:
            if self._rsi[0] <= self.params.rsi_oversold:
                size = self._calc_size(self.data.close[0])
                if size > 0:
                    self._order = self.buy(size=size)


# ── 전략 레지스트리 ──────────────────────────────────────────────

_STRATEGY_MAP: dict[str, type[_BaseStrategy]] = {
    "volatility_breakout": _VBStrategy,
    "moving_average": _MAStrategy,
    "rsi": _RSIStrategy,
}

SUPPORTED_STRATEGIES = list(_STRATEGY_MAP)


def _strategy_params(strategy_name: str) -> dict[str, Any]:
    """전략명별 config 파라미터 매핑."""
    if strategy_name == "volatility_breakout":
        return {"k": settings.volatility_k}
    if strategy_name == "moving_average":
        return {"ma_short": settings.ma_short, "ma_long": settings.ma_long}
    if strategy_name == "rsi":
        return {
            "rsi_period": settings.rsi_period,
            "rsi_oversold": settings.rsi_oversold,
            "rsi_overbought": settings.rsi_overbought,
        }
    return {}


# ── 메인 백테스트 함수 ────────────────────────────────────────────

def run_backtest(
    ohlcv_list: list[dict[str, Any]],
    strategy_name: str,
    initial_cash: float = 10_000_000,
    commission: float = 0.00015,
    is_crypto: bool = False,
) -> BacktestResult:
    """백테스트 실행 (동기 함수 — FastAPI에서는 asyncio.to_thread로 호출).

    Args:
        ohlcv_list:    OHLCV 데이터 (최신 날짜가 앞, 내림차순)
                       MA: 최소 61개 / RSI: 최소 15개 / VB: 최소 2개
        strategy_name: 'volatility_breakout' | 'moving_average' | 'rsi'
        initial_cash:  초기 자본금 (기본 1,000만원)
        commission:    편도 수수료율 (기본 0.015%, 세금 별도)

    Returns:
        BacktestResult

    Raises:
        ValueError: 지원하지 않는 전략명 또는 빈 데이터
    """
    if strategy_name not in _STRATEGY_MAP:
        raise ValueError(
            f"지원하지 않는 전략: '{strategy_name}'. 가능한 값: {SUPPORTED_STRATEGIES}"
        )
    if not ohlcv_list:
        raise ValueError("OHLCV 데이터가 비어있습니다.")
    if initial_cash <= 0:
        raise ValueError("initial_cash 는 양수여야 합니다.")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(_ohlcv_to_feed(ohlcv_list))
    cerebro.addstrategy(
        _STRATEGY_MAP[strategy_name],
        stop_loss_rate=settings.stop_loss_rate,
        max_position_ratio=settings.max_position_ratio,
        is_crypto=is_crypto,
        **_strategy_params(strategy_name),
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.035,  # 한국 국고채 기준 3.5%
    )

    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    return _build_result(strategy_name, initial_cash, final_value, results[0])


def _build_result(
    strategy_name: str,
    initial_cash: float,
    final_value: float,
    strat: bt.Strategy,
) -> BacktestResult:
    """분석기 원시 데이터 → BacktestResult 변환."""
    total_return_pct = (final_value / initial_cash - 1) * 100

    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("closed", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0.0

    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0)

    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    _sr = float(sharpe_raw)
    sharpe = 0.0 if (_sr != _sr or math.isinf(_sr)) else _sr  # NaN/inf → 0

    return BacktestResult(
        strategy=strategy_name,
        initial_cash=initial_cash,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return_pct, 2),
        trade_count=total_trades,
        win_count=won,
        loss_count=lost,
        win_rate=round(win_rate, 1),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
    )
