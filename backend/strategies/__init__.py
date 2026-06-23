"""전략 공통 타입 정의."""

from dataclasses import dataclass, field
from typing import Any, Literal

Signal = Literal["BUY", "SELL", "HOLD"]

# OHLCV 딕셔너리 형식 (KIS·업비트 공통)
# {"date": str, "open": float, "high": float, "low": float, "close": float, "volume": float}
OHLCVList = list[dict[str, Any]]


@dataclass
class StrategyResult:
    """전략 분석 결과.

    Attributes:
        signal:   BUY | SELL | HOLD
        reason:   신호 발생 이유 (로그·알림용 한국어 문자열)
        metadata: 전략별 추가 수치 (목표가, RSI 값 등)
    """

    signal: Signal
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.signal == "SELL"
