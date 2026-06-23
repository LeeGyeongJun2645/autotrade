"""RSI 역추세 전략 (Relative Strength Index Strategy).

RSI(상대강도지수)로 과매도/과매수 구간을 판단해 매수·매도 신호 생성.

매수 조건: RSI < oversold (기본 30) → 과매도 구간, 반등 기대
매도 조건: RSI > overbought (기본 70) → 과매수 구간, 조정 기대

손절 -2%, 익절 +4% 는 risk_manager.py 에서 별도 처리.

독립 실행 테스트 예시:
    from backend.strategies.rsi import analyze
    result = analyze(ohlcv_list)  # 최소 15개 필요 (period=14 기준)
"""

import pandas as pd

from backend.strategies import OHLCVList, StrategyResult


def calculate_rsi(closes: list[float], period: int = 14) -> float:
    """RSI(상대강도지수) 계산 (Wilder's Smoothing 방식).

    Args:
        closes: 종가 리스트 (오래된 날짜 → 최신 날짜 순, 오름차순)
                최소 period + 1 개 필요
        period: RSI 기간 (기본 14)

    Returns:
        현재(마지막) RSI 값 (0 ~ 100).
        데이터 부족 시 float('nan') 반환.
    """
    if len(closes) < period + 1:
        return float("nan")

    series = pd.Series(closes, dtype=float)
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's EMA (com = period - 1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    last = rsi.iloc[-1]
    return float(last) if last == last else float("nan")  # NaN 처리


def analyze(
    ohlcv_list: OHLCVList,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> StrategyResult:
    """RSI 전략 신호 분석.

    Args:
        ohlcv_list:  최신 날짜가 앞에 오는 OHLCV 리스트
                     (최소 period + 1 개 필요, 권장 period × 3 이상)
        period:      RSI 계산 기간 (기본 14)
        oversold:    과매도 기준 RSI (기본 30 이하 → 매수)
        overbought:  과매수 기준 RSI (기본 70 이상 → 매도)

    Returns:
        StrategyResult(signal, reason, metadata)
        metadata 포함 항목:
            rsi:        현재 RSI 값
            oversold:   설정된 과매도 기준
            overbought: 설정된 과매수 기준
            period:     RSI 계산 기간
    """
    if oversold >= overbought:
        raise ValueError(
            f"oversold({oversold}) 는 overbought({overbought}) 보다 작아야 합니다."
        )

    min_required = period + 1
    if len(ohlcv_list) < min_required:
        return StrategyResult(
            signal="HOLD",
            reason=f"데이터 부족 (필요: {min_required}개, 보유: {len(ohlcv_list)}개)",
            metadata={"period": period, "oversold": oversold, "overbought": overbought},
        )

    # API 응답은 최신 날짜가 앞 → 뒤집어서 시계열 순서로 변환
    closes = [float(c["close"]) for c in reversed(ohlcv_list)]

    rsi_value = calculate_rsi(closes, period)

    if rsi_value != rsi_value:  # NaN 체크
        return StrategyResult(
            signal="HOLD",
            reason="RSI 계산 실패 (NaN)",
            metadata={"period": period},
        )

    metadata = {
        "rsi": round(rsi_value, 2),
        "oversold": oversold,
        "overbought": overbought,
        "period": period,
        "data_count": len(ohlcv_list),
    }

    if rsi_value <= oversold:
        return StrategyResult(
            signal="BUY",
            reason=f"과매도 구간 | RSI {rsi_value:.1f} <= {oversold} (매수 신호)",
            metadata=metadata,
        )

    if rsi_value >= overbought:
        return StrategyResult(
            signal="SELL",
            reason=f"과매수 구간 | RSI {rsi_value:.1f} >= {overbought} (매도 신호)",
            metadata=metadata,
        )

    # 중립 구간 — 추세 방향 힌트 포함
    zone = "중립 (상승 기대)" if rsi_value < 50 else "중립 (하락 주의)"
    return StrategyResult(
        signal="HOLD",
        reason=f"중립 구간 | RSI {rsi_value:.1f} ({zone})",
        metadata=metadata,
    )
