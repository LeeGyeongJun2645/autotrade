"""변동성 돌파 전략 (Volatility Breakout Strategy).

래리 윌리엄스가 고안한 단기 추세 추종 전략.
목표가 = 시가 + (전일 고가 - 전일 저가) × K

매수 조건:
    당일 현재가 >= 목표가

매도 조건:
    당일 장 마감 직전 (스케줄러가 별도 호출) → sell_eod() 참고

독립 실행 테스트 예시:
    from backend.strategies.volatility_breakout import analyze
    result = analyze(ohlcv_list, current_price=72000)
"""

from backend.strategies import OHLCVList, StrategyResult


def calculate_target_price(
    prev_high: float,
    prev_low: float,
    today_open: float,
    k: float = 0.5,
) -> float:
    """목표가 계산.

    Args:
        prev_high:  전일 고가
        prev_low:   전일 저가
        today_open: 당일 시가
        k:          변동성 계수 (기본 0.5, 보통 0.4~0.6 범위 사용)

    Returns:
        목표가 (이 가격 이상이면 매수 신호)
    """
    return today_open + (prev_high - prev_low) * k


def analyze(
    ohlcv_list: OHLCVList,
    current_price: float,
    k: float = 0.5,
    position_held: bool = False,
) -> StrategyResult:
    """변동성 돌파 전략 신호 분석.

    Args:
        ohlcv_list:     최신 날짜가 앞에 오는 OHLCV 리스트 (최소 2개 필요)
                        [0] = 오늘(당일 시가 필요), [1] = 전일(고가·저가 필요)
        current_price:  현재 체결가 (실시간)
        k:              변동성 계수 (기본 0.5)
        position_held:  현재 해당 종목 포지션 보유 여부

    Returns:
        StrategyResult(signal, reason, metadata)
        metadata 포함 항목:
            target_price: 목표가
            range:        전일 등락폭
            k:            사용된 K값
    """
    if len(ohlcv_list) < 2:
        return StrategyResult(
            signal="HOLD",
            reason="데이터 부족 (최소 2일치 OHLCV 필요)",
            metadata={},
        )

    today = ohlcv_list[0]
    yesterday = ohlcv_list[1]

    prev_high = float(yesterday["high"])
    prev_low = float(yesterday["low"])
    today_open = float(today["open"])
    day_range = prev_high - prev_low

    if day_range <= 0:
        return StrategyResult(
            signal="HOLD",
            reason="전일 등락폭 0 — 신호 계산 불가",
            metadata={"range": day_range},
        )

    target = calculate_target_price(prev_high, prev_low, today_open, k)
    metadata = {
        "target_price": round(target, 2),
        "today_open": today_open,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "range": round(day_range, 2),
        "k": k,
        "current_price": current_price,
    }

    # 이미 포지션 보유 중 → 종가 매도 대기 (스케줄러에서 sell_eod 호출)
    if position_held:
        return StrategyResult(
            signal="HOLD",
            reason=f"포지션 보유 중 | 목표가 {target:,.0f} | 장 마감 시 자동 매도",
            metadata=metadata,
        )

    # 매수 조건: 현재가 >= 목표가
    if current_price >= target:
        return StrategyResult(
            signal="BUY",
            reason=f"목표가 돌파 | 현재가 {current_price:,.0f} >= 목표가 {target:,.0f} (K={k})",
            metadata=metadata,
        )

    return StrategyResult(
        signal="HOLD",
        reason=f"미돌파 | 현재가 {current_price:,.0f} < 목표가 {target:,.0f}",
        metadata=metadata,
    )


def sell_eod(position_held: bool) -> StrategyResult:
    """장 마감 직전 강제 매도 신호 (스케줄러에서 15:20 KST 호출, 장 마감 15:30 기준).

    변동성 돌파 전략의 원칙은 '당일 매수 → 당일 종가 매도'.
    포지션 미보유 시에는 HOLD 반환.

    Args:
        position_held: 현재 포지션 보유 여부

    Returns:
        StrategyResult(signal='SELL' or 'HOLD')
    """
    if position_held:
        return StrategyResult(
            signal="SELL",
            reason="장 마감 전 정리 (변동성 돌파 당일 매도 원칙)",
            metadata={},
        )
    return StrategyResult(signal="HOLD", reason="보유 포지션 없음", metadata={})
