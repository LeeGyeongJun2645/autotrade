"""이동평균 크로스 전략 (Moving Average Cross Strategy).

단기 이동평균이 장기 이동평균을 상향 돌파 → 골든크로스(매수)
단기 이동평균이 장기 이동평균을 하향 돌파 → 데드크로스(매도)

기본값: 단기 20일, 장기 60일

독립 실행 테스트 예시:
    from backend.strategies.moving_average import analyze
    result = analyze(ohlcv_list)  # 최소 61개 필요
"""

import pandas as pd

from backend.strategies import OHLCVList, StrategyResult

CrossType = str  # "GOLDEN" | "DEAD" | "NONE"


def calculate_ma(closes: list[float], period: int) -> list[float]:
    """단순 이동평균 계산.

    Args:
        closes: 종가 리스트 (오래된 날짜 → 최신 날짜 순, 오름차순)
        period: 이동평균 기간

    Returns:
        이동평균 리스트. period 미만 구간은 NaN → float('nan').
        입력과 동일한 길이.
    """
    series = pd.Series(closes, dtype=float)
    ma = series.rolling(window=period, min_periods=period).mean()
    return ma.tolist()


def detect_cross(
    short_ma: list[float],
    long_ma: list[float],
) -> CrossType:
    """직전 봉 → 현재 봉 사이의 크로스 여부 감지.

    크로스는 '전 봉에서는 반대였다가 현 봉에서 역전'될 때만 감지.
    단순히 short > long 이면 매수가 아니라, '역전'이 일어난 시점만 신호.

    Args:
        short_ma: 단기 MA 리스트 (오름차순, 최소 2개)
        long_ma:  장기 MA 리스트 (오름차순, 최소 2개)

    Returns:
        "GOLDEN" | "DEAD" | "NONE"
    """
    # 유효한 값이 있는 마지막 2개 인덱스 추출
    valid_pairs = [
        (s, l)
        for s, l in zip(short_ma, long_ma)
        if s == s and l == l  # NaN 제외 (float nan != nan)
    ]
    if len(valid_pairs) < 2:
        return "NONE"

    prev_short, prev_long = valid_pairs[-2]
    curr_short, curr_long = valid_pairs[-1]

    # 골든크로스: 이전엔 short <= long, 현재는 short > long
    if prev_short <= prev_long and curr_short > curr_long:
        return "GOLDEN"

    # 데드크로스: 이전엔 short >= long, 현재는 short < long
    if prev_short >= prev_long and curr_short < curr_long:
        return "DEAD"

    return "NONE"


def analyze(
    ohlcv_list: OHLCVList,
    ma_short: int = 20,
    ma_long: int = 60,
) -> StrategyResult:
    """이동평균 크로스 전략 신호 분석.

    Args:
        ohlcv_list: 최신 날짜가 앞에 오는 OHLCV 리스트
                    (최소 ma_long + 1 개 필요, 권장 ma_long × 2)
        ma_short:   단기 이동평균 기간 (기본 20일)
        ma_long:    장기 이동평균 기간 (기본 60일)

    Returns:
        StrategyResult(signal, reason, metadata)
        metadata 포함 항목:
            short_ma:    현재 단기 MA 값
            long_ma:     현재 장기 MA 값
            cross:       "GOLDEN" | "DEAD" | "NONE"
            ma_short:    단기 기간
            ma_long:     장기 기간
    """
    if ma_short >= ma_long:
        raise ValueError(f"단기({ma_short}) 기간은 장기({ma_long})보다 짧아야 합니다.")

    min_required = ma_long + 1  # 크로스 감지에 최소 1봉 여유 필요
    if len(ohlcv_list) < min_required:
        return StrategyResult(
            signal="HOLD",
            reason=f"데이터 부족 (필요: {min_required}개, 보유: {len(ohlcv_list)}개)",
            metadata={"ma_short": ma_short, "ma_long": ma_long},
        )

    # API 응답은 최신 날짜가 앞 → 뒤집어서 시계열 순서로 변환
    closes = [float(c["close"]) for c in reversed(ohlcv_list)]

    short_mas = calculate_ma(closes, ma_short)
    long_mas = calculate_ma(closes, ma_long)

    cross = detect_cross(short_mas, long_mas)

    # 현재(마지막) 유효 MA 값 추출
    curr_short = next((v for v in reversed(short_mas) if v == v), None)
    curr_long = next((v for v in reversed(long_mas) if v == v), None)

    metadata = {
        "short_ma": round(curr_short, 2) if curr_short is not None else None,
        "long_ma": round(curr_long, 2) if curr_long is not None else None,
        "cross": cross,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "data_count": len(ohlcv_list),
    }

    if cross == "GOLDEN":
        return StrategyResult(
            signal="BUY",
            reason=(
                f"골든크로스 발생 | MA{ma_short}({curr_short:,.1f}) "
                f"> MA{ma_long}({curr_long:,.1f})"
            ),
            metadata=metadata,
        )

    if cross == "DEAD":
        return StrategyResult(
            signal="SELL",
            reason=(
                f"데드크로스 발생 | MA{ma_short}({curr_short:,.1f}) "
                f"< MA{ma_long}({curr_long:,.1f})"
            ),
            metadata=metadata,
        )

    # 크로스 없음 — 현재 추세 방향 설명
    if curr_short is not None and curr_long is not None:
        trend = "상승 추세 유지" if curr_short > curr_long else "하락 추세 유지"
        reason = f"크로스 없음 ({trend}) | MA{ma_short}({curr_short:,.1f}) / MA{ma_long}({curr_long:,.1f})"
    else:
        reason = "MA 계산 중 (데이터 누적 대기)"

    return StrategyResult(signal="HOLD", reason=reason, metadata=metadata)
