"""글로벌 시장 sentiment 모듈.

무료 지표만 사용:
- 코인: Crypto Fear & Greed Index (alternative.me) + BTC 도미넌스 (CoinGecko)
- 주식: KOSPI 실시간 등락률 (scheduler에서 30분마다 업데이트)

모듈 레벨 캐시로 모든 에이전트 공유 — 개별 API 호출 없음.
점수 범위: -1.0 (강한 하락장) ~ +1.0 (강한 상승장)
"""

import logging
import time

logger = logging.getLogger(__name__)

# ── 글로벌 sentiment 캐시 ─────────────────────────────────────────
_COIN_SENTIMENT: float = 0.0        # 코인 시장 sentiment (-1 ~ +1)
_STOCK_SENTIMENT: float = 0.0       # 주식 시장 sentiment (-1 ~ +1)
_COIN_UPDATED_AT: float = 0.0       # 마지막 업데이트 시각 (time.time())
_STOCK_UPDATED_AT: float = 0.0
_STALE_SEC: float = 7200.0          # 2시간 넘으면 stale → 중립(0.0) 반환


def update_coin_sentiment(fear_greed_0_100: int, btc_dominance: float) -> None:
    """코인 sentiment 업데이트 (scheduler의 _agent_tick에서 30분마다 호출).

    Args:
        fear_greed_0_100: alternative.me Fear & Greed Index (0=극도공포, 100=극도탐욕)
        btc_dominance: BTC 시총 점유율 (0~100)
    """
    global _COIN_SENTIMENT, _COIN_UPDATED_AT

    # Fear & Greed 역발상 정규화: 탐욕(100) → -1, 공포(0) → +1
    # 탐욕 = 과열 → 하락 위험, 공포 = 저평가 → 반등 가능
    fg_score = -(fear_greed_0_100 - 50) / 50.0  # -1 ~ +1

    # BTC 도미넌스: 높으면 알트 약세, 낮으면 알트시즌
    if btc_dominance >= 62.0:
        dom_score = -0.4    # BTC 독주 = 알트 자금 이탈
    elif btc_dominance >= 55.0:
        dom_score = -0.1
    elif btc_dominance <= 42.0:
        dom_score = +0.4    # 알트시즌
    elif btc_dominance <= 50.0:
        dom_score = +0.2
    else:
        dom_score = 0.0

    combined = max(-1.0, min(1.0, fg_score * 0.7 + dom_score * 0.3))
    _COIN_SENTIMENT = combined
    _COIN_UPDATED_AT = time.time()
    logger.info(
        "[Sentiment] 코인 업데이트 F&G=%d BTC도미넌스=%.1f%% → sentiment=%.3f",
        fear_greed_0_100, btc_dominance, combined,
    )


def update_stock_sentiment(kospi_change_pct: float) -> None:
    """주식 sentiment 업데이트 (scheduler의 _agent_tick에서 30분마다 호출).

    Args:
        kospi_change_pct: KOSPI 당일 등락률 (예: -0.012 = -1.2%)
    """
    global _STOCK_SENTIMENT, _STOCK_UPDATED_AT

    pct = kospi_change_pct
    if pct >= 0.015:
        score = +1.0    # +1.5% 이상: 강세장
    elif pct >= 0.005:
        score = +0.5    # +0.5% 이상: 상승
    elif pct >= -0.003:
        score = 0.0     # ±0.3% 이내: 중립
    elif pct >= -0.010:
        score = -0.5    # -1.0% 이내: 약세
    else:
        score = -1.0    # -1.0% 이하: 강한 약세

    _STOCK_SENTIMENT = score
    _STOCK_UPDATED_AT = time.time()
    logger.info(
        "[Sentiment] 주식 업데이트 KOSPI등락=%.2f%% → sentiment=%.1f",
        pct * 100, score,
    )


def get_coin_sentiment() -> float:
    """코인 시장 sentiment 반환. 2시간 이상 미업데이트 시 중립(0.0) 반환."""
    if time.time() - _COIN_UPDATED_AT > _STALE_SEC:
        return 0.0
    return _COIN_SENTIMENT


def get_stock_sentiment() -> float:
    """주식 시장 sentiment 반환. 2시간 이상 미업데이트 시 중립(0.0) 반환."""
    if time.time() - _STOCK_UPDATED_AT > _STALE_SEC:
        return 0.0
    return _STOCK_SENTIMENT
