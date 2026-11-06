"""글로벌 지수 캐시 모듈 — predict_ensemble() 임계값 보정에 사용.

yfinance로 S&P500·닛케이·VIX 전일 수익률을 수집하고 1시간 캐시.
매 틱마다 API 호출을 피하기 위해 모듈 레벨 캐시 사용.
"""
import logging
import time

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600  # 1시간


def get_global_sentiment() -> dict:
    """S&P500·닛케이225·VIX 전일 수익률 반환. 실패 시 중립값(0.0/20.0).

    Returns:
        {
            'sp500_ret':  float  (전일 수익률, 예: -0.011 = -1.1%)
            'nikkei_ret': float
            'vix_level':  float  (절대값, 예: 18.8)
        }
    """
    global _cache, _cache_ts

    now = time.time()
    if now - _cache_ts < _CACHE_TTL and _cache:
        return _cache

    result = {'sp500_ret': 0.0, 'nikkei_ret': 0.0, 'vix_level': 20.0}
    try:
        import yfinance as yf

        sp   = yf.download('^GSPC', period='5d', auto_adjust=True, progress=False)['Close']
        nk   = yf.download('^N225', period='5d', auto_adjust=True, progress=False)['Close']
        vix  = yf.download('^VIX',  period='5d', auto_adjust=True, progress=False)['Close']

        if len(sp) >= 2:
            result['sp500_ret']  = float(sp.pct_change().iloc[-1])
        if len(nk) >= 2:
            result['nikkei_ret'] = float(nk.pct_change().iloc[-1])
        if len(vix) >= 1:
            result['vix_level']  = float(vix.iloc[-1])

        _cache    = result
        _cache_ts = now
        logger.info(
            "[GlobalMarket] 수집 완료 sp500=%.2f%% nikkei=%.2f%% vix=%.1f",
            result['sp500_ret'] * 100,
            result['nikkei_ret'] * 100,
            result['vix_level'],
        )
    except Exception as e:
        logger.warning("[GlobalMarket] 수집 실패 — 중립값 사용: %s", e)
        if not _cache:
            _cache = result
        else:
            result = _cache

    return result


def global_thr_mult(sp500_ret: float, nikkei_ret: float, vix_level: float) -> float:
    """글로벌 지수 상태에 따른 임계값 보정 계수 반환.

    Args:
        sp500_ret:  S&P500 전일 수익률
        nikkei_ret: 닛케이225 전일 수익률
        vix_level:  VIX 절대값

    Returns:
        float: avg_thr_adj에 곱할 계수 (1.0 = 변화 없음)
    """
    mult = 1.0

    # 글로벌 동반 하락 → 진입 보수적
    if sp500_ret < -0.01 and nikkei_ret < -0.01:
        mult *= 1.15   # 둘 다 -1% 이하
    elif sp500_ret < -0.005 or nikkei_ret < -0.005:
        mult *= 1.07   # 하나라도 -0.5% 이하

    # 글로벌 동반 상승 → 진입 약간 적극적
    elif sp500_ret > 0.01 and nikkei_ret > 0.01:
        mult *= 0.95   # 둘 다 +1% 이상

    # VIX 높으면 변동성 경계 → 추가 보정
    if vix_level >= 30:
        mult *= 1.10
    elif vix_level >= 25:
        mult *= 1.05

    return min(round(mult, 4), 1.10)  # 하락장에서도 임계값 최대 10%만 가중
