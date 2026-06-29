"""바이낸스 공개 API 클라이언트.

인증 없이 사용 가능한 공개 엔드포인트만 사용:
- 현물 가격 (BTCUSDT 등)
- 선물 펀딩비
- USD/KRW 환율 (exchangerate.host 무료 API)
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 8

# ── 캐시 ──────────────────────────────────────────────────────────
_PRICE_CACHE:    dict[str, tuple[float, float]] = {}  # symbol → (price, ts)
_FX_CACHE:       tuple[float, float] | None = None    # (usd_krw, ts)
_FUNDING_CACHE:  dict[str, tuple[float, float]] = {}  # symbol → (rate, ts)
_KIMCHI_CACHE:   dict[str, tuple[float, float]] = {}  # ticker → (premium%, ts)
_HIST_FUNDING_CACHE: dict[str, tuple[list, float]] = {}  # symbol:limit → (data, ts)
_OI_CACHE:       dict[str, tuple[list, float]] = {}   # key → (data, ts)
_TAKER_CACHE:    dict[str, tuple[list, float]] = {}   # key → (data, ts)

_PRICE_TTL        = 30.0
_FX_TTL           = 300.0
_FUNDING_TTL      = 300.0
_KIMCHI_TTL       = 60.0
_HIST_FUNDING_TTL = 3600.0  # 8시간 주기 데이터라 1시간 캐시로 충분
_OI_TTL           = 300.0   # 5분봉과 동일 주기
_TAKER_TTL        = 300.0


async def get_binance_price(symbol: str = "BTCUSDT") -> float:
    """바이낸스 현물 현재가 반환 (USD)."""
    now = time.time()
    if symbol in _PRICE_CACHE:
        price, ts = _PRICE_CACHE[symbol]
        if now - ts < _PRICE_TTL:
            return price
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol},
            )
            resp.raise_for_status()
            price = float(resp.json()["price"])
            _PRICE_CACHE[symbol] = (price, now)
            return price
    except Exception as e:
        logger.warning("[바이낸스] 가격 조회 실패 %s: %s", symbol, e)
        return _PRICE_CACHE.get(symbol, (0.0, 0))[0]


async def get_usd_krw() -> float:
    """USD/KRW 환율 반환 (exchangerate.host 무료 API)."""
    global _FX_CACHE
    now = time.time()
    if _FX_CACHE and now - _FX_CACHE[1] < _FX_TTL:
        return _FX_CACHE[0]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://api.exchangerate-api.com/v4/latest/USD",
            )
            resp.raise_for_status()
            rate = float(resp.json()["rates"]["KRW"])
            _FX_CACHE = (rate, now)
            return rate
    except Exception as e:
        logger.warning("[환율] 조회 실패: %s", e)
        return _FX_CACHE[0] if _FX_CACHE else 1350.0  # 폴백 기본값


async def get_funding_rate(symbol: str = "BTCUSDT") -> float:
    """바이낸스 선물 최근 펀딩비 반환.

    양수(+): 롱 포지션이 숏에 지급 → 과매수 신호
    음수(-): 숏 포지션이 롱에 지급 → 과매도 신호
    """
    now = time.time()
    if symbol in _FUNDING_CACHE:
        rate, ts = _FUNDING_CACHE[symbol]
        if now - ts < _FUNDING_TTL:
            return rate
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"FundingRate API 오류: {data}")
            rate = float(data[0]["fundingRate"]) if data else 0.0
            _FUNDING_CACHE[symbol] = (rate, now)
            return rate
    except Exception as e:
        logger.warning("[펀딩비] 조회 실패 %s: %s", symbol, e)
        return _FUNDING_CACHE.get(symbol, (0.0, 0))[0]


async def get_historical_funding_rates(symbol: str = "BTCUSDT", limit: int = 50) -> list[dict]:
    """바이낸스 선물 과거 펀딩비 히스토리 반환.

    8시간 주기로 발생. limit=50이면 약 16일치.
    반환: [{"fundingTime": ms타임스탬프, "fundingRate": "0.0001"}, ...]
    """
    cache_key = f"{symbol}:{limit}"
    now = time.time()
    if cache_key in _HIST_FUNDING_CACHE:
        data, ts = _HIST_FUNDING_CACHE[cache_key]
        if now - ts < _HIST_FUNDING_TTL:
            return data
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol.upper(), "limit": min(limit, 1000)},
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"펀딩비히스토리 API 오류: {data}")
            _HIST_FUNDING_CACHE[cache_key] = (data, now)
            return data
    except Exception as e:
        logger.warning("[펀딩비히스토리] 조회 실패 %s: %s", symbol, e)
        cached = _HIST_FUNDING_CACHE.get(cache_key)
        return cached[0] if cached else []


async def get_open_interest_hist(symbol: str = "BTCUSDT", period: str = "5m", limit: int = 200) -> list[dict]:
    """바이낸스 선물 미결제약정 히스토리 반환.

    반환: [{"sumOpenInterest": "...", "sumOpenInterestValue": "...", "timestamp": ms}, ...]
    최신 데이터가 맨 뒤 (오름차순 시간).
    """
    cache_key = f"{symbol}:{period}:{limit}"
    now = time.time()
    if cache_key in _OI_CACHE:
        data, ts = _OI_CACHE[cache_key]
        if now - ts < _OI_TTL:
            return data
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": symbol.upper(), "period": period, "limit": min(limit, 500)},
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"OI API 오류 응답: {data}")
            _OI_CACHE[cache_key] = (data, now)
            return data
    except Exception as e:
        logger.warning("[OI히스토리] 조회 실패 %s: %s", symbol, e)
        cached = _OI_CACHE.get(cache_key)
        return cached[0] if cached else []


async def get_taker_ratio_hist(symbol: str = "BTCUSDT", period: str = "5m", limit: int = 200) -> list[dict]:
    """바이낸스 선물 Taker 매수/매도 비율 히스토리 반환.

    반환: [{"buySellRatio": "...", "buyVol": "...", "sellVol": "...", "timestamp": ms}, ...]
    ratio > 1.0: 공격적 매수 우세, ratio < 1.0: 공격적 매도 우세.
    """
    cache_key = f"{symbol}:{period}:{limit}"
    now = time.time()
    if cache_key in _TAKER_CACHE:
        data, ts = _TAKER_CACHE[cache_key]
        if now - ts < _TAKER_TTL:
            return data
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://fapi.binance.com/futures/data/takerlongshortRatio",
                params={"symbol": symbol.upper(), "period": period, "limit": min(limit, 500)},
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Taker API 오류 응답: {data}")
            _TAKER_CACHE[cache_key] = (data, now)
            return data
    except Exception as e:
        logger.warning("[TakerRatio] 조회 실패 %s: %s", symbol, e)
        cached = _TAKER_CACHE.get(cache_key)
        return cached[0] if cached else []


async def get_kimchi_premium(upbit_krw_price: float, binance_symbol: str = "BTCUSDT") -> float:
    """김치프리미엄 계산 (%).

    김치프리미엄 = (업비트 원화가격 / (바이낸스 USD가격 × 환율) - 1) × 100

    양수: 한국 시장이 해외보다 비쌈 → 과열 가능성
    음수: 한국 시장이 저렴 → 저평가 가능성
    """
    cache_key = binance_symbol
    now = time.time()
    if cache_key in _KIMCHI_CACHE:
        premium, ts = _KIMCHI_CACHE[cache_key]
        if now - ts < _KIMCHI_TTL:
            return premium
    try:
        binance_usd = await get_binance_price(binance_symbol)
        usd_krw     = await get_usd_krw()
        if binance_usd <= 0 or usd_krw <= 0:
            return 0.0
        krw_equiv = binance_usd * usd_krw
        premium   = (upbit_krw_price / krw_equiv - 1) * 100
        _KIMCHI_CACHE[cache_key] = (premium, now)
        logger.debug("[김치프리미엄] %.2f%% (업비트 %.0f / 바이낸스환산 %.0f)", premium, upbit_krw_price, krw_equiv)
        return round(premium, 3)
    except Exception as e:
        logger.warning("[김치프리미엄] 계산 실패: %s", e)
        return 0.0
