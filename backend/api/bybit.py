"""바이비트 REST API v5 클라이언트.

HMAC-SHA256 서명 기반 인증. 현물 + 선물(USDT-Perpetual) 지원.
미설정(bybit_api_key=None)이면 공개 API만 사용 가능.
"""

import hashlib
import hmac
import logging
import time

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.bybit.com"
_TIMEOUT = 10
_RECV_WINDOW = "5000"

# ── 공개 API 캐시 ──────────────────────────────────────────────────
_FUNDING_LIST_CACHE: tuple[list[dict], float] | None = None
_FUNDING_LIST_TTL = 240.0  # 4분

_SPOT_SYMBOLS_CACHE: tuple[set[str], float] | None = None
_SPOT_SYMBOLS_TTL = 3600.0  # 1시간


def _sign(payload: str) -> str:
    """HMAC-SHA256 서명."""
    secret = settings.bybit_api_secret or ""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _headers(params_str: str = "", body: str = "") -> dict:
    """인증 헤더 생성."""
    ts = str(int(time.time() * 1000))
    api_key = settings.bybit_api_key or ""
    raw = ts + api_key + _RECV_WINDOW + (params_str or body)
    sig = _sign(raw)
    return {
        "X-BAPI-API-KEY":     api_key,
        "X-BAPI-SIGN":        sig,
        "X-BAPI-SIGN-TYPE":   "2",
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
        "Content-Type":       "application/json",
    }


def _available() -> bool:
    return bool(settings.bybit_api_key and settings.bybit_api_secret)


# ── 공개 API ──────────────────────────────────────────────────────

async def get_all_funding_rates() -> list[dict]:
    """바이비트 선물 전 심볼의 현재(예정) 펀딩비 조회.

    반환: [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "nextFundingTime": ms}, ...]
    """
    global _FUNDING_LIST_CACHE
    now = time.time()
    if _FUNDING_LIST_CACHE and now - _FUNDING_LIST_CACHE[1] < _FUNDING_LIST_TTL:
        return _FUNDING_LIST_CACHE[0]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/market/tickers",
                params={"category": "linear"},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg", "unknown"))
            result = [
                {
                    "symbol":          item["symbol"],
                    "fundingRate":     float(item.get("fundingRate") or 0),
                    "nextFundingTime": int(item.get("nextFundingTime") or 0),
                    "lastPrice":       float(item.get("lastPrice") or 0),
                }
                for item in data["result"]["list"]
                if item.get("symbol", "").endswith("USDT")
            ]
            _FUNDING_LIST_CACHE = (result, now)
            return result
    except Exception as e:
        logger.warning("[Bybit] 전체 펀딩비 조회 실패: %s", e)
        return _FUNDING_LIST_CACHE[0] if _FUNDING_LIST_CACHE else []


async def get_funding_rate_history(symbol: str, limit: int = 3) -> list[dict]:
    """특정 심볼의 과거 펀딩비 히스토리 조회."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            return data["result"]["list"]
    except Exception as e:
        logger.warning("[Bybit] 펀딩비 히스토리 조회 실패 %s: %s", symbol, e)
        return []


async def get_spot_symbols() -> set[str]:
    """바이비트 현물 거래 가능 심볼 집합 (캐시 1시간)."""
    global _SPOT_SYMBOLS_CACHE
    now = time.time()
    if _SPOT_SYMBOLS_CACHE and now - _SPOT_SYMBOLS_CACHE[1] < _SPOT_SYMBOLS_TTL:
        return _SPOT_SYMBOLS_CACHE[0]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/market/instruments-info",
                params={"category": "spot", "limit": 1000},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            symbols = {
                item["symbol"]
                for item in data["result"]["list"]
                if item.get("status") == "Trading"
            }
            _SPOT_SYMBOLS_CACHE = (symbols, now)
            logger.debug("[Bybit] 현물 심볼 %d개 캐시", len(symbols))
            return symbols
    except Exception as e:
        logger.warning("[Bybit] 현물 심볼 목록 조회 실패: %s", e)
        return _SPOT_SYMBOLS_CACHE[0] if _SPOT_SYMBOLS_CACHE else set()


# ── 인증 API — 잔고 ───────────────────────────────────────────────

async def get_spot_balance(coin: str = "USDT") -> float:
    """현물 지갑 특정 코인 잔고 (availableToWithdraw)."""
    if not _available():
        return 0.0
    qs = f"accountType=SPOT&coin={coin}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/account/wallet-balance",
                params={"accountType": "SPOT", "coin": coin},
                headers=_headers(qs),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            coins = data["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == coin:
                    return float(c.get("availableToWithdraw") or 0)
            return 0.0
    except Exception as e:
        logger.warning("[Bybit] 현물 잔고 조회 실패 %s: %s", coin, e)
        return 0.0


async def get_unified_balance(coin: str = "USDT") -> float:
    """통합 마진(UNIFIED) 지갑 특정 코인 잔고."""
    if not _available():
        return 0.0
    qs = f"accountType=UNIFIED&coin={coin}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/account/wallet-balance",
                params={"accountType": "UNIFIED", "coin": coin},
                headers=_headers(qs),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            coins = data["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == coin:
                    return float(c.get("availableToWithdraw") or 0)
            return 0.0
    except Exception as e:
        logger.warning("[Bybit] 통합 잔고 조회 실패 %s: %s", coin, e)
        return 0.0


# ── 인증 API — 현물 주문 ─────────────────────────────────────────

async def spot_market_buy(symbol: str, qty: float) -> dict | None:
    """현물 시장가 매수. qty = 코인 수량."""
    if not _available():
        logger.warning("[Bybit] API 키 미설정, 현물 매수 불가")
        return None
    body = (
        f'{{"category":"spot","symbol":"{symbol}",'
        f'"side":"Buy","orderType":"Market","qty":"{qty:.6f}"}}'
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/v5/order/create",
                content=body,
                headers=_headers(body=body),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            logger.info("[Bybit] 현물 매수 완료 %s %.6f", symbol, qty)
            return data["result"]
    except Exception as e:
        logger.error("[Bybit] 현물 매수 실패 %s: %s", symbol, e)
        return None


async def spot_market_sell(symbol: str, qty: float) -> dict | None:
    """현물 시장가 매도. qty = 코인 수량."""
    if not _available():
        return None
    body = (
        f'{{"category":"spot","symbol":"{symbol}",'
        f'"side":"Sell","orderType":"Market","qty":"{qty:.6f}"}}'
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/v5/order/create",
                content=body,
                headers=_headers(body=body),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            logger.info("[Bybit] 현물 매도 완료 %s %.6f", symbol, qty)
            return data["result"]
    except Exception as e:
        logger.error("[Bybit] 현물 매도 실패 %s: %s", symbol, e)
        return None


# ── 인증 API — 선물 주문 (USDT-Perpetual) ───────────────────────

async def futures_open_short(symbol: str, qty: float) -> dict | None:
    """선물 숏 포지션 진입 (시장가). qty = 계약 수량(코인 수)."""
    if not _available():
        return None
    body = (
        f'{{"category":"linear","symbol":"{symbol}",'
        f'"side":"Sell","orderType":"Market","qty":"{qty:.4f}",'
        f'"positionIdx":2}}'  # 양방향 모드 숏
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/v5/order/create",
                content=body,
                headers=_headers(body=body),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            logger.info("[Bybit] 선물 숏 진입 완료 %s %.4f", symbol, qty)
            return data["result"]
    except Exception as e:
        logger.error("[Bybit] 선물 숏 진입 실패 %s: %s", symbol, e)
        return None


async def futures_close_short(symbol: str, qty: float) -> dict | None:
    """선물 숏 포지션 청산 (시장가 Buy)."""
    if not _available():
        return None
    body = (
        f'{{"category":"linear","symbol":"{symbol}",'
        f'"side":"Buy","orderType":"Market","qty":"{qty:.4f}",'
        f'"positionIdx":2,"reduceOnly":true}}'
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/v5/order/create",
                content=body,
                headers=_headers(body=body),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            logger.info("[Bybit] 선물 숏 청산 완료 %s %.4f", symbol, qty)
            return data["result"]
    except Exception as e:
        logger.error("[Bybit] 선물 숏 청산 실패 %s: %s", symbol, e)
        return None


async def get_futures_position(symbol: str) -> dict | None:
    """선물 특정 심볼 숏 포지션 조회."""
    if not _available():
        return None
    qs = f"category=linear&symbol={symbol}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/v5/position/list",
                params={"category": "linear", "symbol": symbol},
                headers=_headers(qs),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg"))
            for pos in data["result"]["list"]:
                if pos.get("side") == "Sell" and float(pos.get("size") or 0) > 0:
                    return pos
            return None
    except Exception as e:
        logger.warning("[Bybit] 선물 포지션 조회 실패 %s: %s", symbol, e)
        return None


async def set_position_mode(symbol: str) -> None:
    """양방향 포지션 모드 설정 (hedge mode)."""
    if not _available():
        return
    body = f'{{"category":"linear","symbol":"{symbol}","mode":3}}'
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_BASE}/v5/position/switch-mode",
                content=body,
                headers=_headers(body=body),
            )
            data = resp.json()
            # 이미 설정된 경우 retCode=110025, 무시
            if data.get("retCode") not in (0, 110025):
                logger.warning("[Bybit] 양방향 모드 설정 실패 %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        logger.warning("[Bybit] 양방향 모드 설정 예외 %s: %s", symbol, e)
