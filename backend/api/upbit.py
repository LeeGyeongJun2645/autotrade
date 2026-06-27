"""업비트(Upbit) REST API 클라이언트.

공개 엔드포인트(시세): 인증 없이 호출.
비공개 엔드포인트(잔고·주문): JWT(HS256) 서명 헤더 필요.
업비트 공식 문서: https://docs.upbit.com/reference
"""

import asyncio
import hashlib
import logging
import uuid
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from backend.config import settings
from backend.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_BASE = "https://api.upbit.com/v1"
_TIMEOUT = 10
_MIN_ORDER_KRW = 5_000  # 업비트 최소 주문 금액 (원)

# 업비트 주문 API: 초당 8회 제한 (공식 10회보다 여유 있게)
_order_limiter = RateLimiter(max_calls=8, period=1.0)


def _require_upbit_keys() -> tuple[str, str]:
    """업비트 API 키 존재 여부 확인. 미설정 시 명확한 에러 발생."""
    if not settings.upbit_access_key or not settings.upbit_secret_key:
        raise RuntimeError(
            "업비트 API 키가 설정되지 않았습니다. "
            ".env 파일에 UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY 를 입력하세요."
        )
    return settings.upbit_access_key, settings.upbit_secret_key


def _auth_header(query: dict | None = None) -> dict[str, str]:
    """업비트 JWT 인증 헤더 생성.

    query 파라미터가 있는 요청(주문·잔고)은 query_hash 포함.
    업비트 보안 정책: 요청 파라미터를 urlencode → SHA-512 해시 → JWT payload 삽입.
    """
    access_key, secret_key = _require_upbit_keys()

    payload: dict[str, Any] = {
        "access_key": access_key,
        "nonce": str(uuid.uuid4()),
    }
    if query:
        encoded = urlencode(query, doseq=True).encode()
        query_hash = hashlib.sha512(encoded).hexdigest()
        payload["query_hash"] = query_hash
        payload["query_hash_alg"] = "SHA512"

    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _raise_for_upbit_error(data: Any) -> None:
    """업비트 API 오류 응답 감지 및 예외 발생."""
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise RuntimeError(f"업비트 API 오류: {err.get('message')} (name={err.get('name')})")


# ── 거래대금 상위 티커 조회 (공개) ──────────────────────────────────

_TOP_TICKERS_CACHE: tuple[list[str], float] | None = None
_TOP_TICKERS_TTL = 300.0  # 5분 캐시


async def get_top_tickers(n: int = 50) -> list[str]:
    """KRW 마켓 중 24시간 거래대금 상위 N개 티커 반환.

    업비트 전체 266개 KRW 마켓을 조회해 거래대금 순으로 정렬.
    5분 캐시로 API 호출 최소화.
    """
    global _TOP_TICKERS_CACHE
    import time
    now = time.time()
    if _TOP_TICKERS_CACHE and now - _TOP_TICKERS_CACHE[1] < _TOP_TICKERS_TTL:
        return _TOP_TICKERS_CACHE[0][:n]

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        # 1) 전체 마켓 목록
        resp = await client.get(f"{_BASE}/market/all", params={"isDetails": "false"})
        resp.raise_for_status()
        krw_markets = [m["market"] for m in resp.json() if m["market"].startswith("KRW-")]

        # 2) 100개씩 나눠서 현재가(거래대금 포함) 조회
        all_tickers: list[dict] = []
        for i in range(0, len(krw_markets), 100):
            batch = krw_markets[i:i + 100]
            r = await client.get(f"{_BASE}/ticker", params={"markets": ",".join(batch)})
            r.raise_for_status()
            all_tickers.extend(r.json())

    sorted_markets = sorted(all_tickers, key=lambda x: float(x.get("acc_trade_price_24h", 0)), reverse=True)
    top = [d["market"] for d in sorted_markets]
    _TOP_TICKERS_CACHE = (top, now)
    logger.debug("[Upbit] 거래대금 상위 %d개: %s", n, top[:n])
    return top[:n]


# ── 현재가 조회 (공개) ───────────────────────────────────────────

async def get_price(ticker: str) -> dict[str, Any]:
    """업비트 현재가(ticker) 조회.

    Args:
        ticker: 마켓 코드 (예: 'KRW-BTC', 'KRW-ETH')

    Returns:
        {
            "ticker": str,
            "current_price": float,
            "open_price": float,
            "high_price": float,
            "low_price": float,
            "prev_close": float,
            "volume": float,       # 24시간 거래량(코인 수)
            "trade_value": float,  # 24시간 거래대금(원)
            "change_rate": float,  # 전일 대비 등락률 (소수, 예: 0.03 = +3%)
        }
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/ticker", params={"markets": ticker})
        resp.raise_for_status()
        data = resp.json()

    _raise_for_upbit_error(data)

    if not data:
        raise ValueError(f"업비트에서 해당 마켓을 찾을 수 없습니다: {ticker}")

    t = data[0]
    return {
        "ticker": ticker,
        "current_price": float(t["trade_price"]),
        "open_price": float(t["opening_price"]),
        "high_price": float(t["high_price"]),
        "low_price": float(t["low_price"]),
        "prev_close": float(t["prev_closing_price"]),
        "volume": float(t["acc_trade_volume_24h"]),
        "trade_value": float(t["acc_trade_price_24h"]),
        "change_rate": float(t["signed_change_rate"]),
    }


# ── 배치 현재가 조회 (공개) ─────────────────────────────────────

async def get_prices_batch(tickers: list[str]) -> dict[str, float]:
    """여러 티커 현재가 한 번에 조회 (429 방지 — 단일 요청).

    Args:
        tickers: 마켓 코드 리스트 (예: ['KRW-BTC', 'KRW-ETH'])

    Returns:
        {ticker: current_price, ...}
    """
    if not tickers:
        return {}
    markets = ",".join(tickers)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE}/ticker", params={"markets": markets})
            resp.raise_for_status()
            return {item["market"]: float(item["trade_price"]) for item in resp.json()}
    except Exception as e:
        logger.warning("[Upbit] 배치 현재가 조회 실패: %s", e)
        return {}


# ── 호가창 조회 (공개) ───────────────────────────────────────────

async def get_orderbook(ticker: str) -> dict[str, Any]:
    """업비트 호가창(매수벽/매도벽) 분석.

    Returns:
        {
            "bid_ask_spread": float,   # (최저매도 - 최고매수) / 최고매수
            "buy_pressure":  float,    # 매수호가 총량 / (매수+매도 총량)  0~1
            "bid_wall":      float,    # 매수 최대 단일 잔량 / 총 매수 잔량
            "ask_wall":      float,    # 매도 최대 단일 잔량 / 총 매도 잔량
            "total_bid_qty": float,
            "total_ask_qty": float,
        }
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/orderbook", params={"markets": ticker})
        resp.raise_for_status()
        data = resp.json()

    if not data:
        return {}

    ob = data[0]
    units = ob.get("orderbook_units", [])
    if not units:
        return {}

    bid_prices = [u["bid_price"] for u in units]
    ask_prices = [u["ask_price"] for u in units]
    bid_sizes  = [u["bid_size"]  for u in units]
    ask_sizes  = [u["ask_size"]  for u in units]

    best_bid   = max(bid_prices) if bid_prices else 0
    best_ask   = min(ask_prices) if ask_prices else 0
    spread     = (best_ask - best_bid) / best_bid if best_bid > 0 else 0

    total_bid  = sum(bid_sizes)
    total_ask  = sum(ask_sizes)
    total_all  = total_bid + total_ask

    return {
        "bid_ask_spread": round(spread, 6),
        "buy_pressure":   round(total_bid / total_all, 4) if total_all > 0 else 0.5,
        "bid_wall":       round(max(bid_sizes) / total_bid, 4) if total_bid > 0 else 0,
        "ask_wall":       round(max(ask_sizes) / total_ask, 4) if total_ask > 0 else 0,
        "total_bid_qty":  round(total_bid, 4),
        "total_ask_qty":  round(total_ask, 4),
    }


# ── 캔들(OHLCV) 조회 (공개) ─────────────────────────────────────

def _parse_candle(c: dict) -> dict[str, Any]:
    return {
        "date":        c["candle_date_time_kst"],
        "open":        float(c["opening_price"]),
        "high":        float(c["high_price"]),
        "low":         float(c["low_price"]),
        "close":       float(c["trade_price"]),
        "volume":      float(c["candle_acc_trade_volume"]),
        "trade_value": float(c["candle_acc_trade_price"]),
    }


async def get_ohlcv(
    ticker: str,
    interval: str = "days",
    count: int = 200,
) -> list[dict[str, Any]]:
    """업비트 캔들 데이터 조회. 최신 캔들이 리스트 앞에 옴(내림차순).

    count > 200 이면 자동 페이징하여 최대 count개 수집.
    """
    url = f"{_BASE}/candles/{interval}"
    result: list[dict] = []
    to_param: str | None = None

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        while len(result) < count:
            fetch = min(200, count - len(result))
            params: dict = {"market": ticker, "count": fetch}
            if to_param:
                params["to"] = to_param

            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            _raise_for_upbit_error(data)

            if not data:
                break

            result.extend([_parse_candle(c) for c in data])

            if len(data) < fetch:
                break  # 더 이상 데이터 없음

            # 다음 페이지: 마지막 캔들의 시간 기준
            oldest = data[-1]["candle_date_time_utc"]
            to_param = oldest  # 이 시각 이전 캔들 조회

    if not result:
        raise ValueError(f"캔들 데이터 없음: {ticker} / {interval}")
    return result


# ── 잔고 조회 (비공개) ───────────────────────────────────────────

async def get_balance() -> dict[str, Any]:
    """업비트 전체 잔고 조회.

    보유 코인 현재가는 asyncio.gather 로 병렬 조회.

    Returns:
        {
            "krw": float,
            "holdings": [
                {
                    "ticker": str,           # 예: 'BTC'
                    "balance": float,        # 보유 수량
                    "avg_buy_price": float,
                    "current_value": float,  # 현재가 기준 평가액 (원)
                }
            ]
        }
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/accounts", headers=_auth_header())
        resp.raise_for_status()
        data = resp.json()

    _raise_for_upbit_error(data)

    krw = 0.0
    raw_holdings = []
    for item in data:
        currency = item["currency"]
        balance = float(item["balance"])
        if currency == "KRW":
            krw = balance
            continue
        if balance <= 0:
            continue
        raw_holdings.append({
            "ticker": currency,
            "balance": balance,
            "avg_buy_price": float(item["avg_buy_price"]),
        })

    # 보유 코인 현재가 병렬 조회
    async def _fetch_value(h: dict) -> dict:
        try:
            price_data = await get_price(f"KRW-{h['ticker']}")
            current_price = price_data["current_price"]
        except Exception:
            current_price = h["avg_buy_price"]
        return {**h, "current_value": h["balance"] * current_price}

    holdings = await asyncio.gather(*[_fetch_value(h) for h in raw_holdings])

    return {"krw": krw, "holdings": list(holdings)}


# ── 시장가 매수 (비공개) ─────────────────────────────────────────

async def place_buy_order(ticker: str, amount_krw: float) -> dict[str, Any]:
    """업비트 시장가 매수 주문.

    Args:
        ticker:     마켓 코드 (예: 'KRW-BTC')
        amount_krw: 매수에 사용할 원화 금액 (최소 5,000원)

    Returns:
        {"uuid": str, "side": "bid", "ord_type": "price", "price": float, "state": str}
    """
    if amount_krw < _MIN_ORDER_KRW:
        raise ValueError(f"업비트 최소 주문 금액은 {_MIN_ORDER_KRW:,}원입니다. 요청: {amount_krw:.0f}원")

    await _order_limiter.acquire()
    body = {
        "market": ticker,
        "side": "bid",
        "price": str(int(amount_krw)),  # 업비트는 원화 정수 권장
        "ord_type": "price",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/orders",
            json=body,
            headers=_auth_header(body),
        )
        resp.raise_for_status()
        data = resp.json()

    _raise_for_upbit_error(data)
    logger.info("[업비트 매수] %s %.0f원 | uuid: %s", ticker, amount_krw, data.get("uuid"))
    return {
        "uuid": data.get("uuid", ""),
        "side": data.get("side", ""),
        "ord_type": data.get("ord_type", ""),
        "price": float(data.get("price") or 0),
        "state": data.get("state", ""),
    }


# ── 시장가 매도 (비공개) ─────────────────────────────────────────

async def place_sell_order(ticker: str, volume: float) -> dict[str, Any]:
    """업비트 시장가 매도 주문.

    Args:
        ticker: 마켓 코드 (예: 'KRW-BTC')
        volume: 매도 수량 (코인 개수, 소수점 8자리까지)

    Returns:
        {"uuid": str, "side": "ask", "ord_type": "market", "volume": float, "state": str}
    """
    if volume <= 0:
        raise ValueError(f"매도 수량은 0보다 커야 합니다. 요청: {volume}")

    await _order_limiter.acquire()
    body = {
        "market": ticker,
        "side": "ask",
        "volume": str(volume),
        "ord_type": "market",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_BASE}/orders",
            json=body,
            headers=_auth_header(body),
        )
        resp.raise_for_status()
        data = resp.json()

    _raise_for_upbit_error(data)
    logger.info("[업비트 매도] %s %.8f | uuid: %s", ticker, volume, data.get("uuid"))
    return {
        "uuid": data.get("uuid", ""),
        "side": data.get("side", ""),
        "ord_type": data.get("ord_type", ""),
        "volume": float(data.get("volume") or 0),
        "state": data.get("state", ""),
    }


# ── 주문 상태 조회 (비공개) ──────────────────────────────────────

async def get_order(order_uuid: str) -> dict[str, Any]:
    """업비트 주문 상태 조회.

    Args:
        order_uuid: place_buy_order / place_sell_order 반환 uuid

    Returns:
        {"uuid": str, "state": str, "executed_volume": float, "paid_fee": float}
        state: 'wait' | 'watch' | 'done' | 'cancel'
    """
    query = {"uuid": order_uuid}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            f"{_BASE}/order",
            params=query,
            headers=_auth_header(query),
        )
        resp.raise_for_status()
        data = resp.json()

    _raise_for_upbit_error(data)
    return {
        "uuid": data.get("uuid", ""),
        "state": data.get("state", ""),
        "executed_volume": float(data.get("executed_volume") or 0),
        "paid_fee": float(data.get("paid_fee") or 0),
    }
