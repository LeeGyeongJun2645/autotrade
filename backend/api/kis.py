"""한국투자증권 KIS REST API 클라이언트.

모의투자(PAPER) / 실전투자(LIVE) 모드는 config.settings.trade_mode 로 전환.
모든 HTTP 호출은 rate_limiter 를 거쳐 실행한다.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from backend.config import settings
from backend.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# KIS tr_id: 모의 vs 실전 구분 테이블
_TR_ID = {
    "buy": {"PAPER": "VTTC0802U", "LIVE": "TTTC0802U"},
    "sell": {"PAPER": "VTTC0801U", "LIVE": "TTTC0801U"},
    "balance": {"PAPER": "VTTC8434R", "LIVE": "TTTC8434R"},
    "price": {"PAPER": "FHKST01010100", "LIVE": "FHKST01010100"},
}

_rate_limiter = RateLimiter(max_calls=settings.kis_rate_limit_per_sec, period=1.0)


class KISTokenManager:
    """Access Token 발급 및 자동 갱신 담당."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: datetime = datetime.min
        self._lock = asyncio.Lock()

    @property
    def is_valid(self) -> bool:
        """토큰이 유효한지 확인 (만료 5분 전부터 갱신 대상)."""
        return self._token is not None and datetime.now() < self._expires_at - timedelta(minutes=5)

    async def get_token(self) -> str:
        """유효한 토큰 반환. 만료 임박 시 자동 갱신.

        lock으로 동시 다중 발급 요청(race condition)을 차단한다.
        """
        async with self._lock:
            if not self.is_valid:
                await self._issue_token()
        return self._token  # type: ignore[return-value]

    async def _issue_token(self) -> None:
        """KIS OAuth2 토큰 발급 API 호출."""
        url = f"{settings.kis_base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": settings.kis_app_key,
            "appsecret": settings.kis_app_secret,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        self._token = data["access_token"]
        # KIS 토큰 유효시간: 86400초(24시간)
        expires_in = int(data.get("expires_in", 86400))
        self._expires_at = datetime.now() + timedelta(seconds=expires_in)
        logger.info("KIS 토큰 발급 완료. 만료: %s", self._expires_at.strftime("%Y-%m-%d %H:%M"))


_token_manager = KISTokenManager()


async def _kis_request(
    method: str,
    path: str,
    *,
    tr_id: str,
    params: dict | None = None,
    body: dict | None = None,
) -> dict[str, Any]:
    """KIS API 공통 요청 헬퍼.

    Rate Limiter → 토큰 주입 → 헤더 구성 → HTTP 호출 순서로 실행.
    """
    await _rate_limiter.acquire()

    token = await _token_manager.get_token()
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": settings.kis_app_key,
        "appsecret": settings.kis_app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }
    url = f"{settings.kis_base_url}{path}"

    async with httpx.AsyncClient(timeout=10) as client:
        if method == "GET":
            resp = await client.get(url, headers=headers, params=params)
        else:
            resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()

    data = resp.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"KIS API 오류: {data.get('msg1')} (rt_cd={data.get('rt_cd')})")

    return data


# ── 현재가 조회 ──────────────────────────────────────────────────

async def get_price(symbol: str) -> dict[str, Any]:
    """국내 주식 현재가 조회.

    Args:
        symbol: 종목코드 6자리 (예: '005930' = 삼성전자)

    Returns:
        {
            "symbol": str,
            "current_price": int,
            "open_price": int,
            "high_price": int,
            "low_price": int,
            "prev_close": int,
            "volume": int,
            "change_rate": float,  # 등락률 (%)
        }
    """
    mode = settings.trade_mode
    tr_id = _TR_ID["price"][mode]
    data = await _kis_request(
        "GET",
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        tr_id=tr_id,
        params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol},
    )
    out = data["output"]
    return {
        "symbol": symbol,
        "current_price": int(out["stck_prpr"]),
        "open_price": int(out["stck_oprc"]),
        "high_price": int(out["stck_hgpr"]),
        "low_price": int(out["stck_lwpr"]),
        "prev_close": int(out["stck_sdpr"]),
        "volume": int(out["acml_vol"]),
        "change_rate": float(out["prdy_ctrt"]),
    }


# ── 전일 OHLCV 조회 ──────────────────────────────────────────────

async def get_daily_ohlcv(symbol: str, count: int = 30) -> list[dict[str, Any]]:
    """국내 주식 일봉 데이터 조회 (변동성 돌파 / 이동평균 전략용).

    Args:
        symbol: 종목코드 6자리
        count:  가져올 봉 수 (기본 30일치)

    Returns:
        [{"date": str, "open": int, "high": int, "low": int, "close": int, "volume": int}, ...]
        최신 날짜가 리스트 앞에 옴 (내림차순)
    """
    today = datetime.now()
    # count일치 조회하려면 주말·공휴일 감안해 넉넉히 2배 범위 요청
    start = today - timedelta(days=count * 2)
    data = await _kis_request(
        "GET",
        "/uapi/domestic-stock/v1/quotations/inquire-daily-price",
        tr_id="FHKST01010400",
        params={
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_org_adj_prc": "0",
            "fid_period_div_code": "D",
            "fid_input_date_1": start.strftime("%Y%m%d"),
            "fid_input_date_2": today.strftime("%Y%m%d"),
        },
    )
    rows = data.get("output2", [])[:count]
    return [
        {
            "date": r["stck_bsop_date"],
            "open": int(r["stck_oprc"]),
            "high": int(r["stck_hgpr"]),
            "low": int(r["stck_lwpr"]),
            "close": int(r["stck_clpr"]),
            "volume": int(r["acml_vol"]),
        }
        for r in rows
    ]


# ── 거래량 순위 조회 ─────────────────────────────────────────────

_VOLUME_RANK_CACHE: tuple[list[str], float] | None = None
_VOLUME_RANK_TTL = 300.0  # 5분 캐시


async def get_volume_rank(n: int = 50) -> list[str]:
    """코스피+코스닥 거래량 순위 상위 N개 종목코드 반환.

    장 중에만 의미있는 데이터를 반환. 장 마감 후에는 캐시된 값 사용.
    """
    global _VOLUME_RANK_CACHE
    import time
    now = time.time()
    if _VOLUME_RANK_CACHE and now - _VOLUME_RANK_CACHE[1] < _VOLUME_RANK_TTL:
        return _VOLUME_RANK_CACHE[0][:n]

    try:
        data = await _kis_request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            tr_id="FHPST01710000",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            },
        )
        symbols = [row["mksc_shrn_iscd"] for row in data.get("output", []) if row.get("mksc_shrn_iscd")]
        _VOLUME_RANK_CACHE = (symbols, now)
        logger.debug("[KIS] 거래량 상위 %d개: %s", n, symbols[:n])
        return symbols[:n]
    except Exception as e:
        logger.warning("[KIS] 거래량 순위 조회 실패: %s", e)
        if _VOLUME_RANK_CACHE:
            return _VOLUME_RANK_CACHE[0][:n]
        return []


# ── 호가창 조회 ──────────────────────────────────────────────────

_ORDERBOOK_KIS_CACHE: dict[str, tuple[dict, float]] = {}
_ORDERBOOK_KIS_TTL = 5.0  # 5초 캐시


async def get_orderbook_kis(symbol: str) -> dict:
    """국내 주식 호가창 조회 (매수벽/매도벽).

    Returns:
        {
            "bid_ask_spread": float,  # (매도1호가 - 매수1호가) / 매수1호가
            "buy_pressure":   float,  # 매수잔량 / (매수+매도 총잔량)
            "bid_wall":       int,    # 최대 매수호가 잔량
            "ask_wall":       int,    # 최대 매도호가 잔량
            "total_bid_qty":  int,
            "total_ask_qty":  int,
        }
    """
    now = time.time()
    if symbol in _ORDERBOOK_KIS_CACHE:
        cached, ts = _ORDERBOOK_KIS_CACHE[symbol]
        if now - ts < _ORDERBOOK_KIS_TTL:
            return cached

    _DEFAULT = {
        "bid_ask_spread": 0.0, "buy_pressure": 0.5,
        "bid_wall": 0, "ask_wall": 0,
        "total_bid_qty": 0, "total_ask_qty": 0,
    }
    try:
        data = await _kis_request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            tr_id="FHKST01010200",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        out = data.get("output1", {})

        ask_qtys = [int(out.get(f"askp_rsqn{i}", 0) or 0) for i in range(1, 11)]
        bid_qtys = [int(out.get(f"bidp_rsqn{i}", 0) or 0) for i in range(1, 11)]
        best_ask = int(out.get("askp1", 0) or 0)
        best_bid = int(out.get("bidp1", 0) or 0)
        total_ask = int(out.get("total_askp_rsqn", 0) or 0)
        total_bid = int(out.get("total_bidp_rsqn", 0) or 0)

        result = {
            "bid_ask_spread": (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0,
            "buy_pressure": total_bid / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0.5,
            "bid_wall": max(bid_qtys) if bid_qtys else 0,
            "ask_wall": max(ask_qtys) if ask_qtys else 0,
            "total_bid_qty": total_bid,
            "total_ask_qty": total_ask,
        }
        _ORDERBOOK_KIS_CACHE[symbol] = (result, now)
        return result
    except Exception as e:
        logger.warning("[KIS] %s 호가창 조회 실패: %s", symbol, e)
        return _DEFAULT


# ── 투자자별 매매동향 ─────────────────────────────────────────────

_INVESTOR_CACHE: dict[str, tuple[dict, float]] = {}
_INVESTOR_TTL = 300.0  # 5분 캐시


async def get_investor_trend(symbol: str) -> dict:
    """기관/외국인 순매수 동향 조회.

    Returns:
        {
            "institution_net_buy": int,  # 기관 순매수량 (양수=매수우세)
            "foreign_net_buy":     int,  # 외국인 순매수량
            "individual_net_buy":  int,  # 개인 순매수량
        }
    """
    now = time.time()
    if symbol in _INVESTOR_CACHE:
        cached, ts = _INVESTOR_CACHE[symbol]
        if now - ts < _INVESTOR_TTL:
            return cached

    _DEFAULT = {"institution_net_buy": 0, "foreign_net_buy": 0, "individual_net_buy": 0}
    try:
        data = await _kis_request(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            tr_id="FHKST01010900",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        rows = data.get("output", [])
        if not rows:
            raise ValueError("빈 응답")

        today = rows[0]
        result = {
            "institution_net_buy": int(today.get("orgn_ntby_qty", 0) or 0),
            "foreign_net_buy":     int(today.get("frgn_ntby_qty", 0) or 0),
            "individual_net_buy":  int(today.get("prsn_ntby_qty", 0) or 0),
        }
        _INVESTOR_CACHE[symbol] = (result, now)
        logger.debug("[KIS] %s 투자자 동향: 기관%+d 외국인%+d", symbol,
                     result["institution_net_buy"], result["foreign_net_buy"])
        return result
    except Exception as e:
        logger.warning("[KIS] %s 투자자 동향 조회 실패: %s", symbol, e)
        return _DEFAULT


# ── 분봉 OHLCV 조회 ─────────────────────────────────────────────

async def get_minute_ohlcv(symbol: str, interval_min: int = 5, count: int = 200) -> list[dict]:
    """국내 주식 분봉 데이터 조회.

    KIS API는 1회 최대 30봉 반환 → count만큼 반복 호출.
    ML 학습 시 count=2000 등 대량 요청 가능 (1주일치 분봉 획득).

    Args:
        symbol:       종목코드 6자리
        interval_min: 분봉 단위 (1·5·15·30·60)
        count:        가져올 봉 수 (제한 없음 — API 가능 범위까지)

    Returns:
        [{"date": str, "open": int, "high": int, "low": int, "close": int, "volume": int}, ...]
        최신 봉이 앞에 옴 (내림차순)
    """
    result: list[dict] = []
    time_str = ""       # 빈 문자열 = 최신부터
    prev_last_key = None

    while len(result) < count:
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": time_str,
            "FID_PW_DATA_INCU_YN": "Y",
        }
        try:
            data = await _kis_request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                tr_id="FHKST03010200",
                params=params,
            )
        except Exception as e:
            logger.warning("[KIS] %s 분봉 조회 실패: %s", symbol, e)
            break

        rows = data.get("output2", [])
        if not rows:
            break

        for r in rows:
            try:
                result.append({
                    "date":   f"{r['stck_bsop_date']}T{r['stck_cntg_hour']}",
                    "open":   int(r["stck_oprc"]),
                    "high":   int(r["stck_hgpr"]),
                    "low":    int(r["stck_lwpr"]),
                    "close":  int(r["stck_prpr"]),
                    "volume": int(r["cntg_vol"]),
                })
            except Exception:
                continue

        last = rows[-1]
        last_key = f"{last.get('stck_bsop_date', '')}_{last.get('stck_cntg_hour', '')}"

        # 같은 커서면 무한루프 방지
        if last_key == prev_last_key:
            break
        prev_last_key = last_key

        if len(rows) < 30:
            break  # 더 이상 데이터 없음

        # 다음 페이지: 마지막 봉 시각으로 이동
        time_str = last.get("stck_cntg_hour", "")
        if not time_str:
            break

    return result[:count]


# ── 잔고 조회 ────────────────────────────────────────────────────

async def get_balance() -> dict[str, Any]:
    """주식 잔고 및 예수금 조회.

    Returns:
        {
            "cash": int,           # 주문 가능 예수금
            "total_eval": int,     # 총 평가금액
            "holdings": [
                {"symbol": str, "name": str, "qty": int,
                 "avg_price": int, "current_price": int,
                 "eval_profit_loss": int, "profit_rate": float}
            ]
        }
    """
    mode = settings.trade_mode
    tr_id = _TR_ID["balance"][mode]
    data = await _kis_request(
        "GET",
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        tr_id=tr_id,
        params={
            "CANO": settings.kis_account_no,
            "ACNT_PRDT_CD": settings.kis_account_prod_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    output1 = data.get("output1", [])
    output2 = data.get("output2", [{}])[0]

    holdings = [
        {
            "symbol": h["pdno"],
            "name": h["prdt_name"],
            "qty": int(h["hldg_qty"]),
            "avg_price": int(float(h["pchs_avg_pric"])),
            "current_price": int(h["prpr"]),
            "eval_profit_loss": int(h["evlu_pfls_amt"]),
            "profit_rate": float(h["evlu_pfls_rt"]),
        }
        for h in output1
        if int(h.get("hldg_qty", 0)) > 0
    ]

    return {
        # ord_psbl_cash_amt: 당일 실제 주문 가능 현금 (미결제 제외)
        # dnca_tot_amt: 총 예수금 (미결제 포함) — 과대 계상 위험
        "cash": int(output2.get("ord_psbl_cash_amt") or 0),
        "total_eval": int(output2.get("tot_evlu_amt") or 0),
        "holdings": holdings,
    }


# ── 매수 주문 ────────────────────────────────────────────────────

async def place_buy_order(
    symbol: str,
    qty: int,
    price: int = 0,
    order_type: str = "01",
) -> dict[str, Any]:
    """주식 매수 주문.

    Args:
        symbol:     종목코드 6자리
        qty:        주문 수량
        price:      주문 가격 (시장가=0)
        order_type: '01'=시장가, '00'=지정가

    Returns:
        {"order_no": str, "order_time": str}
    """
    mode = settings.trade_mode
    tr_id = _TR_ID["buy"][mode]
    body = {
        "CANO": settings.kis_account_no,
        "ACNT_PRDT_CD": settings.kis_account_prod_code,
        "PDNO": symbol,
        "ORD_DVSN": order_type,
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(price),
    }
    data = await _kis_request("POST", "/uapi/domestic-stock/v1/trading/order-cash", tr_id=tr_id, body=body)
    out = data["output"]
    logger.info("[매수] %s %d주 @ %d원 | 주문번호: %s", symbol, qty, price, out.get("ODNO"))
    return {"order_no": out.get("ODNO", ""), "order_time": out.get("ORD_TMD", "")}


# ── 매도 주문 ────────────────────────────────────────────────────

async def place_sell_order(
    symbol: str,
    qty: int,
    price: int = 0,
    order_type: str = "01",
) -> dict[str, Any]:
    """주식 매도 주문.

    Args:
        symbol:     종목코드 6자리
        qty:        주문 수량
        price:      주문 가격 (시장가=0)
        order_type: '01'=시장가, '00'=지정가

    Returns:
        {"order_no": str, "order_time": str}
    """
    mode = settings.trade_mode
    tr_id = _TR_ID["sell"][mode]
    body = {
        "CANO": settings.kis_account_no,
        "ACNT_PRDT_CD": settings.kis_account_prod_code,
        "PDNO": symbol,
        "ORD_DVSN": order_type,
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(price),
    }
    data = await _kis_request("POST", "/uapi/domestic-stock/v1/trading/order-cash", tr_id=tr_id, body=body)
    out = data["output"]
    logger.info("[매도] %s %d주 @ %d원 | 주문번호: %s", symbol, qty, price, out.get("ODNO"))
    return {"order_no": out.get("ODNO", ""), "order_time": out.get("ORD_TMD", "")}
