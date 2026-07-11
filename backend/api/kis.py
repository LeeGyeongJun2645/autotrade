"""한국투자증권 KIS REST API 클라이언트.

모의투자(PAPER) / 실전투자(LIVE) 모드는 config.settings.trade_mode 로 전환.
모든 HTTP 호출은 rate_limiter 를 거쳐 실행한다.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo as _ZI

_KST = _ZI("Asia/Seoul")

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


_TOKEN_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


class KISTokenManager:
    """Access Token 발급 및 자동 갱신 담당.

    base_url: 토큰 발급 서버 URL (None이면 settings.kis_base_url 사용)
    시세/조회 API 전용으로 live 서버 토큰을 별도 관리하기 위해 base_url 파라미터 추가.
    재시작 후에도 24시간 유효한 KIS 토큰을 재사용하기 위해 파일 캐시 지원.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url  # None → 런타임에 settings.kis_base_url 사용
        self._token: str | None = None
        self._expires_at: datetime = datetime.min
        self._lock = asyncio.Lock()
        # 캐시 파일은 실제 사용 URL에서 lazily 결정 (_get_cache_file 참조)
        # __init__ 시점에 base_url=None이면 settings.kis_base_url이 아직 확정 안 됐을 수 있음

    def _get_cache_file(self) -> "Path":
        """실제 사용 URL로부터 캐시 파일 경로 결정. 29443 포트 = paper, 그 외 = live."""
        base = self._base_url if self._base_url is not None else settings.kis_base_url
        _suffix = "paper" if "29443" in base else "live"
        return _TOKEN_CACHE_DIR / f"kis_token_{_suffix}.json"

    @property
    def is_valid(self) -> bool:
        """토큰이 유효한지 확인 (만료 5분 전부터 갱신 대상)."""
        return self._token is not None and datetime.now(_KST) < self._expires_at - timedelta(minutes=5)

    def _load_cached_token(self) -> bool:
        """파일 캐시에서 토큰 로드. 유효하면 True 반환."""
        try:
            _cf = self._get_cache_file()
            if not _cf.exists():
                return False
            data = json.loads(_cf.read_text())
            exp = datetime.fromisoformat(data["expires_at"])
            if datetime.now(_KST) < exp - timedelta(minutes=5):
                self._token = data["access_token"]
                self._expires_at = exp
                logger.info("KIS 토큰 캐시 로드 완료. 만료: %s", exp.strftime("%Y-%m-%d %H:%M"))
                return True
        except Exception:
            pass
        return False

    def _save_cached_token(self) -> None:
        """토큰을 파일 캐시에 저장."""
        try:
            _TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._get_cache_file().write_text(json.dumps({
                "access_token": self._token,
                "expires_at": self._expires_at.isoformat(),
            }))
        except Exception:
            pass

    async def get_token(self) -> str:
        """유효한 토큰 반환. 만료 임박 시 자동 갱신.

        lock으로 동시 다중 발급 요청(race condition)을 차단한다.
        재시작 후에는 파일 캐시 → API 발급 순으로 시도.
        """
        async with self._lock:
            if not self.is_valid:
                if not self._load_cached_token():
                    await self._issue_token()
        return self._token  # type: ignore[return-value]

    async def _issue_token(self) -> None:
        """KIS OAuth2 토큰 발급 API 호출."""
        base = self._base_url if self._base_url is not None else settings.kis_base_url
        url = f"{base}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": settings.kis_app_key,
            "appsecret": settings.kis_app_secret,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if "access_token" not in data:
            raise ValueError(f"KIS 토큰 발급 실패: {data.get('msg1') or data}")
        self._token = data["access_token"]
        # KIS 토큰 유효시간: 86400초(24시간)
        expires_in = int(data.get("expires_in", 86400))
        self._expires_at = datetime.now(_KST) + timedelta(seconds=expires_in)
        self._save_cached_token()
        logger.info("KIS 토큰 발급 완료 (%s). 만료: %s", base, self._expires_at.strftime("%Y-%m-%d %H:%M"))


# 주문/잔고용: PAPER 모드에서는 paper 서버 토큰, LIVE 모드에서는 live 서버 토큰
_token_manager = KISTokenManager()
# 시세/조회용: 항상 실전 서버 토큰 (모의투자 서버는 시세 조회 API 미지원)
_live_token_manager = KISTokenManager(base_url=settings.kis_base_url_live)


async def _kis_request(
    method: str,
    path: str,
    *,
    tr_id: str,
    params: dict | None = None,
    body: dict | None = None,
    force_live: bool = False,
) -> dict[str, Any]:
    """KIS API 공통 요청 헬퍼.

    Rate Limiter → 토큰 주입 → 헤더 구성 → HTTP 호출 순서로 실행.
    force_live=True: 시세/조회 API — PAPER 모드에서도 실전 서버 사용 (모의투자 서버 미지원)
    """
    await _rate_limiter.acquire()

    token = await (_live_token_manager if force_live else _token_manager).get_token()
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": settings.kis_app_key,
        "appsecret": settings.kis_app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }
    base = settings.kis_base_url_live if force_live else settings.kis_base_url
    url = f"{base}{path}"

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
        force_live=True,
    )
    out = data.get("output") or {}
    if not out:
        raise ValueError(f"[KIS] 주가 응답에 output 없음: {data.get('msg1', '—')}")
    return {
        "symbol": symbol,
        "current_price": int(out.get("stck_prpr") or 0),
        "open_price": int(out.get("stck_oprc") or 0),
        "high_price": int(out.get("stck_hgpr") or 0),
        "low_price": int(out.get("stck_lwpr") or 0),
        "prev_close": int(out.get("stck_sdpr") or 0),
        "volume": int(out.get("acml_vol") or 0),
        "change_rate": float(out.get("prdy_ctrt") or 0),
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
    today = datetime.now(_KST)
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
        force_live=True,
    )
    rows = (data.get("output2") or [])[:count]
    result = []
    for r in rows:
        try:
            result.append({
                "date":   r["stck_bsop_date"],
                "open":   int(r["stck_oprc"]),
                "high":   int(r["stck_hgpr"]),
                "low":    int(r["stck_lwpr"]),
                "close":  int(r["stck_clpr"]),
                "volume": int(r["acml_vol"]),
            })
        except (KeyError, ValueError):
            continue
    return result


# ── 거래량 순위 조회 ─────────────────────────────────────────────

_VOLUME_RANK_CACHE: tuple[list[str], float] | None = None
_VOLUME_RANK_TTL = 300.0  # 5분 캐시
_STOCK_NAMES_FILE = Path(__file__).parent.parent.parent / "data" / "stock_names.json"

def _load_stock_names() -> dict[str, str]:
    try:
        if _STOCK_NAMES_FILE.exists():
            return json.loads(_STOCK_NAMES_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

_STOCK_NAMES: dict[str, str] = _load_stock_names()  # code → 한글 종목명 (재시작 후에도 유지)


async def get_volume_rank(n: int = 50) -> list[str]:
    """코스피+코스닥 거래량 순위 상위 N개 종목코드 반환.

    장 중에만 의미있는 데이터를 반환. 장 마감 후에는 캐시된 값 사용.
    """
    global _VOLUME_RANK_CACHE
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
            force_live=True,
        )
        symbols = []
        updated = False
        for row in data.get("output", []):
            code = row.get("mksc_shrn_iscd", "")
            name = row.get("hts_kor_isnm", "")
            if code:
                symbols.append(code)
                if name and _STOCK_NAMES.get(code) != name:
                    _STOCK_NAMES[code] = name
                    updated = True
        if updated:
            try:
                _STOCK_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                _STOCK_NAMES_FILE.write_text(json.dumps(_STOCK_NAMES, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        if symbols:  # 빈 결과는 캐시하지 않아 다음 호출 시 재시도
            _VOLUME_RANK_CACHE = (symbols, now)
        logger.debug("[KIS] 거래량 상위 %d개: %s", n, symbols[:n])
        return symbols[:n]
    except Exception as e:
        logger.warning("[KIS] 거래량 순위 조회 실패: %s", e)
        if _VOLUME_RANK_CACHE:
            return _VOLUME_RANK_CACHE[0][:n]
        return []


def get_stock_names() -> dict[str, str]:
    """캐시된 종목코드 → 한글 종목명 매핑 반환."""
    return dict(_STOCK_NAMES)


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
            force_live=True,
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
            force_live=True,
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
    seen: set[str] = set()   # 중복 봉 방지 (마지막 봉이 다음 페이지 첫 봉으로 중복 수신)
    time_str = ""             # 빈 문자열 = 최신부터
    prev_last_key = None

    while len(result) < count:
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": time_str,
            "FID_PW_DATA_INCU_YN": "Y",
            "FID_HOUR_CLS_CODE": str(interval_min),
        }
        try:
            data = await _kis_request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                tr_id="FHKST03010200",
                params=params,
                force_live=True,
            )
        except Exception as e:
            logger.warning("[KIS] %s 분봉 조회 실패: %s", symbol, e)
            break

        rows = data.get("output2") or []
        if not rows:
            break

        for r in rows:
            try:
                bar_key = f"{r['stck_bsop_date']}_{r['stck_cntg_hour']}"
                if bar_key in seen:
                    continue
                seen.add(bar_key)
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
    output1 = data.get("output1") or []
    raw2 = data.get("output2")
    output2 = raw2[0] if isinstance(raw2, list) and raw2 else {}

    holdings = [
        {
            "symbol": h.get("pdno", ""),
            "name": h.get("prdt_name", ""),
            "qty": int(float(h.get("hldg_qty") or 0)),
            "avg_price": int(float(h.get("pchs_avg_pric") or 0)),
            "current_price": int(float(h.get("prpr") or 0)),
            "eval_profit_loss": int(float(h.get("evlu_pfls_amt") or 0)),
            "profit_rate": float(h.get("evlu_pfls_rt") or 0),
        }
        for h in output1
        if int(float(h.get("hldg_qty") or 0)) > 0
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
    out = data.get("output") or {}
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
    out = data.get("output") or {}
    logger.info("[매도] %s %d주 @ %d원 | 주문번호: %s", symbol, qty, price, out.get("ODNO"))
    return {"order_no": out.get("ODNO", ""), "order_time": out.get("ORD_TMD", "")}
