"""뉴스 감성 분석 모듈.

영어 소스: CoinDesk, CoinTelegraph, Yahoo Finance, CNBC
한국어 소스: 연합뉴스 경제, 코인데스크 KR, 구글 뉴스(키워드 검색)

코인: 업비트 마켓 API로 266개 전체 코인 이름 자동 지원
주식: 종목 코드 + 구글 뉴스로 관련 기사 수집

감성 분석:
    OPENAI_API_KEY 설정 시 → GPT-4o-mini LLM 분석 (영+한 통합)
    미설정 시              → VADER(영어) + 키워드사전(한국어) 폴백

캐시 TTL: 뉴스 1시간 / 마켓 정보 24시간
"""

import asyncio
import json
import logging
import time
import urllib.parse

import feedparser
import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.config import settings

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

# ── 감성 점수 캐시 {symbol: (score, expires_at)} ─────────────────
_score_cache: dict[str, tuple[float, float]] = {}
_SCORE_TTL = 3600  # 1시간

# ── 헤드라인 캐시 {symbol: (headlines, expires_at)} ──────────────
_headlines_cache: dict[str, tuple[list[str], float]] = {}

# ── 업비트 마켓 정보 캐시 {ticker: {korean_name, english_name}} ──
_market_info: dict[str, dict] = {}
_market_expires = 0.0
_MARKET_TTL = 86400  # 24시간

# ── RSS 피드 ─────────────────────────────────────────────────────
_EN_COIN_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
_EN_ECON_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
]
_KR_STATIC_FEEDS = [
    "https://www.yna.co.kr/rss/economy.xml",   # 연합뉴스 경제
    "https://www.coindeskkorea.com/feed/",      # 코인데스크 KR
]

# ── 한국어 감성 키워드 ────────────────────────────────────────────
_KR_POSITIVE = [
    "급등", "상승", "돌파", "호재", "신고가", "강세", "반등", "회복",
    "기대", "상승세", "매수", "긍정", "호황", "성장", "랠리", "강세장",
    "상향", "확대", "증가", "승인", "채택", "투자", "돌파구", "흑자",
    "호실적", "매출증가", "순이익", "배당", "주가상승",
]
_KR_NEGATIVE = [
    "급락", "하락", "폭락", "악재", "우려", "경고", "위험", "손실",
    "위기", "하락세", "매도", "부정", "침체", "규제", "단속", "금지",
    "사기", "해킹", "붕괴", "충격", "추락", "공포", "패닉", "적자",
    "부진", "실망", "주가하락", "파산", "청산", "매출감소", "영업손실",
]


# ── 업비트 마켓 정보 로드 ────────────────────────────────────────

async def _load_upbit_markets() -> None:
    """업비트 전체 코인 이름 정보 로드 (24시간 캐시)."""
    global _market_expires
    now = time.time()
    if now < _market_expires:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.upbit.com/v1/market/all?isDetails=true")
            resp.raise_for_status()
            markets = resp.json()
        for m in markets:
            if m["market"].startswith("KRW-"):
                _market_info[m["market"]] = {
                    "korean_name":  m.get("korean_name", ""),
                    "english_name": m.get("english_name", ""),
                    "symbol":       m["market"].replace("KRW-", ""),
                }
        _market_expires = now + _MARKET_TTL
        logger.info("[뉴스] 업비트 마켓 정보 로드: %d개", len(_market_info))
    except Exception as e:
        logger.warning("[뉴스] 업비트 마켓 정보 로드 실패: %s", e)


def _get_coin_keywords(ticker: str) -> tuple[list[str], str]:
    """코인 티커 → (영어 키워드 리스트, 한국어 검색어)."""
    info = _market_info.get(ticker)
    if info:
        symbol = info["symbol"].lower()
        en_name = info["english_name"].lower()
        kr_name = info["korean_name"]
        en_kws = list({symbol, en_name, symbol.upper()})
        return en_kws, kr_name
    symbol = ticker.replace("KRW-", "")
    return [symbol.lower(), symbol], symbol


# ── RSS 파싱 ─────────────────────────────────────────────────────

def _fetch_rss(url: str) -> list[str]:
    """RSS 피드 → 제목 리스트 (동기)."""
    try:
        feed = feedparser.parse(url)
        return [e.get("title", "") for e in feed.entries[:40]]
    except Exception as e:
        logger.debug("[뉴스] RSS 실패 (%s): %s", url, e)
        return []


def _google_news_url(query: str) -> str:
    return (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
    )


# ── LLM 감성 분석 (OpenAI GPT-4o-mini) ──────────────────────────

async def _llm_sentiment(titles: list[str], symbol: str) -> float | None:
    """GPT-4o-mini로 뉴스 헤드라인 감성 분석. -1.0 ~ +1.0 반환."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        joined = "\n".join(f"- {t}" for t in titles[:30] if t.strip())
        if not joined:
            return 0.0

        prompt = (
            f"다음은 {symbol} 관련 최신 뉴스 헤드라인입니다.\n"
            f"{joined}\n\n"
            f"위 뉴스들이 {symbol}의 단기(1~24시간) 가격에 미치는 영향을 분석해서 "
            f"-1.0(매우 부정적) ~ +1.0(매우 긍정적) 사이의 숫자를 "
            f'JSON 형식으로만 반환하세요. 형식: {{"score": 0.3}}'
        )

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=15,
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
        return max(-1.0, min(1.0, score))
    except Exception as e:
        logger.warning("[뉴스] LLM 감성 분석 실패, VADER 폴백: %s", e)
        return None


# ── VADER 폴백 감성 점수 계산 ────────────────────────────────────

def _score_en(titles: list[str], keywords: list[str]) -> tuple[float, int]:
    """영어 제목 → VADER compound 평균 + 매칭 건수."""
    scores = [
        _analyzer.polarity_scores(t)["compound"]
        for t in titles
        if any(kw in t.lower() for kw in keywords)
    ]
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


def _score_kr(titles: list[str]) -> tuple[float, int]:
    """한국어 제목 → 키워드 사전 점수 평균 + 매칭 건수."""
    scores = []
    for title in titles:
        pos = sum(1 for w in _KR_POSITIVE if w in title)
        neg = sum(1 for w in _KR_NEGATIVE if w in title)
        total = pos + neg
        if total > 0:
            scores.append((pos - neg) / total)
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


def _weighted(en: float, kr: float) -> float:
    """영어 40% + 한국어 60% 가중 평균. 한쪽만 있으면 그 값만 사용."""
    if en != 0.0 and kr != 0.0:
        return en * 0.4 + kr * 0.6
    return en or kr


# ── 메인 공개 함수 ────────────────────────────────────────────────

async def get_sentiment_score(symbol: str, token: str | None = None) -> float:  # noqa: ARG001
    """뉴스 감성 점수 반환 (-1.0 ~ +1.0).

    코인 (KRW-XXX): 업비트 전체 266개 지원, 코인 + 경제 뉴스 수집
    주식 (숫자코드): 경제 뉴스 + 구글 뉴스 검색 수집
    캐시 TTL 1시간.
    """
    now = time.time()
    if symbol in _score_cache:
        score, expires_at = _score_cache[symbol]
        if now < expires_at:
            return score

    is_coin = symbol.startswith("KRW-")

    if is_coin:
        await _load_upbit_markets()
        en_kws, kr_query = _get_coin_keywords(symbol)
    else:
        en_kws = [symbol]
        kr_query = f"{symbol} 주식"

    if is_coin:
        en_feeds = _EN_COIN_FEEDS + _EN_ECON_FEEDS
    else:
        en_feeds = _EN_ECON_FEEDS

    kr_feeds = _KR_STATIC_FEEDS + [_google_news_url(kr_query)]

    # RSS 전체 병렬 수집
    all_urls = en_feeds + kr_feeds
    results = await asyncio.gather(
        *[asyncio.to_thread(_fetch_rss, url) for url in all_urls],
        return_exceptions=True,
    )

    en_titles: list[str] = []
    for r in results[: len(en_feeds)]:
        if isinstance(r, list):
            en_titles.extend(r)

    kr_titles: list[str] = []
    for r in results[len(en_feeds) :]:
        if isinstance(r, list):
            kr_titles.extend(r)

    # ── LLM 경로 (OPENAI_API_KEY 있을 때) ──────────────────────────
    if settings.openai_api_key:
        relevant_en = [t for t in en_titles if any(kw in t.lower() for kw in en_kws)]
        all_titles = relevant_en[:20] + kr_titles[:15]
        llm_score = await _llm_sentiment(all_titles, symbol)
        if llm_score is not None:
            _score_cache[symbol] = (llm_score, now + _SCORE_TTL)
            _headlines_cache[symbol] = (all_titles, now + _SCORE_TTL)
            logger.info("[뉴스][LLM] %s → %.3f (제목 %d건)", symbol, llm_score, len(all_titles))
            return llm_score

    # ── VADER / 키워드 폴백 경로 ────────────────────────────────────
    en_score, en_cnt = _score_en(en_titles, en_kws)
    kr_score, kr_cnt = _score_kr(kr_titles)
    score = _weighted(en_score, kr_score)

    # 관련 헤드라인만 추려 캐시 (영어: 키워드 매칭, 한국어: 전체)
    relevant_en_vader = [t for t in en_titles if any(kw in t.lower() for kw in en_kws)]
    _score_cache[symbol] = (score, now + _SCORE_TTL)
    _headlines_cache[symbol] = (relevant_en_vader[:15] + kr_titles[:15], now + _SCORE_TTL)

    logger.info(
        "[뉴스][VADER] %s | 영어 %.3f(%d건) + 한국어 %.3f(%d건) → %.3f",
        symbol, en_score, en_cnt, kr_score, kr_cnt, score,
    )
    return score


# ── 통합 뉴스 피드 캐시 ──────────────────────────────────────────
_news_feed_cache: list[dict] = []
_news_feed_expires: float = 0.0
_NEWS_FEED_TTL = 600  # 10분

_DART_RSS   = "https://dart.fss.or.kr/rss/api.do?mdex=1"
_STOCK_RSS  = [
    ("연합뉴스",   "https://www.yna.co.kr/rss/economy.xml"),
    ("한국경제",   "https://www.hankyung.com/rss/economy"),
    ("머니투데이", "https://rss.mt.co.kr/mt/010/010010/010010000.xml"),
]
_COIN_RSS   = [
    ("CoinDesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("코인데스크KR",   "https://www.coindeskkorea.com/feed/"),
]

# 코인 뉴스 필터: 이 키워드 중 하나라도 포함돼야 코인 기사로 인정
_COIN_KEYWORDS = {
    # 코인 일반
    "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "altcoin", "defi",
    "nft", "web3", "token", "staking", "wallet", "exchange", "binance", "upbit",
    "coinbase", "solana", "ripple", "xrp", "cardano", "polkadot", "avalanche",
    # 한국어
    "비트코인", "이더리움", "암호화폐", "가상화폐", "블록체인", "디파이", "코인",
    "알트코인", "스테이킹", "지갑", "거래소", "업비트", "빗썸", "코빗",
    "채굴", "채굴기", "해시", "토큰", "디앱", "스마트계약", "에어드롭",
    # 주요 코인명
    "솔라나", "리플", "도지", "에이다", "폴카닷", "아발란체", "링크",
    "이오스", "트론", "라이트코인", "시바", "페페", "수이", "앱토스",
    # 시장 용어
    "반감기", "불장", "베어장", "펌핑", "덤핑", "청산", "선물", "현물",
    "김치프리미엄", "공포탐욕", "온체인", "레버리지", "펀딩", "ohlcv",
}

# 주식 뉴스 필터: 이 키워드 중 하나라도 포함돼야 주식 기사로 인정
_STOCK_KEYWORDS = {
    "주식", "증시", "코스피", "코스닥", "주가", "etf", "펀드", "채권",
    "금리", "환율", "외환", "달러", "엔화", "위안화", "fed", "fomc",
    "기업", "실적", "공시", "상장", "배당", "자사주", "ipo", "공모",
    "반도체", "삼성", "sk", "현대", "lg", "카카오", "네이버", "셀트리온",
    "증권", "은행", "보험", "금융", "투자", "펀드", "시총", "per", "pbr",
    "stock", "market", "nasdaq", "s&p", "dow", "nikkei", "kospi",
}


def _is_relevant(title: str, category: str) -> bool:
    """카테고리에 맞는 키워드가 제목에 포함돼 있는지 확인."""
    t = title.lower()
    if category == "coin":
        return any(kw in t for kw in _COIN_KEYWORDS)
    if category == "stock":
        return any(kw in t for kw in _STOCK_KEYWORDS)
    return True  # dart는 필터 없음


def _fetch_feed_items(url: str, source: str, category: str, limit: int = 25) -> list[dict]:
    """RSS → 뉴스 아이템 리스트 (동기). 카테고리별 키워드 필터 적용."""
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "AutoTrade/1.0"})
        items = []
        for e in feed.entries[:limit]:
            pub = (e.get("published") or e.get("updated") or "")[:25]
            title = e.get("title", "").strip()
            if not title:
                continue
            # 관련 없는 기사 필터링 (코인 피드에서 드라마/스포츠 기사 제거)
            if not _is_relevant(title, category):
                continue
            items.append({
                "title":     title,
                "url":       e.get("link", ""),
                "published": pub,
                "source":    source,
                "category":  category,
            })
        return items
    except Exception as exc:
        logger.debug("[뉴스피드] %s 실패: %s", source, exc)
        return []


async def get_news_feed() -> list[dict]:
    """코인+주식+DART 공시 통합 뉴스 피드 (10분 캐시).

    category 필드: 'coin' | 'stock' | 'dart'
    """
    global _news_feed_cache, _news_feed_expires
    now = time.time()
    if _news_feed_cache and now < _news_feed_expires:
        return _news_feed_cache

    feeds: list[tuple[str, str, str]] = [
        (_DART_RSS, "DART공시", "dart"),
        *[(url, src, "stock") for src, url in _STOCK_RSS],
        *[(url, src, "coin")  for src, url in _COIN_RSS],
    ]
    results = await asyncio.gather(
        *[asyncio.to_thread(_fetch_feed_items, url, src, cat) for url, src, cat in feeds],
        return_exceptions=True,
    )
    all_items: list[dict] = []
    for r in results:
        if isinstance(r, list):
            all_items.extend(r)

    all_items.sort(key=lambda x: x.get("published", ""), reverse=True)
    _news_feed_cache = all_items[:150]
    _news_feed_expires = now + _NEWS_FEED_TTL
    logger.info("[뉴스피드] %d개 수집 완료", len(_news_feed_cache))
    return _news_feed_cache


def get_headlines(symbol: str) -> list[str]:
    """캐시된 뉴스 헤드라인 반환. 캐시 만료 또는 없으면 빈 리스트."""
    entry = _headlines_cache.get(symbol)
    if entry and time.time() < entry[1]:
        return entry[0]
    return []


def get_cached_score(symbol: str) -> float | None:
    """캐시된 감성 점수 반환 (공개 API). 캐시 없거나 만료 시 None."""
    entry = _score_cache.get(symbol)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


# ── 시장 구조 데이터 캐시 (CoinGecko 무료 API) ───────────────────
_MARKET_DATA_CACHE: dict = {}
_MARKET_DATA_EXPIRES: float = 0.0
_MARKET_DATA_TTL = 600  # 10분


async def get_crypto_market_context() -> dict:
    """BTC 도미넌스, 알트코인 시즌 지수, Fear & Greed를 통합 반환.

    Returns:
        {
          "btc_dominance": float,    # BTC 시총 점유율 0~100
          "total_market_cap_usd": float,
          "altcoin_season": bool,    # True = 알트 시즌 (BTC 도미넌스 < 50%)
          "fear_greed": int,         # 0(극도공포) ~ 100(극도탐욕)
          "fear_greed_label": str,   # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
        }
    """
    global _MARKET_DATA_CACHE, _MARKET_DATA_EXPIRES
    now = time.time()
    if now < _MARKET_DATA_EXPIRES and _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE

    result: dict = {
        "btc_dominance": 58.0,
        "total_market_cap_usd": 0.0,
        "altcoin_season": False,
        "fear_greed": 50,
        "fear_greed_label": "Neutral",
    }

    async with httpx.AsyncClient(timeout=8.0) as client:
        # CoinGecko /global (무료, 인증 불필요)
        try:
            resp = await client.get("https://api.coingecko.com/api/v3/global")
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                btc_dom = data.get("market_cap_percentage", {}).get("btc", 58.0)
                result["btc_dominance"] = round(float(btc_dom), 2)
                result["total_market_cap_usd"] = float(data.get("total_market_cap", {}).get("usd", 0))
                result["altcoin_season"] = btc_dom < 50.0
        except Exception as e:
            logger.debug("[시장데이터] CoinGecko global 실패: %s", e)

        # Alternative.me Fear & Greed (무료, 인증 불필요)
        try:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            if resp.status_code == 200:
                fng_data = resp.json().get("data", [{}])[0]
                result["fear_greed"] = int(fng_data.get("value", 50))
                result["fear_greed_label"] = fng_data.get("value_classification", "Neutral")
        except Exception as e:
            logger.debug("[시장데이터] Alternative.me FNG 실패: %s", e)

    _MARKET_DATA_CACHE = result
    _MARKET_DATA_EXPIRES = now + _MARKET_DATA_TTL
    return result
