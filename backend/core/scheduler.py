"""APScheduler 기반 자동매매 스케줄러.

KIS(주식): 평일 09:00~15:29 매분 실행 | 15:20 변동성 돌파 강제 매도
Upbit(코인): 24/7 5분마다 실행

포지션 상태는 인메모리(_positions)에 보관.
7단계(models/trade.py) 완성 후 DB 기록 추가 예정.
"""

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.api import kis, telegram, upbit
from backend.config import settings
from backend.core.daily_report import send_daily_report
from backend.core.risk_manager import Position, RiskManager
from backend.core import sim_log
from backend.models.trade import delete_position, insert_trade, load_all_positions, upsert_position
from backend.strategies import StrategyResult
from backend.strategies import moving_average as ma_strategy
from backend.strategies import rsi as rsi_strategy
from backend.strategies import volatility_breakout as vb_strategy

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

# ── 주요 매크로 이벤트 (KST): 발표 ±2시간 코인 신규 매수 차단 ─────
_MACRO_EVENTS_KST = [
    # FOMC 2026 (미 동부 14:00 = KST 03:00/04:00)
    "2026-07-29 04:00", "2026-09-17 04:00", "2026-10-29 04:00", "2026-12-10 04:00",
    # US CPI 2026 (미 동부 08:30 = KST 21:30/22:30)
    "2026-07-15 22:30", "2026-08-12 22:30", "2026-09-11 22:30",
    "2026-10-14 22:30", "2026-11-12 22:30", "2026-12-10 22:30",
    # US NFP 비농업고용 2026 (보통 첫째 금요일)
    "2026-07-02 22:30", "2026-08-07 22:30", "2026-09-04 22:30",
    "2026-10-02 22:30", "2026-11-06 22:30", "2026-12-04 22:30",
]


def _is_high_risk_window() -> bool:
    """현재 KST 시각이 주요 매크로 이벤트 발표 ±2시간 이내인지 확인."""
    now = datetime.now(KST).replace(tzinfo=None)
    for es in _MACRO_EVENTS_KST:
        ev = datetime.strptime(es, "%Y-%m-%d %H:%M")
        if abs((now - ev).total_seconds()) <= 7200:
            return True
    return False


# ── 저승률 종목 정적 블랙리스트 ─────────────────────────────────────
# 거래내역 분석에서 누적 승률 < 30% + 군집손절 재발 확인된 종목
_TICKER_BLACKLIST: frozenset[str] = frozenset({
    "KRW-HP",    "KRW-SLX",
    "KRW-TAIKO", "KRW-IN",   "KRW-BREV",
    "KRW-DOGE",  "KRW-WET",  "KRW-USDT",
    "KRW-ID",    "KRW-JTO",  "KRW-DKA",
    "KRW-EDGE",  "KRW-MET2", "KRW-IOTA",
    "KRW-SONIC", "KRW-SAND",
    # 07-04 추가
    "KRW-BLAST", "KRW-ELF",  "KRW-SAHARA",
    "KRW-SOL",   "KRW-MANA",
    # 07-05 추가: EV 음수 + 저승률 확인
    "KRW-PEPE",  "KRW-BERA", "KRW-SUI",
    "KRW-BCH",   "KRW-VTHO",
    "KRW-AI",    "KRW-MIRA",
    # 07-05 2차 추가: 누적 EV < -0.1% 확인 종목
    "KRW-CHIP",  "KRW-ONDO", "KRW-NEAR",
    "KRW-MEGA",  "KRW-GAME2",
    # 07-06 추가: 거래내역 분석 WR<35% 또는 대손절 반복 종목
    "KRW-LINK",  "KRW-AWE",  "KRW-TRUMP",
    "KRW-ELSA",  "KRW-GAS",  "KRW-RED",
    "KRW-BIRB",  "KRW-ZBT",  "KRW-AQT",
})

# inquire-price 500 에러 유발 종목 (ELW, 구조화상품 등 — 거래량 상위 노출되나 API 지원 안됨)
_STOCK_PRICE_BLACKLIST: frozenset[str] = frozenset({
    "198440", "379800",
})

# ── 글로벌 크로스-에이전트 상태 (모든 에이전트 공유) ──────────────────────
# 에이전트별 독립 cooldown의 한계: AI01 손절 → AI02~AI22는 즉시 재진입 가능
# 해결: 모듈 레벨 공유 상태로 전체 에이전트 동시 차단

_GLOBAL_COOLDOWN: dict[str, str] = {}       # ticker → 만료 ISO 시각(KST)
_GLOBAL_LOSS_COUNT: dict[str, int] = {}      # ticker → 전체 에이전트 손절 누적 횟수
_TICKER_STATS: dict[str, dict] = {}          # ticker → {"wins":0, "total":0}
_RECENT_LOSSES: list[tuple] = []             # (ticker, time.time()) — 클러스터 감지용


def _set_global_cooldown(symbol: str, minutes: int = 60) -> None:
    """손절 시 전체 에이전트에 쿨다운 적용 (기본 60분)."""
    from datetime import datetime as _dt, timedelta as _td
    until = (_dt.now(KST) + _td(minutes=minutes)).isoformat()
    _GLOBAL_COOLDOWN[symbol] = until
    _GLOBAL_LOSS_COUNT[symbol] = _GLOBAL_LOSS_COUNT.get(symbol, 0) + 1
    # 10회 이상 손절 → 정적 블랙리스트 추가 권고 로그
    cnt = _GLOBAL_LOSS_COUNT[symbol]
    if cnt in (5, 10, 20):
        logger.warning("[글로벌차단] %s 전체 손절 %d회 → 블랙리스트 추가 검토 권고", symbol, cnt)


def _is_global_cooldown(symbol: str) -> bool:
    """전체 에이전트 공유 쿨다운 체크."""
    until = _GLOBAL_COOLDOWN.get(symbol)
    if not until:
        return False
    from datetime import datetime as _dt
    if _dt.now(KST).isoformat() < until:
        return True
    _GLOBAL_COOLDOWN.pop(symbol, None)
    return False


def _update_ticker_stats(symbol: str, profit: float) -> None:
    """매도 완료 시 전체 에이전트 집계 티커 승률 업데이트."""
    s = _TICKER_STATS.setdefault(symbol, {"wins": 0, "total": 0})
    s["total"] += 1
    if profit > 0:
        s["wins"] += 1
    import time
    if profit < -0.003:
        _RECENT_LOSSES.append((symbol, time.time()))
        if len(_RECENT_LOSSES) > 200:
            _RECENT_LOSSES.pop(0)


def _is_cluster_stop() -> bool:
    """최근 30분 안에 5회 이상 손절 → 전체 시장 위기 → 신규 매수 전면 차단."""
    import time
    now = time.time()
    recent = [ts for _, ts in _RECENT_LOSSES if now - ts < 1800]
    return len(recent) >= 5


def _is_blacklisted(symbol: str, agent) -> bool:
    """정적 블랙리스트 + 글로벌 동적 저승률 차단."""
    if symbol in _TICKER_BLACKLIST:
        return True
    # 글로벌 집계 기준 동적 차단 (인메모리 30개 한도 문제 해소)
    s = _TICKER_STATS.get(symbol)
    if s and s["total"] >= 5:
        if s["wins"] / s["total"] < 0.25:
            return True
    # 폴백: 에이전트 recent_trades 기반 (서버 재시작 직후)
    ticker_sells = [
        t for t in agent.recent_trades
        if t.get("ticker") == symbol and t.get("action") == "SELL"
    ]
    if len(ticker_sells) >= 3:
        win_cnt = sum(1 for t in ticker_sells if (t.get("profit_rate") or 0) > 0)
        if win_cnt / len(ticker_sells) < 0.25:
            return True
    return False


def _is_cooldown(symbol: str, agent) -> bool:
    """손절 후 30분 쿨다운 (에이전트 개별)."""
    until = agent._cooldown_tickers.get(symbol)
    if not until:
        return False
    from datetime import datetime as _dt
    now_str = _dt.now(KST).isoformat()
    if now_str < until:
        return True
    agent._cooldown_tickers.pop(symbol, None)
    return False


def _set_cooldown(symbol: str, agent) -> None:
    """손절 확정 시 해당 에이전트 30분 쿨다운 등록."""
    from datetime import datetime as _dt, timedelta as _td
    until = (_dt.now(KST) + _td(minutes=30)).isoformat()
    agent._cooldown_tickers[symbol] = until


def _compute_r_ratio(trades: list[dict]) -> float:
    """최근 거래 결과에서 R비율(avg_win / avg_loss) 계산."""
    wins  = [t["profit_rate"] for t in trades if (t.get("profit_rate") or 0) > 0]
    losses = [abs(t["profit_rate"]) for t in trades if (t.get("profit_rate") or 0) < 0]
    if not wins or not losses:
        return 1.0
    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))


_DAILY_LOSS_LIMIT = 0.03  # 일일 손실 한도 3% (에이전트별 독립 서킷 브레이커)

def _is_daily_loss_exceeded(agent) -> bool:
    """오늘 손실이 DAILY_LOSS_LIMIT(3%)를 넘으면 신규 매수 차단."""
    from datetime import datetime as _dt
    kst_today = _dt.now(KST).strftime("%Y-%m-%d")
    today_pnl = sum(
        (t.get("profit_rate") or 0)
        for t in agent.recent_trades
        if t.get("traded_at", "")[:10] == kst_today and t.get("action") == "SELL"
    )
    return today_pnl < -_DAILY_LOSS_LIMIT


def _is_coin_night_risk() -> bool:
    """코인 저승률 시간대 차단 (거래내역 분석 기반).

    KST 01~04시: WR 30.8~41.4% (야간 저유동성)
    KST 05~06시: 미국 장 마감 후 (UTC 20~21시) 유동성 공백
    KST 09시: WR=35.5% (아시아 개장 노이즈)
    KST 12시: WR=37.7% (점심 저유동성)
    KST 17시: WR=33.3% (실거래 확인)
    """
    h = datetime.now(KST).hour
    return h in {1, 2, 3, 4, 5, 6, 9, 12, 17}


def _is_stock_open_noise() -> bool:
    """주식 개장 첫 30분 (KST 09:00~09:30) — KOSPI 연구: 개장 직후 30분 수익률 음수."""
    now = datetime.now(KST)
    return now.hour == 9 and now.minute < 30
_OHLCV_SEM = asyncio.Semaphore(5)  # 업비트/KIS OHLCV 동시 요청 최대 5개 (429 방지)

# 시장 구조 캐시 (BTC 도미넌스, Fear & Greed): 10분 TTL, 전역 공유
_scheduler_mkt_cache: dict = {}
_scheduler_mkt_expires: float = 0.0


async def _refresh_market_context() -> None:
    """BTC 도미넌스 + Fear & Greed 갱신 (10분마다 스케줄러에서 호출)."""
    global _scheduler_mkt_cache, _scheduler_mkt_expires
    import time
    if time.time() < _scheduler_mkt_expires:
        return
    try:
        from backend.ml.news import get_crypto_market_context
        ctx = await get_crypto_market_context()
        _scheduler_mkt_cache["ctx"] = ctx
        _scheduler_mkt_expires = time.time() + 600
        logger.info(
            "[시장] BTC도미넌스=%.1f%% F&G=%d(%s) 알트시즌=%s",
            ctx["btc_dominance"], ctx["fear_greed"], ctx["fear_greed_label"],
            "YES" if ctx["altcoin_season"] else "NO",
        )
    except Exception as e:
        logger.debug("[시장] 시장 컨텍스트 갱신 실패: %s", e)


class TradingScheduler:
    """자동매매 스케줄러.

    add_kis_symbol / add_upbit_ticker 로 감시 종목을 등록하고
    start() 를 호출하면 자동매매가 시작된다.
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler(timezone=KST)
        self._positions: dict[str, Position] = {}
        self._kis_symbols: list[str] = []
        self._upbit_tickers: list[str] = []
        self._lock = asyncio.Lock()
        self._ml_signals: dict[str, dict] = {}  # {symbol: {"signal": ..., "buy_prob": ..., "news_score": ...}}
        self._use_atr_risk: bool = False  # 에이전트 검증 완료 시 True → 실매매도 ATR 동적 손익 적용
        self._btc_oi_hist:    list[dict] = []  # BTC 선물 OI 히스토리 캐시 (코인 에이전트 전체 공유)
        self._btc_taker_hist: list[dict] = []  # BTC 선물 Taker 비율 히스토리 캐시
        self._btc_ls_hist:    list[dict] = []  # BTC 글로벌 L/S 비율 히스토리 캐시 (코인 전용)
        self._btc_ohlcv_cache: list[dict] = []  # BTC 5분봉 OHLCV 캐시 (ML 게이트 BTC 상관관계용)
        self._obi_snaps: list[float] = []       # BTC 오더북 OBI 스냅샷 (매 5분, 최대 12개 = 60분봉 1개)
        self._stock_train_cache: dict[str, list] = {}  # 최근 틱에서 수집한 주식 OHLCV (재학습 재사용)
        self._kospi_train_cache: list[dict] = []       # 최근 틱 KOSPI OHLCV (재학습 재사용)
        # DCA 분할매수 상태: {symbol: {"stage": 1~3, "total_budget": float}}
        # stage 1=1차완료, 2=2차완료, 3=3차완료(전량매수)
        self._dca_state: dict[str, dict] = {}
        # 실제 포지션 부분 청산 상태 (Upbit/KIS 실물)
        self._partial_tp_state: dict[str, dict] = {}  # ticker → {"target": price, "done": bool}
        # 매수 진행 중 심볼 추적 (TOCTOU 중복매수 방지: 주문 후 포지션 등록 전 간격 보호)
        self._pending_buys: set[str] = set()
        # KOSPI MIM (Morning Intraday Momentum): 개장 30분 방향 → 당일 필터
        self._morning_direction: int = 0      # +1=상승, -1=하락, 0=미결정
        self._morning_direction_date: str = ""
        # 외국인 수급 캐시 (5분 TTL): {symbol: (data_dict, timestamp)}
        self._investor_cache: dict[str, tuple[dict, float]] = {}

    # ── 종목 관리 ─────────────────────────────────────────────────

    def add_kis_symbol(self, symbol: str) -> None:
        if symbol not in self._kis_symbols:
            self._kis_symbols.append(symbol)
            logger.info("KIS 감시 종목 추가: %s", symbol)

    def remove_kis_symbol(self, symbol: str) -> None:
        self._kis_symbols = [s for s in self._kis_symbols if s != symbol]

    def add_upbit_ticker(self, ticker: str) -> None:
        if ticker not in self._upbit_tickers:
            self._upbit_tickers.append(ticker)
            logger.info("Upbit 감시 티커 추가: %s", ticker)

    def remove_upbit_ticker(self, ticker: str) -> None:
        self._upbit_tickers = [t for t in self._upbit_tickers if t != ticker]

    def get_positions(self) -> dict[str, Position]:
        """현재 보유 포지션 스냅샷 반환."""
        return dict(self._positions)

    def get_ml_signals(self) -> dict[str, dict]:
        """최근 ML 신호 스냅샷 반환."""
        return dict(self._ml_signals)

    # ── 스케줄러 생명주기 ──────────────────────────────────────────

    async def restore_positions(self) -> None:
        """DB에서 포지션 복구 후 거래소 실잔고 비교로 좀비 포지션 제거."""
        saved = await load_all_positions()
        async with self._lock:
            for symbol, (_, pos) in saved.items():
                if symbol not in self._positions:
                    self._positions[symbol] = pos

        # ── 좀비 포지션 검증 (서버 재시작 사이 청산된 포지션 제거) ──
        await self._validate_positions()
        # ── 에이전트 stats 복구 (잔액·거래수·승률·is_active·buy_threshold) ──
        await self._restore_agent_stats()

    async def _restore_agent_stats(self) -> None:
        """서버 재시작 시 DB에서 에이전트 상태 복구 (잔액·거래수·is_active·buy_threshold)."""
        from backend.ml.agents import AGENTS
        from backend.db.database import connect_db
        import aiosqlite

        try:
            async with connect_db() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM agent_stats") as cur:
                    stats_rows = {r["agent_id"]: dict(r) async for r in cur}

                # 실거래 기준 total_trades/win_trades 재계산 (stats 불일치 보정)
                async with db.execute("""
                    SELECT agent_id,
                           COUNT(*) as total,
                           SUM(CASE WHEN profit_rate > 0 THEN 1 ELSE 0 END) as wins
                    FROM agent_trades WHERE action='SELL'
                    GROUP BY agent_id
                """) as cur:
                    trade_counts = {r["agent_id"]: (int(r["total"]), int(r["wins"])) async for r in cur}

                for agent_id, row in stats_rows.items():
                    if agent_id in trade_counts:
                        row["total_trades"], row["win_trades"] = trade_counts[agent_id]

                async with db.execute(
                    "SELECT * FROM agent_positions"
                ) as cur:
                    pos_rows: dict[str, list] = {}
                    async for r in cur:
                        pos_rows.setdefault(r["agent_id"], []).append(dict(r))

                async with db.execute(
                    "SELECT * FROM agent_trades ORDER BY traded_at DESC LIMIT 1000"
                ) as cur:
                    trade_rows: dict[str, list] = {}
                    async for r in cur:
                        aid = r["agent_id"]
                        if len(trade_rows.get(aid, [])) < 30:
                            trade_rows.setdefault(aid, []).append(dict(r))

                # 오늘 BUY 건수 복원 (재시작 시 _today_buy_count 초기화 방지)
                from datetime import datetime as _dt2
                _today_kst_str = _dt2.now(KST).strftime("%Y-%m-%d")
                async with db.execute("""
                    SELECT agent_id, COUNT(*) as cnt
                    FROM agent_trades
                    WHERE action = 'BUY'
                      AND DATE(traded_at) = ?
                    GROUP BY agent_id
                """, (_today_kst_str,)) as cur:
                    today_buy_counts = {r["agent_id"]: int(r["cnt"]) async for r in cur}

                # 재시작 후 _partial_tp_done 복원: 당일 SELL_PARTIAL 거래 → 중복 청산 방지
                async with db.execute("""
                    SELECT agent_id, ticker
                    FROM agent_trades
                    WHERE action = 'SELL_PARTIAL'
                      AND DATE(traded_at) = ?
                """, (_today_kst_str,)) as cur:
                    partial_done_rows: dict[str, set[str]] = {}
                    async for r in cur:
                        partial_done_rows.setdefault(r["agent_id"], set()).add(r["ticker"])

                # 동적 블랙리스트 복원 (재시작 후 00:00 전까지 블랙리스트 소실 방지)
                async with db.execute("""
                    SELECT agent_id, ticker,
                           CAST(SUM(CASE WHEN profit_rate > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as wr,
                           COUNT(*) as cnt
                    FROM agent_trades
                    WHERE action = 'SELL' AND profit_rate IS NOT NULL
                    GROUP BY agent_id, ticker
                    HAVING cnt >= 20
                """) as cur:
                    restored_bl: dict[str, set[str]] = {}
                    async for r in cur:
                        if r["wr"] < 0.35:
                            restored_bl.setdefault(r["agent_id"], set()).add(r["ticker"])

            from backend.ml.agents import ENSEMBLE_AGENTS
            from datetime import datetime as _dt3
            _today_kst_str2 = _dt3.now(KST).strftime("%Y-%m-%d")
            for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
                stats = stats_rows.get(agent.agent_id)
                if stats:
                    agent.restore_from_db(
                        stats,
                        pos_rows.get(agent.agent_id, []),
                        trade_rows.get(agent.agent_id, []),
                    )
                    # 오늘 BUY 건수 복원
                    agent._today_buy_count = today_buy_counts.get(agent.agent_id, 0)
                    agent._today_date = _today_kst_str2
                    # day_start_balance 복원: DB에 저장된 값 사용, 0이면 현재 잔액으로 초기화
                    _dsb = float(stats.get("day_start_balance") or 0.0)
                    agent.day_start_balance = _dsb if _dsb > 0 else (agent._balance + agent.position_value)
                    # _dynamic_tp_mult 복원 (재시작 후 3.0 초기화 방지)
                    agent._dynamic_tp_mult = float(stats.get("dynamic_tp_mult") or 3.0)
                    # _partial_tp_done 복원: 당일 부분청산 완료 종목 마킹 (중복청산 방지)
                    agent._partial_tp_done = partial_done_rows.get(agent.agent_id, set())
                    # _peak_price 초기화: 재시작 시 보유 포지션의 entry_price로 보수적 초기화
                    # (정확한 최고가 알 수 없으므로 진입가 기준으로 트레일링 재시작)
                    for ticker, pos in agent._positions.items():
                        if ticker not in agent._peak_price:
                            agent._peak_price[ticker] = pos.entry_price
                        # _trailing_mode: 부분청산이 완료된 종목은 이미 트레일링 구간이었음
                        if ticker in agent._partial_tp_done:
                            agent._trailing_mode.add(ticker)
                        # _partial_tp_price: 아직 부분청산 안 된 종목은 1.5% 목표가로 보수 복원
                        # (ATR 재계산 불가 → ATR 평균치 근사: entry×1.015)
                        elif ticker not in agent._partial_tp_price:
                            agent._partial_tp_price[ticker] = pos.entry_price * 1.015
                    # 동적 블랙리스트 복원
                    agent._ticker_blacklist = restored_bl.get(agent.agent_id, set())
            logger.info("[복구] 에이전트 stats %d개 복구 완료 (오늘BUY: %s)", len(stats_rows), dict(list(today_buy_counts.items())[:5]))
        except Exception:
            logger.exception("[복구] 에이전트 stats 복구 실패")

        # 코인 에이전트 펀딩비 즉시 로드 (재시작 후 06:05 daily_retrain 전까지 0.0 방지)
        try:
            from backend.api.binance import get_historical_funding_rates
            from backend.ml.agents import AGENTS as _AGENTS_FR, ENSEMBLE_AGENTS as _EA_FR
            _funding_init = await get_historical_funding_rates("BTCUSDT", limit=50)
            if _funding_init:
                for _a in list(_AGENTS_FR.values()) + list(_EA_FR.values()):
                    if _a.market == "coin" and not _a._cached_funding_rates:
                        _a._cached_funding_rates = _funding_init
                logger.info("[복구] 펀딩비 초기 로드 완료 (%d개)", len(_funding_init))
        except Exception:
            logger.warning("[복구] 펀딩비 초기 로드 실패 — 0.0 유지")

        # 재시작 후에도 ATR 전환 조건 재평가 (인메모리 플래그 복구)
        await self._check_atr_upgrade()

    async def _validate_positions(self) -> None:
        """거래소 실잔고와 DB 포지션 비교 — 불일치 포지션(좀비) 자동 제거."""
        zombies: list[str] = []

        # 업비트 코인 좀비 검증
        try:
            if settings.upbit_access_key:
                bal = await upbit.get_balance()
                holdings = {
                    h["ticker"]: float(h.get("balance", 0))
                    for h in bal.get("holdings", [])
                }
                async with self._lock:
                    for symbol, pos in list(self._positions.items()):
                        if not symbol.startswith("KRW-"):
                            continue
                        currency = symbol.replace("KRW-", "")
                        actual_qty = holdings.get(currency, 0.0)
                        if actual_qty < pos.qty * 0.9:  # 10% 이상 차이 → 좀비
                            zombies.append(symbol)
        except Exception:
            logger.warning("[좀비검증] 업비트 잔고 조회 실패, 건너뜀")

        # KIS 주식 좀비 검증
        try:
            if settings.kis_app_key and settings.kis_app_key != "test":
                kis_bal = await kis.get_balance()
                kis_holdings = {h["symbol"]: int(h.get("qty", 0)) for h in kis_bal.get("holdings", [])}
                async with self._lock:
                    for symbol, pos in list(self._positions.items()):
                        if symbol.startswith("KRW-"):
                            continue
                        actual_qty = kis_holdings.get(symbol, 0)
                        if actual_qty < pos.qty * 0.9:  # 10% 이상 차이 → 좀비
                            zombies.append(symbol)
        except Exception:
            logger.warning("[좀비검증] KIS 잔고 조회 실패, 건너뜀")

        for symbol in zombies:
            logger.warning("[좀비포지션] %s DB에 있으나 실잔고 없음 → 제거", symbol)
            async with self._lock:
                self._positions.pop(symbol, None)
                self._partial_tp_state.pop(symbol, None)
            await delete_position(symbol)
            sim_log.push(symbol, f"좀비 포지션 제거 (서버 재시작 사이 청산된 것으로 추정)", "SELL")

    def start(self) -> None:
        """스케줄러 시작. FastAPI lifespan 에서 호출."""
        # KIS: 평일 09:00~15:59 매분 (15:30 이후는 _kis_tick 내부에서 스킵)
        self._scheduler.add_job(
            self._kis_tick,
            CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*", timezone=KST),
            id="kis_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # KIS: 15:20 변동성 돌파 EOD 강제 매도
        self._scheduler.add_job(
            self._kis_eod_sell,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=20, timezone=KST),
            id="kis_eod",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # Upbit: 24/7 5분마다
        self._scheduler.add_job(
            self._upbit_tick,
            CronTrigger(minute="*/5", timezone=KST),
            id="upbit_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # AI 에이전트 경쟁 시뮬레이션: 30분마다 (코인 60분봉/주식 15분봉 → 30분 간격 충분)
        self._scheduler.add_job(
            self._agent_tick,
            CronTrigger(minute="*/30", timezone=KST),
            id="agent_tick",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 코인 일일 현황 레포트: 매일 06:00 KST
        self._scheduler.add_job(
            self._send_daily_coin_report,
            CronTrigger(hour=6, minute=0, timezone=KST),
            id="daily_coin_report",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 전 에이전트 일일 재학습: 매일 06:07 (업비트 틱 06:05 종료 후, 5분 경계 회피)
        self._scheduler.add_job(
            self._daily_retrain,
            CronTrigger(hour=6, minute=7, timezone=KST),
            id="daily_retrain",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 주식 일일 현황 레포트: 평일 15:35 KST (장 마감 후)
        self._scheduler.add_job(
            self._send_daily_stock_report,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=35, timezone=KST),
            id="daily_stock_report",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 포트폴리오 일별 스냅샷: 매일 16:00 KST (장 마감 30분 후)
        self._scheduler.add_job(
            self._portfolio_snapshot,
            CronTrigger(hour=16, minute=0, timezone=KST),
            id="portfolio_snapshot",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 일일 초기화: 매일 00:00 KST — day_start_balance 스냅샷 + 동적 블랙리스트 업데이트
        self._scheduler.add_job(
            self._daily_reset,
            CronTrigger(hour=0, minute=0, timezone=KST),
            id="daily_reset",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # BTC 도미넌스 + Fear & Greed 갱신: 10분마다 (코인 보유 포지션 청산 판단용)
        self._scheduler.add_job(
            _refresh_market_context,
            CronTrigger(minute="*/10", timezone=KST),
            id="market_context",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # 뉴스 감성 점수 정기 갱신: 30분마다 (TTL=1시간, 최대 1.5시간 이내 최신 유지)
        # 기존: _agent_execute에서 get_cached_score()만 사용 → 캐시 없으면 영구 0.0
        # 개선: 선제적으로 모니터링 종목 뉴스 주기적 갱신 → news_score 피처 실데이터화
        self._scheduler.add_job(
            self._refresh_news_scores,
            CronTrigger(minute="*/30", timezone=KST),
            id="news_refresh",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info(
            "스케줄러 시작 | KIS: %s | Upbit: %s",
            self._kis_symbols,
            self._upbit_tickers,
        )

    def stop(self) -> None:
        """스케줄러 중지. FastAPI lifespan shutdown 에서 호출."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("스케줄러 중지")

    # ── KIS 틱 ────────────────────────────────────────────────────

    async def _kis_tick(self) -> None:
        now = datetime.now(KST)
        # 장 마감(15:30) 이후 tick 무시 — cron hour="9-15" 이므로 15:30~15:59 구간 방어
        if now.hour == 15 and now.minute >= 30:
            return

        for symbol in list(self._kis_symbols):
            try:
                await self._process_kis_symbol(symbol)
            except Exception:
                logger.exception("[KIS][%s] 틱 처리 중 예외", symbol)

    async def _process_kis_symbol(self, symbol: str) -> None:
        price_data = await kis.get_price(symbol)
        current_price = float(price_data["current_price"])

        async with self._lock:
            position = self._positions.get(symbol)

        # ── 포지션 보유 중 ──
        if position is not None:
            exit_signal = RiskManager.tick(position, current_price)
            if exit_signal.should_exit:
                await self._execute_kis_sell(symbol, position, exit_signal.reason, current_price)
                return

            # MA/RSI 전략 청산 신호 확인 (5분봉)
            _kis_pos_ohlcv: list[dict] | None = None
            if position.strategy in ("moving_average", "rsi"):
                _kis_pos_ohlcv = await kis.get_minute_ohlcv(symbol, interval_min=5, count=300)
                sell = self._check_strategy_sell(_kis_pos_ohlcv, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_kis_sell(symbol, position, "STRATEGY_SELL", current_price)
                    return

            # DCA 2/3차 추가매수 체크 (가격 하락 + RSI 과매도 동시 충족 시만)
            dca = self._dca_state.get(symbol)
            if dca:
                drop = (current_price - position.entry_price) / position.entry_price
                if _kis_pos_ohlcv is None:
                    try:
                        _kis_pos_ohlcv = await kis.get_minute_ohlcv(symbol, interval_min=5, count=50)
                    except Exception:
                        _kis_pos_ohlcv = []
                _rsi = self._quick_rsi(_kis_pos_ohlcv)
                _oversold = _rsi < 40  # RSI 40 미만 = 과매도 구간 (물타기 적합)
                if dca["stage"] == 1 and drop <= -0.015 and _oversold:
                    await self._execute_kis_dca_add(symbol, current_price, position, 2, 0.35)
                elif dca["stage"] == 2 and drop <= -0.030 and _oversold:
                    await self._execute_kis_dca_add(symbol, current_price, position, 3, 0.25)
            return

        # ── 포지션 없음: 매수 신호 탐색 (5분봉 전략 + 15분봉 ML 게이트) ──
        ohlcv_5min = await kis.get_minute_ohlcv(symbol, interval_min=5, count=300)
        result, strategy_name = self._run_kis_strategies(ohlcv_5min, current_price)
        if result.is_buy:
            try:
                ohlcv_ml = await kis.get_minute_ohlcv(symbol, interval_min=15, count=200)
            except Exception:
                ohlcv_ml = ohlcv_5min
            ml_ok = await self._check_ml_gate(symbol, ohlcv_ml, market="stock")
            if ml_ok:
                sim_log.push(symbol, f"{strategy_name}+5분봉ML — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_kis_buy(symbol, current_price, result, strategy_name)
            else:
                sim_log.push(symbol, f"{strategy_name} 신호 있으나 ML 차단 @ {current_price:,.0f}원", "INFO")
        else:
            sim_log.push(symbol, f"분석 완료 — 신호없음 @ {current_price:,.0f}원", "INFO")

    def _run_kis_strategies(
        self,
        ohlcv: list,
        current_price: float,
    ) -> tuple[StrategyResult, str]:
        """전략 우선순위: 변동성 돌파 → MA 크로스 → RSI."""
        result = vb_strategy.analyze(ohlcv, current_price, k=settings.volatility_k)
        if result.is_buy:
            return result, "volatility_breakout"

        result = ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if result.is_buy:
            return result, "moving_average"

        result = rsi_strategy.analyze(
            ohlcv,
            period=settings.rsi_period,
            oversold=settings.rsi_oversold,
            overbought=settings.rsi_overbought,
        )
        return result, "rsi"

    async def _check_ml_gate(self, symbol: str, ohlcv_ml: list, market: str = "coin") -> bool:
        """앙상블 ML 게이트 (코인=60분봉, 주식=15분봉).

        모든 학습된 에이전트가 최근 성과 기반 가중 투표.
        지금 잘 맞히는 전략이 자동으로 발언권 커짐 → 장세 변화 자동 적응.
        학습된 에이전트 없으면 초기 상태이므로 통과.
        """
        from backend.ml.agents import AGENTS, predict_ensemble

        trained = [a for a in AGENTS.values() if a.market == market and a._model is not None]
        if not trained:
            logger.info("[ML Gate][%s] 학습된 에이전트 없음 → 통과", symbol)
            return True

        try:
            btc_ohlcv  = self._btc_ohlcv_cache if market == "coin" else None
            oi_hist    = self._btc_oi_hist      if market == "coin" else None
            taker_hist = self._btc_taker_hist   if market == "coin" else None
            ls_hist    = self._btc_ls_hist      if market == "coin" else None
            obi_snaps  = self._obi_snaps        if market == "coin" else None
            signal, prob = await predict_ensemble(
                ohlcv_ml, ticker=symbol, market=market,
                btc_ohlcv=btc_ohlcv,
                oi_hist=oi_hist, taker_hist=taker_hist, ls_hist=ls_hist,
                obi_snaps=obi_snaps,
            )
            approved = signal == "buy"
            n = len(trained)
            logger.info(
                "[ML Gate][%s] 앙상블(%d개) %s %.1f%% → %s",
                symbol, n, signal, prob * 100,
                "통과" if approved else "차단",
            )
            return approved
        except Exception:
            logger.warning("[ML Gate][%s] 앙상블 실패, 통과 처리", symbol)
            return True

    def _check_strategy_sell(
        self,
        ohlcv: list,
        strategy: str,
    ) -> StrategyResult | None:
        """보유 포지션의 전략 기반 청산 신호 확인 (MA 데드크로스 / RSI 과매수)."""
        if strategy == "moving_average":
            return ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if strategy == "rsi":
            return rsi_strategy.analyze(
                ohlcv,
                period=settings.rsi_period,
                oversold=settings.rsi_oversold,
                overbought=settings.rsi_overbought,
            )
        return None

    @staticmethod
    def _quick_rsi(ohlcv_list: list[dict], period: int = 9) -> float:
        """OHLCV에서 최신 RSI 빠르게 계산 (DCA 과매도 조건 체크용).

        Wilder 평활 EMA 방식. 데이터 부족 시 중립값 50 반환.
        """
        closes = [float(c["close"]) for c in reversed(ohlcv_list)]
        if len(closes) < period + 2:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            return 100.0
        return round(100 - (100 / (1 + avg_g / avg_l)), 2)

    async def _execute_kis_buy(
        self,
        symbol: str,
        current_price: float,
        signal: StrategyResult,
        strategy_name: str,
    ) -> None:
        try:
            # TOCTOU 방지: 포지션 체크 + pending 마킹을 동일 락 안에서 원자적으로
            async with self._lock:
                if symbol in self._positions or symbol in self._pending_buys:
                    logger.warning("[KIS][%s] 중복매수 차단 (포지션 이미 존재 또는 주문 진행 중)", symbol)
                    return
                self._pending_buys.add(symbol)

            try:
                balance = await kis.get_balance()
                cash = float(balance["cash"])
                total_qty = int(RiskManager.calculate_qty(cash, current_price))
                qty = max(1, int(total_qty * 0.40))  # DCA 1차: 총 수량의 40%

                ok, reason = RiskManager.validate_order(cash, current_price, float(qty))
                if not ok:
                    logger.warning("[KIS][%s] 매수 불가: %s", symbol, reason)
                    return

                await kis.place_buy_order(symbol, qty)

                # ATR 기반 동적 손익 (에이전트 검증 완료 후 자동 전환)
                sl_rate = tp_rate = None
                if self._use_atr_risk:
                    from backend.ml.agents import AGENTS
                    atr_vals = [a._last_atr_pct for a in AGENTS.values() if a.market == "stock" and a._last_atr_pct > 0.001]
                    if atr_vals:
                        atr = sum(atr_vals) / len(atr_vals)
                        sl_rate = -(atr * 1.5)
                        tp_rate = atr * 2.0

                position = RiskManager.open_position(
                    symbol=symbol,
                    entry_price=current_price,
                    qty=float(qty),
                    strategy=strategy_name,
                    is_crypto=False,
                    stop_loss_rate=sl_rate,
                    take_profit_rate=tp_rate,
                )
                async with self._lock:
                    self._positions[symbol] = position
                    self._dca_state[symbol] = {"stage": 1, "total_qty": total_qty, "bought_qty": qty}
                await upsert_position(symbol, "KIS", position)
                await insert_trade(symbol, "KIS", "BUY", float(qty), current_price, strategy_name, signal.reason)
                sim_log.push(symbol, f"매수 체결 {qty}주 @ {current_price:,.0f}원 | 손절 {position.stop_loss_price:,.0f}", "BUY")

                logger.info(
                    "[KIS 매수] %s %d주 @ %,.0f원 | 전략: %s | %s",
                    symbol, qty, current_price, strategy_name, signal.reason,
                )
                await telegram.notify_buy(symbol, float(qty), current_price, strategy_name, signal.reason)
            finally:
                self._pending_buys.discard(symbol)
        except Exception:
            logger.exception("[KIS][%s] 매수 실행 중 예외", symbol)

    async def _execute_kis_sell(
        self,
        symbol: str,
        position: Position,
        reason: str,
        current_price: float,
    ) -> None:
        _saved_dca: dict | None = None
        _order_placed = False
        try:
            # 이중 매도 방지: 락 아래서 포지션 + DCA 상태 원자적 제거
            async with self._lock:
                if symbol not in self._positions:
                    logger.warning("[KIS][%s] 매도 중복 차단 (이미 청산됨)", symbol)
                    return
                _saved_dca = self._dca_state.get(symbol)
                self._positions.pop(symbol)
                self._dca_state.pop(symbol, None)
                self._partial_tp_state.pop(symbol, None)

            qty = max(1, round(position.qty))
            await kis.place_sell_order(symbol, qty)
            _order_placed = True  # 이 시점부터 복구 불가

            profit_rate = (current_price - position.entry_price) / position.entry_price
            logger.info(
                "[KIS 매도] %s %d주 @ %,.0f원 | 수익률 %.2f%% | 사유: %s",
                symbol, qty, current_price, profit_rate * 100, reason,
            )
            _update_ticker_stats(symbol, profit_rate)  # 실제 손절도 클러스터 스탑에 반영
            await insert_trade(symbol, "KIS", "SELL", float(qty), current_price, position.strategy, reason, profit_rate)
            sim_log.push(symbol, f"매도 체결 {qty}주 @ {current_price:,.0f}원 | 수익률 {profit_rate*100:+.2f}% | {reason}", "SELL")
            await telegram.notify_sell(symbol, float(qty), current_price, profit_rate, reason)
            try:
                await delete_position(symbol)
            except Exception:
                logger.exception("[KIS][%s] DB position 삭제 실패 — 다음 재시작 시 좀비 포지션 정리됨", symbol)
        except Exception:
            if not _order_placed:
                # 주문 자체가 실패 → 포지션 복구
                async with self._lock:
                    if symbol not in self._positions:
                        self._positions[symbol] = position
                        if _saved_dca is not None:
                            self._dca_state[symbol] = _saved_dca
            else:
                # 주문 체결 후 DB 기록 실패 → 복구 안 함 (재매도 방지)
                logger.error("[KIS][%s] 매도 체결됐으나 DB 기록 실패 — 수동 확인 필요", symbol)
            logger.exception("[KIS][%s] 매도 실행 중 예외", symbol)

    async def _execute_kis_dca_add(
        self,
        symbol: str,
        current_price: float,
        position: Position,
        next_stage: int,
        qty_ratio: float,
    ) -> None:
        """KIS DCA 추가매수 (2차/3차)."""
        try:
            dca = self._dca_state.get(symbol)
            if not dca:
                return
            add_qty = max(1, int(dca["total_qty"] * qty_ratio))
            if add_qty < 1:
                return

            await kis.place_buy_order(symbol, add_qty)
            # 주문 체결 직후 — 락 안에서 원자적으로 포지션 갱신
            async with self._lock:
                if symbol not in self._positions:
                    return
                dca["stage"] = next_stage
                dca["bought_qty"] = int(dca.get("bought_qty", 0)) + add_qty
                total_cost = position.entry_price * position.qty + current_price * add_qty
                new_qty = position.qty + add_qty
                new_avg = total_cost / new_qty
                position.entry_price = new_avg
                position.qty = new_qty
                position.stop_loss_price = new_avg * (1 + settings.stop_loss_rate)
                position.take_profit_price = new_avg * (1 + settings.take_profit_rate)
                position.highest_price = new_avg
                position.trailing_stop_price = new_avg * (1 - settings.trailing_stop_rate)
                self._positions[symbol] = position
            await upsert_position(symbol, "KIS", position)
            await insert_trade(symbol, "KIS", "BUY", float(add_qty), current_price, position.strategy, f"DCA {next_stage}차")
            sim_log.push(symbol, f"DCA {next_stage}차 {add_qty}주 추가매수 @ {current_price:,.0f}원 | 평균단가 {new_avg:,.0f}원", "BUY")
            logger.info("[KIS DCA %d차] %s %d주 @ %,.0f원 | 평균단가 %,.0f원", next_stage, symbol, add_qty, current_price, new_avg)
            await telegram.notify_buy(symbol, float(add_qty), current_price, position.strategy, f"DCA {next_stage}차 추가매수")
        except Exception:
            logger.exception("[KIS][%s] DCA %d차 추가매수 예외", symbol, next_stage)

    # ── KIS EOD 강제 매도 ────────────────────────────────────────

    async def _kis_eod_sell(self) -> None:
        """15:20 KST - 변동성 돌파 전략 포지션 강제 청산."""
        for symbol in list(self._kis_symbols):
            try:
                async with self._lock:
                    position = self._positions.get(symbol)

                # 변동성 돌파 전략으로 열린 포지션만 EOD 매도 대상
                if position is None or position.strategy != "volatility_breakout":
                    continue

                price_data = await kis.get_price(symbol)
                await self._execute_kis_sell(
                    symbol,
                    position,
                    "EOD_VOLATILITY_BREAKOUT",
                    float(price_data["current_price"]),
                )
            except Exception:
                logger.exception("[KIS][%s] EOD 매도 중 예외", symbol)

    # ── Upbit 틱 ─────────────────────────────────────────────────

    async def _upbit_tick(self) -> None:
        if not settings.upbit_access_key:
            return  # Upbit 키 미설정 시 전체 스킵
        if settings.is_paper:
            return  # PAPER 모드: 업비트 모의투자 API 없음 → 실주문 방지

        for ticker in list(self._upbit_tickers):
            try:
                await self._process_upbit_ticker(ticker)
            except Exception:
                logger.exception("[Upbit][%s] 틱 처리 중 예외", ticker)

    async def _process_upbit_ticker(self, ticker: str) -> None:
        price_data = await upbit.get_price(ticker)
        current_price = float(price_data["current_price"])

        async with self._lock:
            position = self._positions.get(ticker)

        # ── 포지션 보유 중 ──
        if position is not None:
            exit_signal = RiskManager.tick(position, current_price)
            if exit_signal.should_exit:
                await self._execute_upbit_sell(ticker, position, exit_signal.reason, current_price)
                return

            # ── 부분 청산 Scale-out: ATR×1.5 도달 시 40% 선익절 ──
            _pts = self._partial_tp_state.get(ticker)
            if (_pts and not _pts["done"] and current_price >= _pts["target"]):
                await self._execute_upbit_partial_sell(ticker, position, current_price)

            # OI 극단값 강제 청산 (롱 쏠림 + 고펀딩비 = 강제청산 폭발 위험)
            # OI 5봉 변화율 > 3% AND 펀딩비 > 0.003 (0.3%) → 롱 포지션 미리 청산
            if self._btc_oi_hist:
                try:
                    from backend.api.binance import get_funding_rate
                    _funding = await get_funding_rate("BTCUSDT")
                    _recent_oi = [float(d["sumOpenInterest"]) for d in self._btc_oi_hist[-6:]]
                    if len(_recent_oi) >= 2:
                        _oi_base = _recent_oi[0]
                        _oi_chg = (_recent_oi[-1] - _oi_base) / _oi_base if _oi_base > 0 else 0.0
                        if _oi_chg > 0.03 and _funding > 0.003:
                            sim_log.push(ticker, f"OI 급등({_oi_chg:.1%}) + 고펀딩비({_funding:.4f}) → 강제청산", "SELL")
                            await self._execute_upbit_sell(ticker, position, "OI_EXTREME_LONG_FLUSH", current_price)
                            return
                except Exception:
                    pass

            # MA/RSI 전략 기반 청산 (5분봉) — DCA보다 먼저 체크 (청산 시 DCA 스킵)
            _upbit_pos_ohlcv: list[dict] | None = None
            if position.strategy in ("moving_average", "rsi"):
                _upbit_pos_ohlcv = await upbit.get_ohlcv(ticker, interval="minutes/5", count=300)
                sell = self._check_strategy_sell(_upbit_pos_ohlcv, position.strategy)
                if sell is not None and sell.is_sell:
                    await self._execute_upbit_sell(ticker, position, "STRATEGY_SELL", current_price)
                    return

            # DCA 2/3차 추가 매수 체크 (가격 하락 + RSI 과매도 동시 충족 시만)
            dca = self._dca_state.get(ticker)
            if dca:
                drop = (current_price - position.entry_price) / position.entry_price
                if _upbit_pos_ohlcv is None:
                    try:
                        _upbit_pos_ohlcv = await upbit.get_ohlcv(ticker, interval="minutes/5", count=50)
                    except Exception:
                        _upbit_pos_ohlcv = []
                _rsi = self._quick_rsi(_upbit_pos_ohlcv)
                _oversold = _rsi < 40
                if dca["stage"] == 1 and drop <= -0.015 and _oversold:
                    async with self._lock:
                        if ticker not in self._positions:
                            return
                    await self._execute_upbit_dca_add(ticker, current_price, position, 2, 0.35)
                elif dca["stage"] == 2 and drop <= -0.030 and _oversold:
                    async with self._lock:
                        if ticker not in self._positions:
                            return
                    await self._execute_upbit_dca_add(ticker, current_price, position, 3, 0.25)
            return

        # ── 포지션 없음: 매수 신호 탐색 (5분봉) ──
        # 실매매에도 야간 고위험 시간대 차단 적용 (가상매매 _agent_execute와 동일 기준)
        if _is_coin_night_risk():
            return
        ohlcv_5min = await upbit.get_ohlcv(ticker, interval="minutes/5", count=300)
        result, strategy_name = self._run_upbit_strategies(ohlcv_5min, current_price)
        if result.is_buy:
            try:
                ohlcv_ml = await upbit.get_ohlcv(ticker, interval="minutes/60", count=200)
            except Exception:
                ohlcv_ml = ohlcv_5min
            ml_ok = await self._check_ml_gate(ticker, ohlcv_ml, market="coin")
            if ml_ok:
                sim_log.push(ticker, f"{strategy_name}+5분봉ML — {result.reason} @ {current_price:,.0f}원", "BUY")
                await self._execute_upbit_buy(ticker, current_price, result, strategy_name)
            else:
                sim_log.push(ticker, f"{strategy_name} 신호 있으나 ML 차단 @ {current_price:,.0f}원", "INFO")
        else:
            sim_log.push(ticker, f"분석 완료 — 신호없음 @ {current_price:,.0f}원", "INFO")

    def _run_upbit_strategies(
        self,
        ohlcv: list,
        current_price: float,
    ) -> tuple[StrategyResult, str]:
        """코인: MA 크로스 → RSI.

        변동성 돌파는 EOD 강제 매도가 필요한 전략이라 코인(24/7)에 부적합하므로 제외.
        """
        result = ma_strategy.analyze(ohlcv, ma_short=settings.ma_short, ma_long=settings.ma_long)
        if result.is_buy:
            return result, "moving_average"

        result = rsi_strategy.analyze(
            ohlcv,
            period=settings.rsi_period,
            oversold=settings.rsi_oversold,
            overbought=settings.rsi_overbought,
        )
        return result, "rsi"

    async def _execute_upbit_buy(
        self,
        ticker: str,
        current_price: float,
        signal: StrategyResult,
        strategy_name: str,
    ) -> None:
        try:
            # TOCTOU 방지: 포지션 체크 + pending 마킹을 동일 락 안에서 원자적으로
            async with self._lock:
                if ticker in self._positions or ticker in self._pending_buys:
                    logger.warning("[Upbit][%s] 중복매수 차단 (포지션 이미 존재 또는 주문 진행 중)", ticker)
                    return
                self._pending_buys.add(ticker)

            try:
                balance = await upbit.get_balance()
                krw = float(balance["krw"])
                total_budget = krw * settings.max_position_ratio
                amount_krw = total_budget * 0.40  # DCA 1차: 총 예산의 40%

                if amount_krw < 5_000:
                    logger.warning("[Upbit][%s] 매수 불가: 잔액 부족 (%.0f원)", ticker, krw)
                    return

                await upbit.place_buy_order(ticker, amount_krw)

                # qty는 추정값 (실제 체결 수량은 주문 체결 후 확정됨)
                qty = amount_krw / current_price

                # ATR 기반 동적 손익 (에이전트 검증 완료 후 자동 전환)
                sl_rate = tp_rate = None
                if self._use_atr_risk:
                    from backend.ml.agents import AGENTS
                    atr_vals = [a._last_atr_pct for a in AGENTS.values() if a.market == "coin" and a._last_atr_pct > 0.001]
                    if atr_vals:
                        atr = sum(atr_vals) / len(atr_vals)
                        sl_rate = -(atr * 1.5)
                        tp_rate = atr * 2.0

                position = RiskManager.open_position(
                    symbol=ticker,
                    entry_price=current_price,
                    qty=qty,
                    strategy=strategy_name,
                    is_crypto=True,
                    stop_loss_rate=sl_rate,
                    take_profit_rate=tp_rate,
                )
                async with self._lock:
                    self._positions[ticker] = position
                    self._dca_state[ticker] = {"stage": 1, "total_budget": total_budget}
                    # 부분 청산 1차 목표가 (ATR×1.5)
                    if sl_rate is not None and sl_rate != 0:
                        _atr_est = abs(sl_rate) / 1.5  # sl_rate = -(atr*1.5) 역산
                        self._partial_tp_state[ticker] = {
                            "target": current_price * (1 + _atr_est * 1.5),
                            "done": False,
                        }
                await upsert_position(ticker, "UPBIT", position)
                await insert_trade(ticker, "UPBIT", "BUY", qty, current_price, strategy_name, signal.reason)
                sim_log.push(ticker, f"매수 체결 {qty:.6f} @ {current_price:,.0f}원 | 손절 {position.stop_loss_price:,.0f}", "BUY")

                logger.info(
                    "[Upbit 매수] %s %.8f @ %.0f원 | 전략: %s | %s",
                    ticker, qty, current_price, strategy_name, signal.reason,
                )
                await telegram.notify_buy(ticker, qty, current_price, strategy_name, signal.reason, is_crypto=True)
            finally:
                self._pending_buys.discard(ticker)
        except Exception:
            logger.exception("[Upbit][%s] 매수 실행 중 예외", ticker)

    async def _execute_upbit_dca_add(
        self,
        ticker: str,
        current_price: float,
        position: Position,
        next_stage: int,
        budget_ratio: float,
    ) -> None:
        """DCA 추가 매수 (2차 35% / 3차 25%). 평균 단가 갱신 후 DB 업데이트."""
        try:
            dca = self._dca_state.get(ticker)
            if not dca:
                return
            add_krw = dca["total_budget"] * budget_ratio
            if add_krw < 5_000:
                return

            await upbit.place_buy_order(ticker, add_krw)
            add_qty = add_krw / current_price
            # 주문 체결 직후 — 락 안에서 원자적으로 포지션 갱신
            async with self._lock:
                if ticker not in self._positions:
                    return
                dca["stage"] = next_stage
                total_cost = position.entry_price * position.qty + current_price * add_qty
                new_qty    = position.qty + add_qty
                new_avg    = total_cost / new_qty
                position.entry_price        = new_avg
                position.qty                = new_qty
                position.stop_loss_price    = new_avg * (1 + settings.stop_loss_rate)
                position.take_profit_price  = new_avg * (1 + settings.take_profit_rate)
                position.highest_price       = new_avg
                position.trailing_stop_price = new_avg * (1 - settings.trailing_stop_rate)
                self._positions[ticker] = position
            await upsert_position(ticker, "UPBIT", position)
            await insert_trade(ticker, "UPBIT", "BUY", add_qty, current_price, position.strategy, f"DCA {next_stage}차")
            sim_log.push(ticker, f"DCA {next_stage}차 {add_qty:.6f} @ {current_price:,.0f}원 | 평균단가 {new_avg:,.0f}원", "BUY")
            await telegram.notify_buy(ticker, add_qty, current_price, position.strategy, f"DCA {next_stage}차 추가매수 | 평균단가 {new_avg:,.0f}원", is_crypto=True)
        except Exception:
            logger.exception("[Upbit][%s] DCA %d차 추가매수 실패", ticker, next_stage)

    async def _execute_upbit_partial_sell(
        self,
        ticker: str,
        position: Position,
        current_price: float,
    ) -> None:
        """Upbit 실제 포지션 부분 청산 (40%). ATR×1.5 도달 시 선익절, 나머지 홀딩."""
        try:
            # sell_qty를 락 아래서 결정 → await 사이 DCA 추가에 의한 qty 불일치 방지
            async with self._lock:
                if ticker not in self._positions:
                    return
                sell_qty = self._positions[ticker].qty * 0.40
            if sell_qty * current_price < 5_000:
                return
            await upbit.place_sell_order(ticker, sell_qty)
            profit_rate = (current_price - position.entry_price) / position.entry_price
            async with self._lock:
                if ticker not in self._positions:
                    return
                self._positions[ticker].qty -= sell_qty
                position = self._positions[ticker]
                _pts = self._partial_tp_state.get(ticker)
                if _pts:
                    _pts["done"] = True
            await upsert_position(ticker, "UPBIT", position)
            await insert_trade(ticker, "UPBIT", "SELL_PARTIAL", sell_qty, current_price,
                               position.strategy, f"부분청산40% +{profit_rate*100:.1f}%")
            sim_log.push(ticker,
                f"[부분청산40%] {sell_qty:.6f} @ {current_price:,.0f}원 +{profit_rate*100:.1f}%", "BUY")
            await telegram.notify_message(
                f"🔀 <b>부분청산(40%)</b> {ticker}\n"
                f"매도: {sell_qty:.6f} @ {current_price:,.0f}원\n"
                f"수익: +{profit_rate*100:.1f}% | 잔여 {position.qty:.6f} 홀딩 중"
            )
        except Exception:
            logger.exception("[Upbit][%s] 부분청산 실패", ticker)

    async def _execute_upbit_sell(
        self,
        ticker: str,
        position: Position,
        reason: str,
        current_price: float,
    ) -> None:
        _saved_dca_upbit: dict | None = None
        _order_placed = False
        try:
            # 이중 매도 방지: 락 아래서 포지션 + DCA 상태 원자적 제거
            async with self._lock:
                if ticker not in self._positions:
                    logger.warning("[Upbit][%s] 매도 중복 차단 (이미 청산됨)", ticker)
                    return
                _saved_dca_upbit = self._dca_state.get(ticker)
                self._positions.pop(ticker)
                self._dca_state.pop(ticker, None)
                self._partial_tp_state.pop(ticker, None)

            await upbit.place_sell_order(ticker, position.qty)
            _order_placed = True  # 이 시점부터 복구 불가

            profit_rate = (current_price - position.entry_price) / position.entry_price
            logger.info(
                "[Upbit 매도] %s %.8f @ %.0f원 | 수익률 %.2f%% | 사유: %s",
                ticker, position.qty, current_price, profit_rate * 100, reason,
            )
            _update_ticker_stats(ticker, profit_rate)  # 실제 손절도 클러스터 스탑에 반영
            await insert_trade(ticker, "UPBIT", "SELL", position.qty, current_price, position.strategy, reason, profit_rate)
            sim_log.push(ticker, f"매도 체결 {position.qty:.6f} @ {current_price:,.0f}원 | 수익률 {profit_rate*100:+.2f}% | {reason}", "SELL")
            await telegram.notify_sell(ticker, position.qty, current_price, profit_rate, reason, is_crypto=True)
            try:
                await delete_position(ticker)
            except Exception:
                logger.exception("[Upbit][%s] DB position 삭제 실패 — 다음 재시작 시 좀비 포지션 정리됨", ticker)
        except Exception:
            if not _order_placed:
                # 주문 자체가 실패 → 포지션 복구
                async with self._lock:
                    if ticker not in self._positions:
                        self._positions[ticker] = position
                        if _saved_dca_upbit is not None:
                            self._dca_state[ticker] = _saved_dca_upbit
            else:
                # 주문 체결 후 DB 기록 실패 → 복구 안 함 (재매도 방지)
                logger.error("[Upbit][%s] 매도 체결됐으나 DB 기록 실패 — 수동 확인 필요", ticker)
            logger.exception("[Upbit][%s] 매도 실행 중 예외", ticker)


    async def _daily_report(self) -> None:
        """매일 08:00 — 주요 종목 백테스트 리포트 텔레그램 발송."""
        try:
            await send_daily_report()
        except Exception:
            logger.exception("[리포트] 일일 리포트 실행 중 예외")

    async def _agent_tick(self) -> None:
        """30분마다 — 20개 에이전트 가상 매매 실행 (코인 60분봉, 주식 15분봉).

        코인: 24/7 — 업비트 거래대금 상위 50개
        주식: 평일 장중(09:00~15:30)에 추가 — KIS 거래량 상위 50개
        """
        from backend.api import upbit as _upbit
        from backend.api import kis as _kis
        from backend.ml.agents import AGENTS
        from backend.db.database import connect_db
        import aiosqlite

        now = datetime.now(KST)
        is_market_open = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 0))
            and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
        )

        # 코인 티커 (24/7)
        try:
            coin_tickers = await _upbit.get_top_tickers(50)
        except Exception:
            coin_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]

        # 주식 종목 (장중에만 추가)
        _STOCK_FALLBACK = ["005930", "000660", "035420", "035720", "000270", "207940", "373220", "068270"]
        stock_symbols: list[str] = []
        if is_market_open:
            try:
                stock_symbols = await _kis.get_volume_rank(50)
            except Exception:
                stock_symbols = list(self._kis_symbols)
            # 거래량 순위 조회 실패(빈 리스트) 시 대표 종목 폴백
            if not stock_symbols:
                stock_symbols = list(self._kis_symbols) or _STOCK_FALLBACK
                logger.warning("[AgentTick] 거래량 순위 빈 결과 → 폴백 %d개 사용", len(stock_symbols))
            # inquire-price 지원 안 되는 ELW/구조화상품 제거
            stock_symbols = [s for s in stock_symbols if s not in _STOCK_PRICE_BLACKLIST]

        # ── 코인 인터벌별 OHLCV + 현재가 병렬 프리패치 ────────────
        coin_intervals = list({a.interval_str for a in AGENTS.values() if a.market == "coin"})

        # 보유 포지션 코인이 상위 50개에서 빠져도 반드시 청산 체크
        open_coin_pos = {
            t for a in AGENTS.values() if a.market == "coin"
            for t in a._positions.keys()
        }
        extra_tickers = sorted(open_coin_pos - set(coin_tickers))
        fetch_coin_tickers = coin_tickers + extra_tickers
        if extra_tickers:
            logger.info("[AgentTick] 상위50 이탈 포지션 %d개 추가 조회: %s", len(extra_tickers), extra_tickers)

        async def _fetch_coin_ohlcv(ticker: str, interval: str) -> tuple[str, str, list]:
            async with _OHLCV_SEM:
                try:
                    data = await _upbit.get_ohlcv(ticker, interval=interval, count=200)
                except Exception:
                    data = []
            return ticker, interval, data

        async def _fetch_coin_price(ticker: str) -> tuple[str, float]:
            try:
                pd_ = await _upbit.get_price(ticker)
                return ticker, float(pd_["current_price"])
            except Exception:
                return ticker, 0.0

        # 배치 OHLCV 요청 — 10개씩 나눠서 배치 간 0.2초 딜레이 (429 방지)
        _all_ohlcv_tasks = [
            (t, iv) for t in fetch_coin_tickers for iv in coin_intervals
        ]
        ohlcv_results: list = []
        _BATCH = 10
        for _bi in range(0, len(_all_ohlcv_tasks), _BATCH):
            _chunk = _all_ohlcv_tasks[_bi:_bi + _BATCH]
            _chunk_results = await asyncio.gather(*[_fetch_coin_ohlcv(t, iv) for t, iv in _chunk])
            ohlcv_results.extend(_chunk_results)
            if _bi + _BATCH < len(_all_ohlcv_tasks):
                await asyncio.sleep(0.2)
        ohlcv_cache: dict[str, list] = {
            f"{t}:{iv}": data for t, iv, data in ohlcv_results
        }

        # 배치 현재가 (상위50 + 포지션 보유 코인 — 1회 요청으로 429 방지)
        try:
            coin_prices = await _upbit.get_prices_batch(fetch_coin_tickers)
        except Exception:
            coin_prices = {}

        # BTC + 주요 알트코인 선물 OI/Taker/L/S 히스토리 (코인별 개별 수급 신호)
        # BTC만 사용하면 ETH·XRP 등 개별 코인 수급 반영 불가 → 코인별 Binance 심볼 매핑
        from backend.api.binance import get_open_interest_hist, get_taker_ratio_hist, get_ls_ratio_hist
        # 업비트 KRW 티커 → 바이낸스 퍼페추얼 심볼 매핑 (시총 상위 & 업비트 거래량 상위)
        _UPBIT_TO_BINANCE: dict[str, str] = {
            "KRW-BTC": "BTCUSDT", "KRW-ETH": "ETHUSDT", "KRW-XRP": "XRPUSDT",
            "KRW-SOL": "SOLUSDT", "KRW-ADA": "ADAUSDT", "KRW-AVAX": "AVAXUSDT",
            "KRW-DOT": "DOTUSDT", "KRW-ATOM": "ATOMUSDT", "KRW-TRX": "TRXUSDT",
            "KRW-UNI": "UNIUSDT", "KRW-LTC": "LTCUSDT", "KRW-ETC": "ETCUSDT",
            "KRW-WAVES": "WAVEUSDT", "KRW-AAVE": "AAVEUSDT", "KRW-ARB": "ARBUSDT",
        }
        # 현재 모니터링 티커 중 Binance 매핑 있는 것만 추가 수집 (BTC 포함, API 부하 최소화)
        _binance_targets = list(dict.fromkeys(
            _UPBIT_TO_BINANCE[t] for t in coin_tickers if t in _UPBIT_TO_BINANCE
        ))[:6]  # BTC 포함 최대 6개 (Binance API 부하 제한)
        if "BTCUSDT" not in _binance_targets:
            _binance_targets.insert(0, "BTCUSDT")

        # 각 심볼별 OI/Taker/LS 병렬 수집
        _coin_flow_cache: dict[str, dict[str, list]] = {}
        try:
            _fetch_tasks = []
            for _sym in _binance_targets:
                _fetch_tasks.extend([
                    get_open_interest_hist(_sym, period="1h", limit=200),
                    get_taker_ratio_hist(_sym, period="1h", limit=200),
                    get_ls_ratio_hist(_sym, period="1h", limit=200),
                ])
            _results = await asyncio.gather(*_fetch_tasks, return_exceptions=True)
            for i, _sym in enumerate(_binance_targets):
                _oi   = _results[i * 3]     if not isinstance(_results[i * 3], Exception) else []
                _tk   = _results[i * 3 + 1] if not isinstance(_results[i * 3 + 1], Exception) else []
                _ls   = _results[i * 3 + 2] if not isinstance(_results[i * 3 + 2], Exception) else []
                _coin_flow_cache[_sym] = {"oi": _oi, "taker": _tk, "ls": _ls}
        except Exception:
            pass

        # BTC 기본값 (폴백용) — 이전 캐시 사용
        _btc_flow = _coin_flow_cache.get("BTCUSDT", {})
        btc_oi_hist    = _btc_flow.get("oi")    or self._btc_oi_hist
        btc_taker_hist = _btc_flow.get("taker") or self._btc_taker_hist
        btc_ls_hist    = _btc_flow.get("ls")    or self._btc_ls_hist
        if _btc_flow:
            self._btc_oi_hist    = btc_oi_hist
            self._btc_taker_hist = btc_taker_hist
            self._btc_ls_hist    = btc_ls_hist

        # 주식 OHLCV + 현재가도 병렬 프리패치
        stock_intervals = list({a.interval_min for a in AGENTS.values() if a.market == "stock"})
        stock_ohlcv_cache: dict[str, list] = {}
        stock_prices: dict[str, float] = {}

        # KOSPI 지수 15분봉 (주식 에이전트 상대강도 피처용 — 15분봉 에이전트와 타임프레임 맞춤)
        kospi_ohlcv: list[dict] = []
        if is_market_open and stock_symbols:
            try:
                kospi_ohlcv = await _kis.get_minute_ohlcv("0001", 15, count=200)
            except Exception:
                kospi_ohlcv = []

        # KOSPI MIM: 09:30~10:30 구간에 개장 방향 계산 (날짜 기준 1회)
        _today_str = datetime.now(KST).strftime("%Y-%m-%d")
        if (is_market_open and kospi_ohlcv
                and _today_str != self._morning_direction_date
                and 9 <= datetime.now(KST).hour < 11):
            try:
                _sorted_k = sorted(kospi_ohlcv, key=lambda x: x["date"])
                if len(_sorted_k) >= 6:  # 최소 30분(6봉) 데이터 필요
                    _first_open = float(_sorted_k[0]["open"])
                    _latest_close = float(_sorted_k[-1]["close"])
                    if _latest_close > _first_open * 1.001:
                        self._morning_direction = 1
                        self._morning_direction_date = _today_str
                        logger.info("[KOSPI MIM] 개장 상승(%+.2f%%) → 당일 BUY 허용",
                                    (_latest_close / _first_open - 1) * 100)
                    elif _latest_close < _first_open * 0.999:
                        self._morning_direction = -1
                        self._morning_direction_date = _today_str
                        logger.info("[KOSPI MIM] 개장 하락(%+.2f%%) → 당일 BUY 억제",
                                    (_latest_close / _first_open - 1) * 100)
            except Exception:
                pass

        if is_market_open and stock_symbols:
            async def _fetch_stock_ohlcv(sym: str, iv_min: int) -> tuple[str, int, list]:
                async with _OHLCV_SEM:
                    try:
                        data = await _kis.get_minute_ohlcv(sym, iv_min, count=200)
                    except Exception:
                        data = []
                return sym, iv_min, data

            async def _fetch_stock_price(sym: str) -> tuple[str, float]:
                try:
                    pd_ = await _kis.get_price(sym)
                    return sym, float(pd_["current_price"])
                except Exception:
                    return sym, 0.0

            s_ohlcv_tasks = [
                _fetch_stock_ohlcv(s, iv) for s in stock_symbols for iv in stock_intervals
            ]
            s_results = await asyncio.gather(*s_ohlcv_tasks)
            stock_ohlcv_cache = {f"{s}:{iv}": data for s, iv, data in s_results}

            sp_results = await asyncio.gather(*[_fetch_stock_price(s) for s in stock_symbols])
            stock_prices = {s: p for s, p in sp_results if p > 0}

        # BTC OHLCV 캐시 갱신 (ML 게이트 btc_corr_20 피처용 — 60분봉)
        _fresh_btc = ohlcv_cache.get("KRW-BTC:minutes/60", [])
        if _fresh_btc:
            self._btc_ohlcv_cache = _fresh_btc

        # BTC 오더북 OBI 스냅샷 수집 (매 5분 — 60분봉 1개당 12개 집계)
        try:
            from backend.api.binance import get_orderbook_obi
            _obi = await get_orderbook_obi("BTCUSDT", levels=5)
            self._obi_snaps.append(_obi)
            if len(self._obi_snaps) > 12:
                self._obi_snaps = self._obi_snaps[-12:]
        except Exception:
            pass

        # 주식 학습 데이터 캐시 누적 갱신 (최대 500봉 rolling) — 틱마다 쌓아서 데이터 풍부하게
        if stock_ohlcv_cache:
            for _k, _new in stock_ohlcv_cache.items():
                _existing = self._stock_train_cache.get(_k, [])
                if not _existing:
                    self._stock_train_cache[_k] = list(_new)
                else:
                    _exist_dates = {c["date"] for c in _existing}
                    _extra = [c for c in _new if c["date"] not in _exist_dates]
                    _combined = _extra + _existing  # 신규 봉 앞에 추가
                    self._stock_train_cache[_k] = _combined[:500]
        if kospi_ohlcv:
            if not self._kospi_train_cache:
                self._kospi_train_cache = list(kospi_ohlcv)
            else:
                _exist_dates = {c["date"] for c in self._kospi_train_cache}
                _extra = [c for c in kospi_ohlcv if c["date"] not in _exist_dates]
                self._kospi_train_cache = (_extra + self._kospi_train_cache)[:500]

        # 주식 온더플라이 학습용 공유 캐시 — 10개 에이전트가 같은 데이터 재사용 (API 중복 방지)
        _otf_stock_ohlcv: dict[str, list[dict]] = {}   # key: symbol
        _otf_kospi_ohlcv: list[dict] = []

        async with connect_db() as db:
            for agent in AGENTS.values():
                try:
                    if agent.market == "coin":
                        # ── 코인 전담 (AI01~AI10): 24/7 ──────────────
                        agent.update_position_values(coin_prices)
                        # BTC OHLCV (비BTC 코인의 상관관계 피처용 — 60분봉)
                        btc_ohlcv_cache = ohlcv_cache.get(f"KRW-BTC:{agent.interval_str}", [])
                        for ticker in fetch_coin_tickers:
                            ohlcv = ohlcv_cache.get(f"{ticker}:{agent.interval_str}", [])
                            if not ohlcv:
                                continue
                            # BTC 자신은 BTC 상관관계 불필요
                            _btc_ref = btc_ohlcv_cache if ticker != "KRW-BTC" else None
                            if agent._model is None and not agent.load_model():
                                if agent.interval_min <= 5:
                                    train_count = 2000
                                elif agent.interval_min <= 15:
                                    train_count = 1000
                                else:
                                    train_count = 700
                                try:
                                    train_ohlcv = await _upbit.get_ohlcv(
                                        ticker, interval=agent.interval_str, count=train_count
                                    )
                                    # 학습용 BTC 데이터도 동일 기간으로 조회
                                    _btc_train = None
                                    if ticker != "KRW-BTC":
                                        try:
                                            _btc_train = await _upbit.get_ohlcv(
                                                "KRW-BTC", interval=agent.interval_str, count=train_count
                                            )
                                        except Exception:
                                            pass
                                except Exception:
                                    train_ohlcv = ohlcv
                                    _btc_train = _btc_ref
                                _oi_ref    = btc_oi_hist    if btc_oi_hist    else None
                                _taker_ref = btc_taker_hist if btc_taker_hist else None
                                _ls_init   = btc_ls_hist    if btc_ls_hist    else None
                                async with agent._train_lock:
                                    trained = await asyncio.to_thread(
                                        agent.train, train_ohlcv,
                                        agent._cached_funding_rates or None,
                                        _btc_train, None,
                                        _oi_ref, _taker_ref, None, _ls_init,
                                    )
                                if not trained:
                                    continue
                            price = coin_prices.get(ticker, 0.0)
                            if price <= 0:
                                continue
                            # 코인별 개별 Binance 수급 데이터 (없으면 BTC 폴백)
                            _binance_sym = _UPBIT_TO_BINANCE.get(ticker, "BTCUSDT")
                            _coin_flow   = _coin_flow_cache.get(_binance_sym, {})
                            _oi_ref    = (_coin_flow.get("oi")    or btc_oi_hist)    or None
                            _taker_ref = (_coin_flow.get("taker") or btc_taker_hist) or None
                            _ls_ref    = (_coin_flow.get("ls")    or btc_ls_hist)    or None
                            signal, prob = agent.predict(ohlcv, btc_ohlcv=_btc_ref, oi_hist=_oi_ref, taker_hist=_taker_ref, ls_hist=_ls_ref, ticker=ticker, obi_snaps=self._obi_snaps)
                            await self._agent_execute(db, agent, ticker, signal, prob, price)

                    else:
                        # ── 주식 전담 (AI11~AI20): 장중만 ────────────
                        if not is_market_open:
                            continue
                        agent.update_position_values(stock_prices)
                        # 모델 없으면 전체 종목 데이터로 즉석 multi-stock 학습 시도
                        # 학습 데이터는 에이전트 간 공유 (_otf_stock_ohlcv) — API 중복 호출 방지
                        if agent._model is None and not agent.load_model():
                            # 이번 틱 캐시에서 모든 종목 OHLCV 수집 (없으면 _stock_train_cache 폴백)
                            for _sym in stock_symbols:
                                if _sym not in _otf_stock_ohlcv:
                                    _otf_stock_ohlcv[_sym] = (
                                        stock_ohlcv_cache.get(f"{_sym}:{agent.interval_min}")
                                        or self._stock_train_cache.get(f"{_sym}:{agent.interval_min}", [])
                                    )
                            if not _otf_kospi_ohlcv:
                                _otf_kospi_ohlcv[:] = kospi_ohlcv or self._kospi_train_cache or []
                            _kospi_tr = _otf_kospi_ohlcv
                            # 유효 데이터 있는 종목만 선별
                            _ohlcv_by_sym = {s: v for s, v in _otf_stock_ohlcv.items() if len(v) >= 50}
                            if _ohlcv_by_sym:
                                async with agent._train_lock:
                                    await asyncio.to_thread(
                                        agent.train_multi, _ohlcv_by_sym, _kospi_tr
                                    )
                        for symbol in stock_symbols:
                            ohlcv = stock_ohlcv_cache.get(f"{symbol}:{agent.interval_min}", [])
                            if not ohlcv:
                                continue
                            if agent._model is None:
                                continue  # 즉석 학습도 실패 → 이번 틱 스킵
                            price = stock_prices.get(symbol, 0.0)
                            if price <= 0:
                                continue
                            signal, prob = agent.predict(ohlcv, kospi_ohlcv=kospi_ohlcv or None, ticker=symbol)
                            await self._agent_execute(db, agent, symbol, signal, prob, price)

                    # agent_stats upsert (today_buy_count/dynamic_tp_mult 포함 → 재시작 후 복원 가능)
                    await db.execute(
                        """INSERT INTO agent_stats
                           (agent_id, interval_min, label_threshold, buy_threshold, feature_set,
                            total_trades, win_trades, win_rate, total_return, current_balance,
                            is_champion, is_active, today_buy_count, dynamic_tp_mult, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(agent_id) DO UPDATE SET
                             total_trades=excluded.total_trades, win_trades=excluded.win_trades,
                             win_rate=excluded.win_rate, total_return=excluded.total_return,
                             current_balance=excluded.current_balance, is_champion=excluded.is_champion,
                             buy_threshold=excluded.buy_threshold, is_active=excluded.is_active,
                             today_buy_count=excluded.today_buy_count,
                             dynamic_tp_mult=excluded.dynamic_tp_mult,
                             updated_at=excluded.updated_at""",
                        (
                            agent.agent_id, agent.interval_min, agent.label_threshold,
                            round(agent.buy_threshold, 4), agent.feature_set,
                            agent.total_trades, agent.win_trades,
                            round(agent.win_rate, 4), round(agent.total_return, 4),
                            round(agent._balance, 2), int(agent.is_champion),
                            int(agent.is_active), agent._today_buy_count,
                            round(agent._dynamic_tp_mult, 4),
                            now.strftime("%Y-%m-%dT%H:%M:%S"),
                        ),
                    )
                except Exception:
                    logger.exception("[Agent][%s] 틱 처리 실패", agent.agent_id)

            try:
                await db.commit()
            except Exception:
                logger.exception("[Agent] DB commit 실패 — 이번 틱 데이터 롤백됨")

        # ── 앙상블 그림자 에이전트 — 실매매 앙상블 신호 가상 추적 ──────────
        from backend.ml.agents import ENSEMBLE_AGENTS, predict_ensemble, refresh_champion_flags
        try:
            async with connect_db() as _edb:
                _edb.row_factory = aiosqlite.Row
                for _ea in ENSEMBLE_AGENTS.values():
                    _ea.update_position_values(coin_prices if _ea.market == "coin" else stock_prices)
                    _tickers = (coin_tickers if _ea.market == "coin" else stock_symbols)[:20]
                    for _eticker in _tickers:
                        _key = f"{_eticker}:{_ea.interval_str}" if _ea.market == "coin" else f"{_eticker}:{_ea.interval_min}"
                        _eohlcv = (ohlcv_cache if _ea.market == "coin" else stock_ohlcv_cache).get(_key, [])
                        if not _eohlcv:
                            continue
                        _eprice = (coin_prices if _ea.market == "coin" else stock_prices).get(_eticker, 0.0)
                        if _eprice <= 0:
                            continue
                        try:
                            _esig, _eprob = await predict_ensemble(
                                _eohlcv, ticker=_eticker, market=_ea.market,
                                btc_ohlcv=self._btc_ohlcv_cache if _ea.market == "coin" else None,
                                oi_hist=btc_oi_hist if _ea.market == "coin" else None,
                                taker_hist=btc_taker_hist if _ea.market == "coin" else None,
                                ls_hist=btc_ls_hist if _ea.market == "coin" else None,
                            )
                        except Exception:
                            continue
                        await self._agent_execute(_edb, _ea, _eticker, _esig, _eprob, _eprice)
                    await _edb.execute(
                        """INSERT INTO agent_stats
                           (agent_id, interval_min, label_threshold, buy_threshold, feature_set,
                            total_trades, win_trades, win_rate, total_return, current_balance,
                            is_champion, is_active, today_buy_count, dynamic_tp_mult, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,0,1,?,?,?)
                           ON CONFLICT(agent_id) DO UPDATE SET
                             total_trades=excluded.total_trades, win_trades=excluded.win_trades,
                             win_rate=excluded.win_rate, total_return=excluded.total_return,
                             current_balance=excluded.current_balance,
                             buy_threshold=excluded.buy_threshold,
                             today_buy_count=excluded.today_buy_count,
                             dynamic_tp_mult=excluded.dynamic_tp_mult,
                             updated_at=excluded.updated_at""",
                        (_ea.agent_id, _ea.interval_min, _ea.label_threshold,
                         round(_ea.buy_threshold, 4), _ea.feature_set,
                         _ea.total_trades, _ea.win_trades,
                         round(_ea.win_rate, 4), round(_ea.total_return, 4),
                         round(_ea._balance, 2), _ea._today_buy_count,
                         round(_ea._dynamic_tp_mult, 4),
                         now.strftime("%Y-%m-%dT%H:%M:%S")),
                    )
                await _edb.commit()
        except Exception:
            logger.exception("[EnsembleAgent] 앙상블 그림자 에이전트 틱 실패")

        refresh_champion_flags()

    async def _agent_execute(self, db, agent, symbol: str, signal: str, prob: float, price: float) -> None:
        """에이전트 단일 종목 가상 매수/매도 실행 + DB 기록."""
        # ── ATR 기반 동적 손익 ────────────────────────────────────────
        # ATR이 유효하면(>0.1%) 시장 변동성에 자동 적응, 없으면 고정값 폴백
        atr_pct = agent._last_atr_pct
        if atr_pct > 0.001:
            STOP_LOSS   = min(-(atr_pct * 1.2), -0.005)  # ATR×1.2, 최소 -0.5% 보장 (min: ATR 클수록 넓어짐)
            STOP_LOSS   = max(STOP_LOSS, -0.015)          # 상한 -1.5% (대손절 방지)
            tp_base     = max(atr_pct * agent._dynamic_tp_mult, 0.012)  # 동적 배수 (R비율 기반, 기본 3.0)
        else:
            STOP_LOSS   = -0.015
            tp_base     = 0.05

        # ── 보유 포지션 익절/손절 우선 체크 ─────────────────────────
        if symbol in agent._positions:
            pos = agent._positions[symbol]

            # 보유 시간 계산 (entered_at 에 tzinfo 있어도 safe하게 naive로 통일)
            try:
                now_kst  = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
                entered  = datetime.fromisoformat(pos.entered_at).replace(tzinfo=None)
                held_min = (now_kst - entered).total_seconds() / 60
            except Exception:
                held_min = 0

            # 가격 데이터 없이 24시간 이상 묶인 포지션 → 마지막 알려진 가격으로 강제 청산
            if price <= 0:
                if held_min >= 1440:  # 24시간
                    if pos.qty <= 0:
                        return  # qty 0이면 처리 불가 (ZeroDivisionError 방지)
                    last_price = agent._last_position_values.get(symbol, 0) / pos.qty
                    if last_price > 0:
                        signal = "sell"
                        price = last_price
                        sim_log.push(agent.agent_id, f"[타임아웃강제청산] {symbol} 24h 가격데이터 없음 @ {price:,.0f}원", "SELL")
                        # 아래 sell 로직으로 계속 진행 (return 없음)
                    else:
                        return  # 마지막 가격도 없으면 스킵
                else:
                    return  # 24시간 미만이면 스킵

            unreal = (price - pos.entry_price) / pos.entry_price

            # 강제청산: 코인 60분봉 lookahead 8봉=480분(8h), 주식 360분(6h)
            # 코인을 6h로 단축하면 lookahead 유효 구간 마지막 2h를 사용 못 함 → 480분으로 복원
            _force_close_min = 480 if agent.market == "coin" else 360
            if held_min >= _force_close_min:
                signal = "sell"
                sim_log.push(agent.agent_id, f"[{_force_close_min//60}h강제청산] {symbol} {held_min:.0f}분 보유 {unreal*100:.1f}% @ {price:,.0f}원", "SELL")

            # 최소 보유 10분: 진입 직후 신호 기반 조기 청산 완전 차단
            # 분석: 0~10분 청산 WR=38.2%(최악) → 노이즈 신호에 의한 손절이 주원인
            if held_min < 10:
                extreme_sl = STOP_LOSS * 1.5  # ATR×1.8 (급락 시에만 즉시 손절)
                if unreal >= tp_base:
                    signal = "sell"
                    sim_log.push(agent.agent_id, f"[즉시익절] {symbol} +{unreal*100:.1f}% ({held_min:.0f}분) @ {price:,.0f}원", "SELL")
                elif unreal <= extreme_sl:
                    signal = "sell"
                    sim_log.push(agent.agent_id, f"[급락손절] {symbol} {unreal*100:.1f}% (ATR×1.8={extreme_sl*100:.1f}%) @ {price:,.0f}원", "SELL")
                else:
                    signal = "hold"  # 그 외 신호 무시: 노이즈 청산 차단
            else:
                # 시간 경과 하향 ROI: 오래 묶일수록 익절 기준 완만 하향
                # R:R 항상 1.3 이상 유지 — 연구: MFE 분석 → 현재 너무 일찍 익절, 수익 놓침
                # 타임밴드 확장 (150min→240min): 코인 모멘텀 4~6시간 지속 연구 기반
                sl_abs = abs(STOP_LOSS)
                if held_min < 30:
                    take_profit = tp_base
                elif held_min < 120:
                    take_profit = max(tp_base * 0.90, sl_abs * 1.3)  # 30~120분: R:R 1.8:1 유지
                elif held_min < 240:
                    take_profit = max(tp_base * 0.75, sl_abs * 1.3)  # 120~240분: R:R 1.5:1 유지
                else:
                    # 240분(4h) 이상 보유 → 알파 소멸 구간, 수익·손실 무관 즉시 청산
                    # 수익: 알파 소멸 후 평균회귀 위험
                    # 손실: 방치 시 360min까지 추가 손실 가능 → 대칭 처리
                    if unreal > 0:
                        signal = "sell"
                        sim_log.push(agent.agent_id, f"[알파소멸익절] {symbol} +{unreal*100:.1f}% 240분 보유 강제청산 @ {price:,.0f}원", "SELL")
                    else:
                        signal = "sell"
                        sim_log.push(agent.agent_id, f"[알파소멸손절] {symbol} {unreal*100:.1f}% 240분 손실 추가방치 차단 @ {price:,.0f}원", "SELL")
                    take_profit = max(tp_base * 0.75, sl_abs * 1.3)

                # ── 부분 청산 Scale-out: ATR×2.0 도달 시 30% 선익절 (1.5→2.0, 40%→30%: 더 많은 상승 포착)
                _partial_target = agent._partial_tp_price.get(symbol, 0)
                if (_partial_target > 0
                        and price >= _partial_target
                        and symbol not in agent._partial_tp_done):
                    _ptrade = agent.virtual_partial_sell(symbol, price, 0.30)
                    if _ptrade:
                        await db.execute(
                            "INSERT INTO agent_trades (agent_id, ticker, action, price, qty,"
                            " entry_price, profit_rate, balance, buy_prob, buy_adx, buy_vol_ratio, traded_at)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (_ptrade.agent_id, _ptrade.ticker, _ptrade.action, _ptrade.price,
                             _ptrade.qty, _ptrade.entry_price, _ptrade.profit_rate,
                             _ptrade.balance, 0.0, 0.0, 0.0, _ptrade.traded_at),
                        )
                        # 잔여 qty를 agent_positions에 반영 (재시작 시 qty 과대계상 방지)
                        _rem_pos = agent._positions.get(symbol)
                        if _rem_pos:
                            await db.execute(
                                "INSERT OR REPLACE INTO agent_positions"
                                " (agent_id, ticker, entry_price, qty, entered_at)"
                                " VALUES (?,?,?,?,?)",
                                (agent.agent_id, symbol, _rem_pos.entry_price,
                                 _rem_pos.qty, _rem_pos.entered_at),
                            )
                        sim_log.push(agent.agent_id,
                            f"[부분청산30%] {symbol} @ {price:,.0f}원 +{(_ptrade.profit_rate or 0)*100:.1f}%",
                            "BUY")

                # 트레일링 스탑: TP의 50% 도달 시 최고점 추적으로 전환 (90%→50%: 수익 조기 보호 + 추가 상승 포착)
                # 연구: trailing stop이 fixed TP보다 MFE 포착률 높음 (40% 이상 개선)
                if unreal >= take_profit * 0.5:
                    agent._trailing_mode.add(symbol)
                if symbol in agent._trailing_mode:
                    _peak = agent._peak_price.get(symbol, price)
                    agent._peak_price[symbol] = max(_peak, price)
                    # Chandelier Exit: 업계 최검증 ATR×3 (수익팩터 1.61)
                    # sl_abs=ATR×1.2이므로 ×2.5=ATR×3.0 (Chandelier 동치)
                    # max(atr_pct×3, sl_abs×2.0, 1.5%): ATR 불유효 시 sl_abs×2 폴백, 최소 1.5% 보장
                    _chandelier_pct = max(atr_pct * 3.0, sl_abs * 2.0, 0.015)
                    _trail_sl = agent._peak_price[symbol] * (1 - _chandelier_pct)
                    if price <= _trail_sl and unreal > 0:
                        signal = "sell"
                        sim_log.push(agent.agent_id, f"[트레일링] {symbol} 최고점 대비 하락 {unreal*100:.1f}% @ {price:,.0f}원", "SELL")

                if signal != "sell":
                    # 트레일링 활성 시 하드 TP 스킵 → 트레일링이 더 많은 수익 포착 담당
                    if symbol not in agent._trailing_mode and unreal >= take_profit:
                        signal = "sell"
                        sim_log.push(agent.agent_id, f"[익절] {symbol} +{unreal*100:.1f}% ({held_min:.0f}분) ATR손익({tp_base*100:.1f}%/{STOP_LOSS*100:.1f}%) @ {price:,.0f}원", "SELL")
                    elif unreal <= STOP_LOSS:
                        signal = "sell"
                        sim_log.push(agent.agent_id, f"[손절] {symbol} {unreal*100:.1f}% (ATR기준 {STOP_LOSS*100:.1f}%) @ {price:,.0f}원", "SELL")

                # ── BTC 도미넌스 급등 시 알트코인 조기 청산 (시장 구조 모니터링) ───
                # 연구: BTC 도미넌스 > 60% + 상승 추세 = 알트코인 자금 BTC로 이탈
                if signal != "sell" and agent.market == "coin" and unreal > 0 and held_min > 20:
                    try:
                        _mkt = _scheduler_mkt_cache.get("ctx")
                        if _mkt and _mkt.get("btc_dominance", 58.0) > 60.0 and not symbol.startswith("KRW-BTC"):
                            signal = "sell"
                            sim_log.push(agent.agent_id, f"[BTC도미넌스] {symbol} BTC지배율>{_mkt['btc_dominance']:.1f}% 알트이탈→익절", "SELL")
                    except Exception:
                        pass

                # ── 뉴스 감성 기반 홀딩/청산 판단 ───────────────────────────────
                # 매수 후 보유 중인 종목의 뉴스를 계속 확인:
                #   - 뉴스 악화(score < -0.5) + 수익 있음 → 조기 익절
                #   - 뉴스 매우 나쁨(score < -0.7) → 손실 중에도 청산
                #   - 뉴스 강한 호재(score > 0.6) → TP 기준 10% 완화 (더 오를 것으로 판단)
                if signal != "sell" and held_min >= 15:
                    try:
                        from backend.ml.news import get_cached_score, get_sentiment_score
                        _ns = get_cached_score(symbol)
                        if _ns is None:
                            # 캐시 없으면 실시간 뉴스 분석 (보유 종목 우선 갱신)
                            try:
                                _ns = await get_sentiment_score(symbol)
                            except Exception:
                                _ns = None
                        if _ns is not None:
                            if _ns < -0.5 and unreal > 0:
                                # 뉴스 악화 + 수익 있음 → 지금 파는 게 낫다
                                signal = "sell"
                                sim_log.push(agent.agent_id,
                                    f"[뉴스악화익절] {symbol} 뉴스점수={_ns:.2f} 수익{unreal*100:.1f}% → 청산", "SELL")
                            elif _ns < -0.7:
                                # 뉴스가 매우 나쁨 → 손실이어도 추가 하락 전에 청산
                                signal = "sell"
                                sim_log.push(agent.agent_id,
                                    f"[뉴스위험청산] {symbol} 뉴스점수={_ns:.2f}(극악) {unreal*100:.1f}% → 강제청산", "SELL")
                            elif _ns > 0.6 and symbol not in agent._trailing_mode:
                                # 뉴스 강한 호재 → TP를 10% 올려서 더 오를 때까지 홀딩
                                take_profit = take_profit * 1.10
                                sim_log.push(agent.agent_id,
                                    f"[뉴스호재홀딩] {symbol} 뉴스점수={_ns:.2f} → TP+10% {take_profit*100:.1f}%까지 홀딩", "INFO")
                    except Exception:
                        pass

                # ── 종합 지속성 점수: 모든 기술 지표 통합 판단 ──────────────────
                # RSI·MACD·ADX·거래량·MTF·HA·BB·OI·Taker·펀딩비·BTC상관 종합
                # score < -0.4 + 수익 있음 → 조기 익절
                # score < -0.6            → 손실이어도 추가 하락 차단 청산
                # score >  0.5            → TP를 15% 완화(홀딩 연장)
                if signal != "sell" and held_min >= 20:
                    try:
                        _ohlcv_for_cont = getattr(agent, "_last_ohlcv_cache", {}).get(symbol)
                        if _ohlcv_for_cont and len(_ohlcv_for_cont) >= 30:
                            _btc_for_cont = (
                                self._btc_ohlcv_cache if agent.market == "coin" else None
                            )
                            _cont_score, _cont_reasons = agent.continuation_score(
                                ohlcv_list=_ohlcv_for_cont,
                                btc_ohlcv=_btc_for_cont,
                                ticker=symbol,
                            )
                            _reason_str = " | ".join(_cont_reasons[:4]) if _cont_reasons else ""
                            if _cont_score < -0.6:
                                signal = "sell"
                                sim_log.push(agent.agent_id,
                                    f"[종합판단청산] {symbol} 지속성={_cont_score:.2f} {_reason_str} → 강제청산",
                                    "SELL")
                            elif _cont_score < -0.4 and unreal > 0:
                                signal = "sell"
                                sim_log.push(agent.agent_id,
                                    f"[종합판단익절] {symbol} 지속성={_cont_score:.2f} 수익{unreal*100:.1f}% {_reason_str} → 조기익절",
                                    "SELL")
                            elif _cont_score > 0.5 and symbol not in agent._trailing_mode:
                                take_profit = take_profit * 1.15
                                sim_log.push(agent.agent_id,
                                    f"[종합판단홀딩] {symbol} 지속성={_cont_score:.2f} {_reason_str} → TP+15% {take_profit*100:.1f}%",
                                    "INFO")
                    except Exception:
                        pass

        if signal == "buy":
            # 일일 손실 한도 서킷 브레이커 (에이전트별 독립, 3% 초과 시 당일 차단)
            if _is_daily_loss_exceeded(agent):
                sim_log.push(agent.agent_id, f"[서킷브레이커] {symbol} 오늘 손실 3% 초과 → 신규매수 차단", "WARN")
                return
            # 코인: 매크로 이벤트 발표 ±2시간은 변동성 폭발 위험 → 신규 매수 차단
            if agent.market == "coin" and _is_high_risk_window():
                sim_log.push(agent.agent_id, f"[이벤트회피] {symbol} 매수 차단 (매크로 발표 ±2시간)", "INFO")
                return
            # 코인: 야간 고위험 시간대 (KST 01:00~04:59) 신규 매수 차단
            # KST 00시(WR=57.5%) 해제, KST 04시(WR=34.4%) 추가, KST 16-17시 해제(WR=52%)
            if agent.market == "coin" and _is_coin_night_risk():
                sim_log.push(agent.agent_id, f"[시간차단] {symbol} 저승률 시간대(01~04/09/12시)", "INFO")
                return
            # 주식: 개장 첫 25분 노이즈 구간 신규 매수 차단
            if agent.market == "stock" and _is_stock_open_noise():
                sim_log.push(agent.agent_id, f"[개장차단] {symbol} 09:00~09:25 노이즈 구간", "INFO")
                return
            # 저승률 종목 블랙리스트 (정적 + 글로벌 동적)
            if _is_blacklisted(symbol, agent):
                sim_log.push(agent.agent_id, f"[블랙리스트] {symbol} 저승률 종목 차단", "INFO")
                return
            # 전체 에이전트 공유 쿨다운 (손절 60분, 크로스-에이전트 재진입 방지)
            if _is_global_cooldown(symbol):
                sim_log.push(agent.agent_id, f"[글로벌쿨다운] {symbol} 전체에이전트 차단 중", "INFO")
                return
            # 에이전트 개별 쿨다운 (30분)
            if _is_cooldown(symbol, agent):
                sim_log.push(agent.agent_id, f"[쿨다운] {symbol} 손절 후 30분 재매수 금지", "INFO")
                return
            # 클러스터 손절 감지: 30분 안에 5회+ 손절 → 시장 전체 위기 → 신규매수 전면 차단
            if _is_cluster_stop():
                sim_log.push(agent.agent_id, f"[클러스터차단] {symbol} 30분내 다중손절 → 전면차단", "WARN")
                return
            # 중기 하락추세 하드 게이트: OHLCV 캐시로 16시간 수익률 확인
            # 코인 전용 (주식은 장 시간 제한으로 OHLCV 부족)
            if agent.market == "coin":
                try:
                    _ohlcv_gate = getattr(agent, "_last_ohlcv_cache", {}).get(symbol) or []
                    if len(_ohlcv_gate) >= 100:
                        _c_new = float(_ohlcv_gate[-1]["close"])
                        _c_old = float(_ohlcv_gate[0]["close"])
                        _ret_lookback = (_c_new / _c_old - 1) if _c_old > 0 else 0
                        if _ret_lookback < -0.05:
                            sim_log.push(agent.agent_id,
                                f"[하락추세차단] {symbol} {_ret_lookback*100:.1f}% 중기하락 진입금지", "INFO")
                            return
                except Exception:
                    pass
            # 군집매수 방지: 동일 종목을 3개+ 에이전트가 이미 보유 시 차단
            # (BTC 하락 시 동일 종목 전 에이전트 동시 손절 방지)
            # 앙상블 에이전트는 제외 — 이미 다수결 합의(40%+ buy_votes)로 필터링됨
            from backend.ml.agents import AGENTS as _ALL_AGENTS, ENSEMBLE_AGENTS as _EA
            if agent.agent_id not in _EA:
                _holders = sum(1 for a in _ALL_AGENTS.values() if a.market == agent.market and symbol in a._positions)
                if _holders >= 3:
                    sim_log.push(agent.agent_id, f"[군집매수차단] {symbol} {_holders}개 에이전트 보유 중", "INFO")
                    return
            # 3연속 손실 → 매수 차단 후 재학습 예약 (다음 주기에 자동 처리)
            if agent.needs_retrain:
                sim_log.push(agent.agent_id, f"[재학습대기] {symbol} 3연속손실 — 매수 보류", "INFO")
                return
            # 동적 블랙리스트: WR<35% + 20건 이상인 종목 진입 차단 (반복 손실 종목 자동 제외)
            if symbol in agent._ticker_blacklist:
                return
            # 일일 BUY 제한: WR 기반 자동 조정 (30건 미만 시 기본값 고정)
            # EV음수(WR<48%) 구간에서는 거래 수 억제, BEP 돌파 시 자동 확대
            _today_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
            if agent._today_date != _today_kst:
                agent._today_date = _today_kst
                agent._today_buy_count = 0
            _wr = agent.win_rate
            _min_trades_for_dynamic = 30  # 통계 신뢰 최소 건수
            if agent.total_trades < _min_trades_for_dynamic:
                # 학습 초기: 기본값으로 고정
                _daily_limit = 8 if agent.market == "coin" else 5
            elif agent.market == "coin":
                if _wr < 0.45:
                    _daily_limit = 6    # EV 음수, 최소화
                elif _wr < 0.48:
                    _daily_limit = 8    # BEP 직전, 보수적 유지
                elif _wr < 0.55:
                    _daily_limit = 12   # BEP 돌파, 확대
                else:
                    _daily_limit = 20   # 고승률, 적극 활용
            else:  # stock
                if _wr < 0.45:
                    _daily_limit = 4
                elif _wr < 0.48:
                    _daily_limit = 5
                elif _wr < 0.55:
                    _daily_limit = 8
                else:
                    _daily_limit = 12
            if agent._today_buy_count >= _daily_limit:
                return
            # ADX 필터: 코인 횡보장(ADX<15) 진입 차단 — 추세 없는 구간에서 WR 급락 방지
            # 20→15: VSA·FVG 반전 신호는 ADX 15~20 구간(횡보→추세 전환 직전)에서 유효
            if agent.market == "coin" and 0 < agent._last_adx_14 < 15:
                sim_log.push(agent.agent_id, f"[ADX차단] {symbol} ADX={agent._last_adx_14:.1f}<15 횡보장", "INFO")
                return
            # KOSPI MIM: 개장 30분 하락 방향 시 주식 당일 BUY 억제 (MDPI Finance 검증 전략)
            if agent.market == "stock" and self._morning_direction == -1:
                sim_log.push(agent.agent_id, f"[MIM차단] {symbol} KOSPI 개장 하락 → 당일 BUY 보류", "INFO")
                return
            # 외국인 수급: 외국인 500주+ 순매도 종목 BUY 차단 (주식 전용, 5분 TTL 캐시)
            if agent.market == "stock":
                import time as _time
                _cached_inv = self._investor_cache.get(symbol)
                _now_ts = _time.time()
                if _cached_inv and _now_ts - _cached_inv[1] < 300:
                    _inv = _cached_inv[0]
                else:
                    try:
                        from backend.api.kis import _kis
                        _inv = await _kis.get_investor_trend(symbol)
                        self._investor_cache[symbol] = (_inv, _now_ts)
                    except Exception:
                        _inv = {"foreign_net_buy": 0}
                if _inv.get("foreign_net_buy", 0) < -500:
                    sim_log.push(agent.agent_id, f"[외국인차단] {symbol} 외국인 {_inv['foreign_net_buy']:,}주 순매도", "INFO")
                    return
            # 뉴스 감성 필터: -0.5 이하 강한 부정 뉴스 → BUY 차단 (캐시 없으면 통과)
            try:
                from backend.ml.news import get_cached_score, get_sentiment_score
                _ns = get_cached_score(symbol)
                if _ns is None:
                    _ns = await get_sentiment_score(symbol)
                if _ns is not None and _ns < -0.5:
                    sim_log.push(agent.agent_id, f"[뉴스차단] {symbol} 감성점수={_ns:.2f} 부정뉴스 급락우려", "WARN")
                    return
            except Exception:
                pass
            # RVOL 필터: 거래량 1.5배 미만이면 포지션 50%로 줄임 (약한 신호 크기 축소)
            _vol_ratio = agent._last_vol_ratio
            _rvol_portion = 1.0 if _vol_ratio >= 1.5 else 0.5
            # DCA: 확률 강도에 비례한 진입 비율 (약한 신호는 50%, 강한 신호는 100%)
            _gap = prob - agent.buy_threshold
            _signal_portion = 1.0 if _gap >= 0.08 else (0.7 if _gap >= 0.04 else 0.5)
            _portion = min(_signal_portion, _rvol_portion)  # 두 필터 중 더 보수적인 값 적용
            trade = agent.virtual_buy(symbol, price, portion=_portion)
            if trade:
                await db.execute(
                    """INSERT INTO agent_trades
                       (agent_id, ticker, action, price, qty, entry_price, profit_rate, balance,
                        buy_prob, buy_adx, buy_vol_ratio, traded_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty,
                     trade.entry_price, trade.profit_rate, trade.balance,
                     round(prob, 4), round(agent._last_adx_14, 2), round(agent._last_vol_ratio, 3),
                     trade.traded_at),
                )
                await db.execute(
                    "INSERT OR REPLACE INTO agent_positions (agent_id, ticker, entry_price, qty, entered_at) VALUES (?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.price, trade.qty, trade.traded_at),
                )
                agent._today_buy_count += 1
                agent.record_buy_context(symbol)  # 매수 시점 신호 스냅샷 저장 (손실 분석용)
                sim_log.push(trade.agent_id, f"[가상매수] {symbol} @ {price:,.0f}원 (확률 {prob:.1%} / 진입{_portion*100:.0f}% / 일{agent._today_buy_count}건)", "BUY")
                # 부분 청산 1차 목표가 설정 (ATR×2.0), ATR 없으면 미설정
                if atr_pct > 0.001:
                    agent._partial_tp_price[symbol] = price * (1 + atr_pct * 2.0)

        elif signal == "sell" and symbol in agent._positions:
            # 손실 분석: 매도 전 보유 시간 계산 (virtual_sell 후 포지션 소멸)
            _sell_pos = agent._positions.get(symbol)
            _held_min = 0.0
            if _sell_pos and _sell_pos.entered_at:
                try:
                    from datetime import datetime as _dt_sell
                    _now_kst  = _dt_sell.now(KST)
                    _ent_dt   = _dt_sell.fromisoformat(_sell_pos.entered_at)
                    if _ent_dt.tzinfo is None:
                        _ent_dt = _ent_dt.replace(tzinfo=KST)
                    _held_min = (_now_kst - _ent_dt).total_seconds() / 60
                except Exception:
                    pass
            trade = agent.virtual_sell(symbol, price)
            if trade:
                pct = (trade.profit_rate or 0) * 100
                await db.execute(
                    "INSERT INTO agent_trades (agent_id, ticker, action, price, qty, entry_price,"
                    " profit_rate, balance, buy_prob, buy_adx, buy_vol_ratio, traded_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (trade.agent_id, trade.ticker, trade.action, trade.price, trade.qty,
                     trade.entry_price, trade.profit_rate, trade.balance,
                     0.0, 0.0, 0.0, trade.traded_at),
                )
                # 포지션 영속성: 청산 시 제거
                await db.execute(
                    "DELETE FROM agent_positions WHERE agent_id=? AND ticker=?",
                    (trade.agent_id, trade.ticker),
                )
                level = "BUY" if (trade.profit_rate or 0) > 0 else "SELL"
                sim_log.push(trade.agent_id, f"[가상매도] {symbol} @ {price:,.0f}원 | {pct:+.2f}%", level)
                # 손실 발생 시 소급 분석 → 패턴 학습
                if (trade.profit_rate or 0) < 0:
                    _analysis = agent.analyze_loss(symbol, trade.profit_rate, _held_min)
                    _cause_str = " | ".join(_analysis["causes"][:3]) if _analysis["causes"] else "원인 불명"
                    _repeat    = _analysis["pattern_repeat"]
                    sim_log.push(
                        agent.agent_id,
                        f"[손실분석] {symbol} {pct:+.1f}% 보유{_held_min:.0f}분 "
                        f"— {_cause_str}"
                        + (f" (패턴{_repeat}회 반복→패널티↑)" if _repeat >= 2 else ""),
                        "SELL",
                    )
                # 매도 통계 업데이트 (글로벌 티커 승률 추적)
                _update_ticker_stats(symbol, trade.profit_rate or 0)
                # 손절 시 쿨다운 등록: 에이전트 개별(30분) + 전체 공유(60분)
                if (trade.profit_rate or 0) < -0.003:
                    _set_cooldown(symbol, agent)
                    _set_global_cooldown(symbol, 60)
                # 5연속 손실 시 즉시 재학습 (백그라운드)
                if agent.needs_retrain:
                    agent.needs_retrain = False
                    agent._consecutive_losses = 0
                    sim_log.push(agent.agent_id, f"[즉시재학습] 5연속손실 감지 — 모델 갱신 시작", "INFO")
                    import asyncio as _asyncio

                    async def _safe_retrain(a=agent):
                        try:
                            await self._emergency_retrain(a)
                        except Exception as _e:
                            logger.warning("[비상재학습] %s 태스크 예외: %s", a.agent_id, _e)

                    _asyncio.create_task(_safe_retrain())

    async def _emergency_retrain(self, agent) -> None:
        """5연속 손실 감지 시 해당 에이전트 즉시 재학습 (하루 3회 제한)."""
        from backend.api import upbit as _upbit, kis as _kis
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        try:
            # 하루 횟수 확인 (날짜 바뀌면 리셋)
            today = _dt.now(_ZI("Asia/Seoul")).strftime("%Y-%m-%d")
            if agent._last_retrain_date != today:
                agent._daily_retrain_count = 0
                agent._last_retrain_date = today
            agent._daily_retrain_count += 1
            logger.info("[비상재학습] %s 시작 (오늘 %d회차)", agent.agent_id, agent._daily_retrain_count)

            if agent.market == "coin":
                for _t in ["KRW-BTC", "KRW-ETH", "KRW-XRP"]:
                    try:
                        _ohlcv = await _upbit.get_ohlcv(_t, interval=agent.interval_str, count=700)
                        if len(_ohlcv) >= 200:
                            async with agent._train_lock:  # daily_retrain과 race condition 방지
                                ok = await asyncio.to_thread(agent.train, _ohlcv)
                            if ok:
                                logger.info("[비상재학습] %s 코인 모델 갱신 완료 (%s)", agent.agent_id, _t)
                            return
                    except Exception:
                        continue
            else:
                for _sym in ["005930", "000660"]:
                    try:
                        _ohlcv = await _kis.get_minute_ohlcv(_sym, agent.interval_min, count=500)
                        if len(_ohlcv) >= 100:
                            async with agent._train_lock:  # daily_retrain과 race condition 방지
                                ok = await asyncio.to_thread(agent.train, _ohlcv)
                            if ok:
                                logger.info("[비상재학습] %s 주식 모델 갱신 완료 (%s)", agent.agent_id, _sym)
                            return
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("[비상재학습] %s 실패: %s", agent.agent_id, e)

    async def _daily_retrain(self) -> None:
        """매일 06:07 KST — 전 에이전트 최신 데이터로 재학습.

        코인: BTC·ETH·XRP·SOL 4개 순차 시도, 60분봉 700봉 (약 29일치)
        주식: 삼성전자·SK하이닉스·NAVER 3개 순차 시도, 15분봉 500봉 (약 25거래일치)
        에이전트별로 피처셋이 다르므로 같은 데이터로 학습해도 모델이 달라짐.
        """
        from backend.api import upbit as _upbit, kis as _kis
        from backend.ml.agents import AGENTS, ENSEMBLE_AGENTS

        logger.info("[Retrain] 전 에이전트 일일 재학습 시작")

        # ── 학습 데이터 수집 ─────────────────────────────────────
        COIN_TICKERS  = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
        STOCK_SYMBOLS = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER

        coin_ohlcv: list[dict] = []
        for ticker in COIN_TICKERS:
            try:
                coin_ohlcv = await _upbit.get_ohlcv(ticker, interval="minutes/60", count=700)
                if len(coin_ohlcv) >= 200:
                    logger.info("[Retrain] 코인 학습 데이터: %s (%d봉, 60분봉)", ticker, len(coin_ohlcv))
                    break
            except Exception as e:
                logger.warning("[Retrain] 코인 OHLCV 수집 실패 (%s): %s", ticker, e)
                continue

        if not coin_ohlcv:
            logger.error("[Retrain] 코인 학습 데이터 수집 전부 실패 — 코인 에이전트 재학습 건너뜀")

        # 주식 학습 데이터 수집 (symbol → ohlcv 딕셔너리)
        # 1순위: _agent_tick 이 누적한 틱 캐시 전체 활용 (재시작 후 25분 이상 경과 시 50종목 데이터)
        _stock_retrain_map: dict[str, list[dict]] = {}
        for _k, _v in self._stock_train_cache.items():
            sym = _k.split(":")[0]
            if len(_v) >= 50 and sym not in _stock_retrain_map:
                _stock_retrain_map[sym] = list(_v)
        if _stock_retrain_map:
            logger.info("[Retrain] 주식 학습 데이터(틱캐시): %d종목", len(_stock_retrain_map))
        else:
            # 2순위: 캐시 없으면(재시작 직후) 대표 종목 API 직접 조회, 최대 3회 재시도
            for symbol in STOCK_SYMBOLS:
                for attempt in range(3):
                    try:
                        _tmp = await _kis.get_minute_ohlcv(symbol, 15, count=500)
                        if len(_tmp) >= 50:
                            _stock_retrain_map[symbol] = _tmp
                            logger.info("[Retrain] 주식 학습 데이터(API): %s (%d봉, 15분봉)", symbol, len(_tmp))
                            break
                        await asyncio.sleep(3)
                    except Exception as e:
                        logger.warning("[Retrain] 주식 OHLCV 수집 실패 %s (시도%d): %s", symbol, attempt+1, e)
                        await asyncio.sleep(5)
        # 하위 호환: pool/map 변수 유지 (텔레그램 알림 등에서 참조)
        stock_ohlcv_pool = list(_stock_retrain_map.values())
        stock_ohlcv      = stock_ohlcv_pool[0] if stock_ohlcv_pool else []

        # ── 코인 펀딩비 히스토리 수집 (BTC 대표값, 50봉 = 약 16일치) ──
        from backend.api.binance import get_historical_funding_rates
        coin_funding: list[dict] = []
        try:
            coin_funding = await get_historical_funding_rates("BTCUSDT", limit=50)
            logger.info("[Retrain] 펀딩비 히스토리: %d개", len(coin_funding))
        except Exception:
            logger.warning("[Retrain] 펀딩비 히스토리 수집 실패, 펀딩비=0으로 학습")

        # ── BTC 학습용 데이터 (코인 에이전트 상관관계 피처용) ────────
        btc_train_ohlcv: list[dict] = []
        try:
            btc_train_ohlcv = await _upbit.get_ohlcv("KRW-BTC", interval="minutes/60", count=700)
            logger.info("[Retrain] BTC 학습 데이터: %d봉 (60분봉)", len(btc_train_ohlcv))
        except Exception:
            logger.warning("[Retrain] BTC 상관관계 학습 데이터 수집 실패")

        # ── KOSPI 학습용 데이터 (주식 에이전트 상대강도 피처용) ──────
        kospi_train_ohlcv: list[dict] = []
        if len(self._kospi_train_cache) >= 50:
            kospi_train_ohlcv = list(self._kospi_train_cache)
            logger.info("[Retrain] KOSPI 학습 데이터(캐시): %d봉", len(kospi_train_ohlcv))
        else:
            try:
                kospi_train_ohlcv = await _kis.get_minute_ohlcv("0001", 15, count=200)
                logger.info("[Retrain] KOSPI 학습 데이터(API): %d봉 (15분봉)", len(kospi_train_ohlcv))
            except Exception:
                logger.warning("[Retrain] KOSPI 학습 데이터 수집 실패")

        # ── BTC 선물 OI + Taker + L/S 히스토리 (코인 에이전트 학습용) ──
        from backend.api.binance import get_open_interest_hist, get_taker_ratio_hist, get_ls_ratio_hist
        btc_oi_train:    list[dict] = []
        btc_taker_train: list[dict] = []
        btc_ls_train:    list[dict] = []
        try:
            btc_oi_train, btc_taker_train, btc_ls_train = await asyncio.gather(
                get_open_interest_hist("BTCUSDT", period="1h", limit=700),
                get_taker_ratio_hist("BTCUSDT", period="1h", limit=700),
                get_ls_ratio_hist("BTCUSDT", period="1h", limit=700),
            )
            logger.info("[Retrain] BTC OI: %d개, Taker비율: %d개, L/S비율: %d개",
                        len(btc_oi_train), len(btc_taker_train), len(btc_ls_train))
        except Exception:
            logger.warning("[Retrain] BTC OI/Taker/L/S 데이터 수집 실패")

        # ── 에이전트 거래 결과 조회 (수익/손실 패턴 흡수용) ─────────
        import aiosqlite
        from backend.db.database import connect_db
        agent_trade_results: dict[str, list[dict]] = {}
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
                async with db.execute(
                    """
                    SELECT b.traded_at AS buy_at, b.buy_prob, b.buy_adx, b.buy_vol_ratio,
                           s.profit_rate
                    FROM agent_trades b
                    INNER JOIN agent_trades s
                        ON  s.agent_id = b.agent_id
                        AND s.ticker   = b.ticker
                        AND s.action   IN ('SELL', 'SELL_PARTIAL')
                        AND s.id = (
                            SELECT MIN(id) FROM agent_trades
                            WHERE agent_id = b.agent_id
                              AND ticker   = b.ticker
                              AND action   IN ('SELL', 'SELL_PARTIAL')
                              AND id > b.id
                        )
                    WHERE b.action = 'BUY'
                      AND s.profit_rate IS NOT NULL
                      AND b.agent_id = ?
                    ORDER BY b.id DESC
                    LIMIT 300
                    """,
                    (agent.agent_id,),
                ) as cur:
                    rows = await cur.fetchall()
                agent_trade_results[agent.agent_id] = [dict(r) for r in rows]
                if rows:
                    logger.info(
                        "[Retrain][%s] 거래 결과 %d건 로드 (수익: %d건)",
                        agent.agent_id, len(rows),
                        sum(1 for r in rows if float(r["profit_rate"] or 0) > 0),
                    )

        # 주식 에이전트 재학습용 multi-stock 딕셔너리 (위에서 이미 구성됨)
        _stock_ohlcv_by_sym = _stock_retrain_map

        # ── 에이전트별 재학습 (AI01-AI20 + ENSEMBLE_COIN/ENSEMBLE_STOCK) ───
        from backend.ml.agents import ENSEMBLE_AGENTS
        success = 0
        for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
            fr         = coin_funding if agent.market == "coin" else None
            _btc_ref   = btc_train_ohlcv  if agent.market == "coin"  else None
            _kospi_ref = kospi_train_ohlcv if agent.market == "stock" else None
            _oi_ref    = btc_oi_train      if agent.market == "coin"  else None
            _taker_ref = btc_taker_train   if agent.market == "coin"  else None
            if agent.market == "coin":
                agent._cached_funding_rates = coin_funding
                if btc_oi_train:
                    agent._cached_oi_hist    = btc_oi_train
                if btc_taker_train:
                    agent._cached_taker_hist = btc_taker_train
                if btc_ls_train:
                    agent._cached_ls_hist    = btc_ls_train
            _trade_res = agent_trade_results.get(agent.agent_id) or None
            try:
                async with agent._train_lock:
                    if agent.market == "stock":
                        # 전체 종목 데이터 풀링으로 학습 (분포 다양성 확보)
                        if not _stock_ohlcv_by_sym:
                            logger.warning("[Retrain][%s] 주식 학습 데이터 없음, 스킵", agent.agent_id)
                            continue
                        trained = await asyncio.to_thread(
                            agent.train_multi, _stock_ohlcv_by_sym, _kospi_ref
                        )
                    else:
                        data = coin_ohlcv
                        if not data:
                            logger.warning("[Retrain][%s] 코인 학습 데이터 없음, 스킵", agent.agent_id)
                            continue
                        _ls_ref = btc_ls_train if btc_ls_train else None
                        trained = await asyncio.to_thread(
                            agent.train, data, fr, _btc_ref, None, _oi_ref, _taker_ref, _trade_res, _ls_ref
                        )
                if trained:
                    success += 1
                    logger.info("[Retrain][%s] 재학습 완료", agent.agent_id)
                else:
                    logger.warning("[Retrain][%s] 재학습 실패 (데이터 부족 또는 레이블 편향)", agent.agent_id)
            except Exception:
                logger.exception("[Retrain][%s] 재학습 중 예외", agent.agent_id)

        logger.info("[Retrain] 완료: %d/22 에이전트 재학습 (AI01-AI20 + ENSEMBLE_COIN/STOCK)", success)

        # ── MFE/MAE 기반 동적 TP 배수 갱신 (R비율 분석) ─────────────────────
        for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
            _trades_for_r = agent_trade_results.get(agent.agent_id, [])
            if len(_trades_for_r) >= 10:
                _r = _compute_r_ratio(_trades_for_r)
                if _r < 1.0:
                    agent._dynamic_tp_mult = 2.5
                elif _r < 1.5:
                    agent._dynamic_tp_mult = 2.75
                elif _r < 2.5:
                    agent._dynamic_tp_mult = 3.0
                else:
                    agent._dynamic_tp_mult = 3.5
                logger.debug("[Retrain][%s] R비율=%.2f → TP배수=%.2f", agent.agent_id, _r, agent._dynamic_tp_mult)

        # ── Meta-Labeling 2차 모델 학습 (buy_prob 데이터 30개+ 시 자동 학습) ──
        meta_success = 0
        for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
            _meta_data = agent_trade_results.get(agent.agent_id, [])
            _meta_valid = [t for t in _meta_data if t.get("buy_prob") is not None]
            if len(_meta_valid) >= 30:
                try:
                    ok = await asyncio.to_thread(agent.train_meta, _meta_valid)
                    if ok:
                        meta_success += 1
                except Exception as _me:
                    logger.debug("[Retrain] %s Meta 학습 실패: %s", agent.agent_id, _me)

        try:
            await telegram.notify_message(
                f"🔄 <b>AI 일일 재학습 완료</b>\n"
                f"성공: {success}/22 에이전트 (AI01-AI20 + ENSEMBLE×2)\n"
                f"Meta-Labeling: {meta_success}개 학습\n"
                f"코인 데이터: {len(coin_ohlcv)}봉 | 주식 데이터: {len(stock_ohlcv_pool)}종목\n"
                f"※ 전일 거래내역 sample_weight 반영 → 수익률 지속 개선"
            )
        except Exception:
            pass

        # 재학습 완료 후 성능 기반 임계값 자동 조정
        await self._auto_adjust_thresholds()
        # 에이전트 검증 충분 시 실매매 ATR 손익 자동 전환 체크
        await self._check_atr_upgrade()

    async def _auto_adjust_thresholds(self) -> None:
        """재학습 후 수익률 기반 임계값 자동 조정 + 텔레그램 알림.

        수익률 < -15%: buy_threshold +0.05 (보수적 매수)
        수익률 < -30%: 비활성화 (매수 완전 스킵)
        수익률 > +5%: 임계값/활성 상태 정상 복구
        """
        from backend.ml.agents import AGENTS, AGENT_CONFIGS
        from backend.db.database import connect_db
        import aiosqlite

        default_thresholds = {cfg[0]: cfg[3] for cfg in AGENT_CONFIGS}
        adjustments: list[str] = []

        async with connect_db() as db:
            for agent in AGENTS.values():
                if agent.total_trades < 20:
                    continue  # 거래 수 부족 → 판단 보류

                ret_pct = agent.total_return * 100
                default_thr = default_thresholds.get(agent.agent_id, 0.55)

                if ret_pct < -30 and agent.is_active:
                    agent.is_active = False
                    agent.buy_threshold = min(agent.buy_threshold + 0.05, 0.80)
                    adjustments.append(
                        f"⛔ {agent.agent_id}({agent.feature_set}): 비활성화 (수익률 {ret_pct:.1f}%)"
                    )
                elif ret_pct < -15 and agent.is_active:
                    new_thr = min(round(agent.buy_threshold + 0.05, 4), 0.75)
                    if new_thr > agent.buy_threshold:
                        agent.buy_threshold = new_thr
                        adjustments.append(
                            f"⚠️ {agent.agent_id}({agent.feature_set}): 임계값→{new_thr:.2f} (수익률 {ret_pct:.1f}%)"
                        )
                elif ret_pct > 5 and not agent.is_active:
                    agent.is_active = True
                    agent.buy_threshold = default_thr
                    adjustments.append(
                        f"✅ {agent.agent_id}({agent.feature_set}): 재활성화 (수익률 {ret_pct:.1f}%)"
                    )
                # train()이 이미 재학습 후 최적 임계값을 설정했으므로 별도 복구 없음
                # (이전 elif 블록: ret_pct > 5 → default_thr 복원 → 훈련 최적화값 덮어쓰기 제거)

                await db.execute(
                    "UPDATE agent_stats SET is_active=?, buy_threshold=? WHERE agent_id=?",
                    (int(agent.is_active), round(agent.buy_threshold, 4), agent.agent_id),
                )

            await db.commit()

        if adjustments:
            msg = "🔧 <b>AI 임계값 자동 조정</b>\n\n" + "\n".join(adjustments)
        else:
            msg = "✅ <b>AI 임계값 점검 완료</b>\n전 에이전트 정상 운영 중 (조정 없음)"

        try:
            await telegram.notify_message(msg)
        except Exception:
            pass

    async def _check_atr_upgrade(self) -> None:
        """에이전트 가상매매 성과 기반 실매매 ATR 손익 자동 전환.

        조건: 활성 에이전트 총 거래 500건 이상 + 평균 수익률 > 0
        충족 시 self._use_atr_risk = True → 이후 실매매 매수 시 ATR 기반 손익 적용.
        """
        if self._use_atr_risk:
            return  # 이미 전환됨

        from backend.ml.agents import AGENTS
        active = [a for a in AGENTS.values() if a.total_trades >= 10 and a.is_active]
        if not active:
            return

        total_trades = sum(a.total_trades for a in active)
        avg_return   = sum(a.total_return for a in active) / len(active)

        if total_trades >= 500 and avg_return > 0:
            self._use_atr_risk = True
            logger.info(
                "[ATR전환] 에이전트 검증 완료 (%d건, 평균수익률 %.2f%%) → 실매매 ATR 손익 적용",
                total_trades, avg_return * 100,
            )
            try:
                await telegram.notify_message(
                    f"✅ <b>실매매 ATR 손익 자동 전환</b>\n"
                    f"에이전트 총 거래: {total_trades}건\n"
                    f"평균 수익률: {avg_return*100:+.2f}%\n\n"
                    f"고정 손절/익절 → ATR×1.5 손절 / ATR×2.0 익절\n"
                    f"(변동성 자동 적응 실매매 적용 시작)"
                )
            except Exception:
                pass
        else:
            logger.info(
                "[ATR전환] 조건 미충족 (거래 %d/500건, 평균수익률 %.2f%%)",
                total_trades, avg_return * 100,
            )

    # ── 에이전트 역할명 매핑 ──────────────────────────────────────────
    _AGENT_ROLE: dict[tuple, str] = {
        ("all",      3): "단기종합",   # RSI/MACD/추세/거래량 복합, 15분 내 청산
        ("all",      5): "중기종합",   # 복합전략, 25분 내 청산
        ("all",      8): "장기종합",   # 복합전략, 40분 내 청산
        ("momentum", 3): "단기모멘텀", # RSI·MACD·스토캐스틱, 속도 추종
        ("momentum", 5): "중기모멘텀",
        ("momentum", 8): "장기모멘텀",
        ("trend",    3): "단기추세",   # MA·ADX·VWAP·이치모쿠, 방향성 추종
        ("trend",    5): "중기추세",
        ("trend",    8): "장기추세",
        ("volume",   3): "단기거래량", # OBV·CMF·MFI·OFI, 거래량 이상 감지
        ("volume",   5): "중기거래량",
        ("volume",   8): "장기거래량",
    }

    def _build_agent_section(self, agents: list, initial_capital: float) -> str:
        """에이전트 목록 → 텔레그램용 상세 문자열 생성."""
        total_now  = sum(a._balance + a.position_value for a in agents)
        total_init = len(agents) * initial_capital
        pnl_amt    = total_now - total_init
        pnl_pct    = pnl_amt / total_init * 100 if total_init else 0
        all_trades = sum(a.total_trades for a in agents)
        all_wins   = sum(a.win_trades for a in agents)
        wr = all_wins / all_trades * 100 if all_trades else 0
        active = sum(1 for a in agents if a.is_active)

        # 전체 합산 요약
        pnl_emoji = "📈" if pnl_amt >= 0 else "📉"
        summary = (
            f"총 원금: {total_init:,.0f}원\n"
            f"총 현재: {total_now:,.0f}원\n"
            f"{pnl_emoji} 총 손익: <b>{pnl_amt:+,.0f}원 ({pnl_pct:+.2f}%)</b>\n"
            f"전체 승률: {wr:.1f}%  거래: {all_trades}건  활성: {active}/{len(agents)}"
        )

        # 에이전트별 개별 원금 → 현재가 → 손익
        lines = []
        for a in sorted(agents, key=lambda x: -x.total_return):
            flag = "⛔" if not a.is_active else ("★" if a.is_champion else "·")
            role = self._AGENT_ROLE.get((a.feature_set, a.lookahead), a.feature_set)
            cur_val   = a._balance + a.position_value
            agent_pnl = cur_val - initial_capital
            agent_pct = agent_pnl / initial_capital * 100 if initial_capital else 0
            lines.append(
                f"{flag} {a.agent_id}({role})\n"
                f"   원금 {initial_capital:,.0f} → {cur_val:,.0f}원 "
                f"| <b>{agent_pnl:+,.0f}원({agent_pct:+.1f}%)</b> "
                f"| 승률 {a.win_rate*100:.0f}% | {a.total_trades}건"
            )
        return summary + "\n\n" + "\n".join(lines)

    async def _send_daily_coin_report(self) -> None:
        """매일 06:00 KST — 코인 AI 현황 + 실제 업비트 잔고 텔레그램 전송."""
        from backend.ml.agents import AGENTS, INITIAL_CAPITAL

        now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

        # 실제 업비트 잔고
        real_lines = []
        try:
            bal = await upbit.get_balance()
            krw = bal.get("krw", 0)
            holdings = bal.get("holdings", [])
            coins_str = (
                "  ".join(
                    f"{h.get('ticker','?')} {h.get('balance',0):.4f}개 ({h.get('current_value',0):,.0f}원)"
                    for h in holdings[:5]
                ) if holdings else "없음"
            )
            real_lines = [
                f"💰 <b>실제 업비트 계좌</b>",
                f"KRW 잔고: {krw:,.0f}원",
                f"보유 코인: {coins_str}",
            ]
        except Exception:
            real_lines = ["💰 <b>실제 업비트 계좌</b>", "잔고 조회 실패"]

        coin_agents = [a for a in AGENTS.values() if a.market == "coin"]
        agent_section = self._build_agent_section(coin_agents, INITIAL_CAPITAL)

        msg = (
            f"🪙 <b>코인 AI 일일 현황</b>  {now_str}\n"
            f"{'─'*30}\n"
            + "\n".join(real_lines) + "\n"
            f"{'─'*30}\n"
            f"🤖 <b>AI 에이전트 (코인 10개)</b>\n"
            + agent_section
        )
        try:
            await telegram.notify_message(msg)
        except Exception:
            logger.exception("[레포트] 코인 일일 레포트 발송 실패")

    async def _send_daily_stock_report(self) -> None:
        """평일 15:35 KST — 주식 AI 현황 + 실제 KIS 잔고 텔레그램 전송."""
        from backend.ml.agents import AGENTS, INITIAL_CAPITAL

        now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

        # 실제 KIS 잔고
        real_lines = []
        try:
            bal = await kis.get_balance()
            cash       = bal.get("cash", 0)
            total_eval = bal.get("total_eval", 0)
            holdings   = bal.get("holdings", [])
            stocks_str = (
                "  ".join(
                    f"{h.get('name','?')} {h.get('qty',0)}주 ({h.get('profit_rate',0)*100:+.1f}%)"
                    for h in holdings[:5]
                ) if holdings else "없음"
            )
            real_lines = [
                f"💰 <b>실제 KIS 계좌</b>",
                f"예수금: {cash:,.0f}원  총평가: {total_eval:,.0f}원",
                f"보유 주식: {stocks_str}",
            ]
        except Exception:
            real_lines = ["💰 <b>실제 KIS 계좌</b>", "잔고 조회 실패 (API 키 미설정 가능)"]

        stock_agents = [a for a in AGENTS.values() if a.market == "stock"]
        agent_section = self._build_agent_section(stock_agents, INITIAL_CAPITAL)

        msg = (
            f"📈 <b>주식 AI 일일 현황</b>  {now_str}\n"
            f"{'─'*30}\n"
            + "\n".join(real_lines) + "\n"
            f"{'─'*30}\n"
            f"🤖 <b>AI 에이전트 (주식 10개)</b>\n"
            + agent_section
        )
        try:
            await telegram.notify_message(msg)
        except Exception:
            logger.exception("[레포트] 주식 일일 레포트 발송 실패")

    async def _refresh_news_scores(self) -> None:
        """30분마다 모니터링 종목 뉴스 감성 점수 갱신.

        캐시 TTL=1시간이므로 30분 주기 호출 시 최대 1.5시간 이내 최신 데이터 유지.
        get_sentiment_score()는 TTL 내라면 즉시 반환 → 실제 RSS 호출은 만료 시만 발생.
        우선순위: 포지션 보유 종목 → 코인 상위 15 → 주식 상위 10
        """
        from backend.ml.news import get_sentiment_score

        # 갱신 대상 수집
        targets: list[str] = []
        # 1순위: 현재 보유 포지션 (손절/익절 판단에 바로 영향)
        async with self._lock:
            targets.extend(list(self._positions.keys()))
        # 2순위: 모니터링 코인 상위 15개
        targets.extend(t for t in self._upbit_tickers[:15] if t not in targets)
        # 3순위: 모니터링 주식 상위 10개
        targets.extend(s for s in self._kis_symbols[:10] if s not in targets)

        if not targets:
            return

        # 최대 20개 병렬 갱신 (RSS API 부하 제한, 각 종목 내부적으로 병렬 RSS 수집)
        targets = targets[:20]
        results = await asyncio.gather(
            *[get_sentiment_score(t) for t in targets],
            return_exceptions=True,
        )
        ok = sum(1 for r in results if isinstance(r, float))
        logger.info("[뉴스갱신] %d/%d 종목 감성 점수 갱신 완료", ok, len(targets))

    async def _portfolio_snapshot(self) -> None:
        """매일 16:00 KST — KIS + 업비트 총자산 스냅샷 저장."""
        from backend.core.portfolio_snapshot import take_snapshot
        try:
            result = await take_snapshot()
            logger.info("[포트폴리오] 스냅샷 저장: 총 %.0f원", result["total_value"])
        except Exception:
            logger.exception("[포트폴리오] 스냅샷 저장 실패")

    async def _daily_reset(self) -> None:
        """매일 00:00 KST — 일일 P&L 기준점 스냅샷 + 동적 블랙리스트 업데이트."""
        from backend.ml.agents import AGENTS, ENSEMBLE_AGENTS
        from backend.db.database import connect_db

        all_agents = list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values())

        # 1) day_start_balance 스냅샷: 현재 잔액을 오늘의 기준점으로 저장
        async with connect_db() as db:
            for agent in all_agents:
                agent.day_start_balance = agent._balance + agent.position_value
                agent._today_buy_count = 0  # 일일 매수 카운터 리셋
                agent._today_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
                try:
                    await db.execute(
                        "UPDATE agent_stats SET day_start_balance=? WHERE agent_id=?",
                        (agent.day_start_balance, agent.agent_id),
                    )
                except Exception:
                    pass
            await db.commit()

        # 2) 동적 블랙리스트 업데이트: WR<35% + 거래 20건 이상 종목 차단
        async with connect_db() as db:
            async with db.execute("""
                SELECT agent_id, ticker,
                       COUNT(*) as cnt,
                       CAST(SUM(CASE WHEN profit_rate > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as wr
                FROM agent_trades
                WHERE action = 'SELL' AND profit_rate IS NOT NULL
                GROUP BY agent_id, ticker
                HAVING cnt >= 20
            """) as cur:
                rows = await cur.fetchall()

        agent_map = {a.agent_id: a for a in all_agents}
        new_bl: dict[str, set[str]] = {}
        for row in rows:
            aid, ticker, cnt, wr = row[0], row[1], row[2], row[3]
            if wr < 0.35:
                new_bl.setdefault(aid, set()).add(ticker)

        for agent in all_agents:
            agent._ticker_blacklist = new_bl.get(agent.agent_id, set())

        bl_total = sum(len(v) for v in new_bl.values())
        logger.info("[일일리셋] day_start_balance 저장 완료 / 블랙리스트 %d종목 차단", bl_total)

    def get_agents_snapshot(self) -> list[dict]:
        """에이전트 상태 스냅샷 반환 (챔피언 먼저, 이후 코인/주식 → 앙상블 순)."""
        from backend.ml.agents import AGENTS, ENSEMBLE_AGENTS
        agents = list(AGENTS.values())
        coins  = sorted([a for a in agents if a.market == "coin"],  key=lambda a: (not a.is_champion, -a.win_rate))
        stocks = sorted([a for a in agents if a.market == "stock"], key=lambda a: (not a.is_champion, -a.win_rate))
        ensembles = list(ENSEMBLE_AGENTS.values())
        return [a.to_dict() for a in coins + stocks + ensembles]


# 싱글톤 인스턴스 — FastAPI main.py 에서 import 해서 사용
scheduler = TradingScheduler()
