"""시장 분석 에이전트.

30분마다 실행하여:
1. KOSPI/KOSDAQ + 코인 시장 레짐(bull/bear/sideways) 분석
2. 에이전트 매수/차단 이유 자동 기록 (market_decisions 테이블)
3. 청산 후 피처 기여도 사후 분석
4. 뉴스 헤드라인 → 시장 영향 요약
5. 분석 결과 텔레그램 리포트 (1시간마다)

DB 테이블: market_decisions, market_regimes
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import aiosqlite

from backend.db.database import connect_db
from backend.config import settings

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")

# ── 마지막 텔레그램 발송 시각 (하루 1회) ─────────────────────────
_last_report_at: float = 0.0
_REPORT_COOLDOWN = 86400.0  # 24h — 시장 분석 결과는 학습에 자동 반영, 리포트는 하루 1회만

# ── 시장 레짐 캐시 ────────────────────────────────────────────────
_market_regime_cache: dict = {}  # {"kospi": "bull"|"bear"|"sideways", "coin": ..., "ts": float}


def get_regime_adaptive_thresholds(market: str) -> dict:
    """현재 장세에 따른 ML 매수 기준 동적 조정값 반환.

    반환:
        label_threshold_mult: 재학습 시 label_threshold 배수 (0.7~1.1)
        avg_thr_mult:         predict_ensemble avg_thr 조정 배수 (0.80~1.0)
        min_buy_votes:        최소 매수 동의 에이전트 수
    """
    regime = _market_regime_cache.get(market, "unknown")
    if regime == "bear":
        # 약세장: 레이블 기준 낮춤(buy 샘플 확보) + avg_thr 완화(회복 신호 포착)
        return {"label_threshold_mult": 0.70, "avg_thr_mult": 0.82, "min_buy_votes": 1}
    elif regime == "bull":
        # 강세장: 레이블 기준 높임(품질 유지) + avg_thr 정상(불필요한 추격매수 방지)
        return {"label_threshold_mult": 1.05, "avg_thr_mult": 1.00, "min_buy_votes": 2}
    else:  # sideways / unknown
        # 횡보장: 적당히 완화
        return {"label_threshold_mult": 0.85, "avg_thr_mult": 0.90, "min_buy_votes": 1}


# ── DB 테이블 DDL ─────────────────────────────────────────────────

_CREATE_MARKET_DECISIONS = """
CREATE TABLE IF NOT EXISTS market_decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at  TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    market      TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    final_prob  REAL,
    buy_votes   INTEGER,
    avg_thr     REAL,
    top_features TEXT,
    reason      TEXT,
    regime      TEXT,
    news_score  REAL
)
"""

_CREATE_MARKET_REGIMES = """
CREATE TABLE IF NOT EXISTS market_regimes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT    NOT NULL,
    kospi_ret1d REAL,
    kospi_ret5d REAL,
    kosdaq_ret1d REAL,
    coin_ret1d  REAL,
    regime_coin TEXT,
    regime_stock TEXT,
    top_rising  TEXT,
    top_falling TEXT,
    news_summary TEXT
)
"""

_CREATE_POST_TRADE = """
CREATE TABLE IF NOT EXISTS post_trade_analysis (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analyzed_at     TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    market          TEXT    NOT NULL,
    entry_price     REAL,
    exit_price      REAL,
    profit_rate     REAL,
    hold_minutes    INTEGER,
    top_features    TEXT,
    regime_at_entry TEXT,
    news_score      REAL,
    outcome         TEXT
)
"""


async def init_analyst_db() -> None:
    """분석 테이블 초기화 (init_db 이후 호출)."""
    async with connect_db() as db:
        await db.execute(_CREATE_MARKET_DECISIONS)
        await db.execute(_CREATE_MARKET_REGIMES)
        await db.execute(_CREATE_POST_TRADE)
        _migrations = [
            "CREATE INDEX IF NOT EXISTS idx_mkt_dec_symbol ON market_decisions(symbol, decided_at)",
            "CREATE INDEX IF NOT EXISTS idx_mkt_reg_at ON market_regimes(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_post_trade_symbol ON post_trade_analysis(symbol, analyzed_at)",
        ]
        for sql in _migrations:
            try:
                await db.execute(sql)
            except aiosqlite.OperationalError:
                pass
        await db.commit()
    logger.info("[MarketAnalyst] DB 테이블 초기화 완료")


# ── 시장 레짐 감지 ────────────────────────────────────────────────

async def detect_market_regime() -> dict:
    """KOSPI/코인 장세 분석 → {"coin": "bull"|"bear"|"sideways", "stock": ...}"""
    global _market_regime_cache
    if _market_regime_cache and time.time() - _market_regime_cache.get("ts", 0) < 900:
        return _market_regime_cache

    result = {"coin": "unknown", "stock": "unknown", "ts": time.time(),
              "kospi_ret1d": 0.0, "kospi_ret5d": 0.0,
              "coin_ret1d": 0.0, "top_rising": [], "top_falling": []}

    # KOSPI 지수 분석
    try:
        from backend.api import kis
        kospi_ohlcv = await kis.get_daily_ohlcv("0001", count=10)  # KOSPI 지수코드
        if len(kospi_ohlcv) >= 6:
            today  = float(kospi_ohlcv[-1]["close"])
            prev1  = float(kospi_ohlcv[-2]["close"])
            prev5  = float(kospi_ohlcv[-6]["close"])
            ret1d = (today - prev1) / prev1
            ret5d = (today - prev5) / prev5
            result["kospi_ret1d"] = round(ret1d * 100, 2)
            result["kospi_ret5d"] = round(ret5d * 100, 2)
            if ret5d > 0.015:
                result["stock"] = "bull"
            elif ret5d < -0.015:
                result["stock"] = "bear"
            else:
                result["stock"] = "sideways"
    except Exception as e:
        logger.debug("[MarketAnalyst] KOSPI 레짐 분석 실패: %s", e)

    # 코인 시장 — BTC 기준
    try:
        from backend.api import upbit
        btc_ohlcv = await upbit.get_ohlcv("KRW-BTC", interval="days", count=7)
        if len(btc_ohlcv) >= 6:
            today = float(btc_ohlcv[-1]["close"])
            prev1 = float(btc_ohlcv[-2]["close"])
            prev5 = float(btc_ohlcv[-6]["close"])
            ret1d = (today - prev1) / prev1
            ret5d = (today - prev5) / prev5
            result["coin_ret1d"] = round(ret1d * 100, 2)
            if ret5d > 0.05:
                result["coin"] = "bull"
            elif ret5d < -0.05:
                result["coin"] = "bear"
            else:
                result["coin"] = "sideways"
    except Exception as e:
        logger.debug("[MarketAnalyst] BTC 레짐 분석 실패: %s", e)

    # 상승/하락 종목 탑 5
    try:
        from backend.api import kis
        vol_top = await kis.get_volume_rank(30)
        rising, falling = [], []
        for sym in vol_top[:20]:
            try:
                pd_ = await kis.get_price(sym)
                cr = float(pd_.get("change_rate", 0))
                name = kis.get_stock_names().get(sym, sym)
                if cr >= 3.0:
                    rising.append(f"{name}({cr:+.1f}%)")
                elif cr <= -3.0:
                    falling.append(f"{name}({cr:+.1f}%)")
            except Exception:
                pass
        result["top_rising"]  = rising[:5]
        result["top_falling"] = falling[:5]
    except Exception as e:
        logger.debug("[MarketAnalyst] 상승/하락 종목 분석 실패: %s", e)

    _market_regime_cache = result
    return result


async def log_trade_decision(
    symbol: str,
    market: str,
    signal: str,
    final_prob: float,
    buy_votes: int,
    avg_thr: float,
    top_features: list[tuple],
    reason: str,
    news_score: float = 0.0,
) -> None:
    """ML게이트 결정 (buy/sell/hold 차단) 기록."""
    try:
        regime = _market_regime_cache.get(market, "unknown")
        now = datetime.now(KST).isoformat()
        features_json = json.dumps(top_features[:5], ensure_ascii=False)
        async with connect_db() as db:
            await db.execute(
                """INSERT INTO market_decisions
                   (decided_at, symbol, market, signal, final_prob, buy_votes, avg_thr,
                    top_features, reason, regime, news_score)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (now, symbol, market, signal, round(final_prob, 4), buy_votes,
                 round(avg_thr, 4), features_json, reason[:200], str(regime), round(news_score, 3)),
            )
            await db.commit()
    except Exception as e:
        logger.debug("[MarketAnalyst] 결정 기록 실패: %s", e)


async def log_post_trade(
    symbol: str,
    market: str,
    entry_price: float,
    exit_price: float,
    hold_minutes: int,
    top_features: list[tuple],
    news_score: float = 0.0,
) -> None:
    """청산 후 사후 분석 기록."""
    try:
        profit_rate = (exit_price - entry_price) / entry_price
        outcome = "WIN" if profit_rate > 0 else "LOSS"
        regime_at_entry = json.dumps(_market_regime_cache, ensure_ascii=False)[:200]
        now = datetime.now(KST).isoformat()
        async with connect_db() as db:
            await db.execute(
                """INSERT INTO post_trade_analysis
                   (analyzed_at, symbol, market, entry_price, exit_price, profit_rate,
                    hold_minutes, top_features, regime_at_entry, news_score, outcome)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (now, symbol, market, entry_price, exit_price, round(profit_rate, 4),
                 hold_minutes, json.dumps(top_features[:5], ensure_ascii=False),
                 regime_at_entry, round(news_score, 3), outcome),
            )
            await db.commit()
        logger.info("[MarketAnalyst][%s] 사후분석 기록 → %s (%.2f%%)", symbol, outcome, profit_rate * 100)
    except Exception as e:
        logger.debug("[MarketAnalyst] 사후분석 기록 실패: %s", e)


async def save_regime_snapshot(regime: dict) -> None:
    """시장 레짐 스냅샷 DB 저장."""
    try:
        async with connect_db() as db:
            await db.execute(
                """INSERT INTO market_regimes
                   (recorded_at, kospi_ret1d, kospi_ret5d, kosdaq_ret1d, coin_ret1d,
                    regime_coin, regime_stock, top_rising, top_falling, news_summary)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(KST).isoformat(),
                    regime.get("kospi_ret1d", 0.0),
                    regime.get("kospi_ret5d", 0.0),
                    0.0,
                    regime.get("coin_ret1d", 0.0),
                    regime.get("coin", "unknown"),
                    regime.get("stock", "unknown"),
                    json.dumps(regime.get("top_rising", []), ensure_ascii=False),
                    json.dumps(regime.get("top_falling", []), ensure_ascii=False),
                    "",
                ),
            )
            await db.commit()
    except Exception as e:
        logger.debug("[MarketAnalyst] 레짐 스냅샷 저장 실패: %s", e)


async def get_recent_decision_summary() -> dict:
    """최근 1시간 결정 통계 반환."""
    try:
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT market, signal, COUNT(*) as cnt, AVG(final_prob) as avg_prob
                   FROM market_decisions
                   WHERE decided_at >= datetime('now', '-1 hour', 'localtime')
                   GROUP BY market, signal""",
            )
            rows = await cur.fetchall()
        return {f"{r['market']}_{r['signal']}": {"count": r["cnt"], "avg_prob": r["avg_prob"]} for r in rows}
    except Exception:
        return {}


async def get_recent_trade_performance() -> dict:
    """최근 7일 실매매 승률/수익률 요약."""
    try:
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN profit_rate > 0 THEN 1 ELSE 0 END) as wins,
                     AVG(profit_rate) as avg_ret,
                     MIN(profit_rate) as worst,
                     MAX(profit_rate) as best
                   FROM trades
                   WHERE side='SELL'
                     AND profit_rate IS NOT NULL
                     AND created_at >= datetime('now', '-7 days', 'localtime')""",
            )
            row = await cur.fetchone()
            if row and row["total"]:
                return dict(row)
    except Exception:
        pass
    return {}


async def run_market_analysis() -> None:
    """30분마다 실행되는 메인 분석 함수."""
    global _last_report_at

    # 1. 장세 파악
    regime = await detect_market_regime()
    logger.info(
        "[MarketAnalyst] 장세 | 주식=%s(KOSPI 1d%+.1f%% 5d%+.1f%%) | 코인=%s(BTC 1d%+.1f%%)",
        regime["stock"], regime["kospi_ret1d"], regime["kospi_ret5d"],
        regime["coin"], regime["coin_ret1d"],
    )

    # 2. 상승/하락 종목 로그
    if regime["top_rising"]:
        logger.info("[MarketAnalyst] 급등 종목: %s", " | ".join(regime["top_rising"]))
    if regime["top_falling"]:
        logger.info("[MarketAnalyst] 급락 종목: %s", " | ".join(regime["top_falling"]))

    # 3. DB 스냅샷 저장
    await save_regime_snapshot(regime)

    # 4. 뉴스 감성 요약
    news_summary = ""
    try:
        from backend.ml.news import get_cached_score, fetch_and_score
        btc_score = await fetch_and_score("KRW-BTC")
        eth_score = await fetch_and_score("KRW-ETH")
        news_summary = f"BTC뉴스:{btc_score:+.2f} ETH뉴스:{eth_score:+.2f}"
        logger.info("[MarketAnalyst] 뉴스감성 | %s", news_summary)
    except Exception as e:
        logger.debug("[MarketAnalyst] 뉴스 분석 실패: %s", e)

    # 5. 결정 통계 (최근 1시간)
    decisions = await get_recent_decision_summary()
    if decisions:
        _parts = [f"{k}:{v['count']}건(avg_prob {v['avg_prob']:.2f})" for k, v in decisions.items()]
        logger.info("[MarketAnalyst] 최근 1h 결정: %s", " | ".join(_parts))

    # 6. 텔레그램 리포트 (1시간 쿨다운)
    if time.time() - _last_report_at >= _REPORT_COOLDOWN:
        await _send_market_report(regime, decisions, news_summary)
        _last_report_at = time.time()


async def _send_market_report(regime: dict, decisions: dict, news_summary: str) -> None:
    """시장 분석 텔레그램 리포트 발송."""
    try:
        from backend.api import telegram

        perf = await get_recent_trade_performance()
        now_str = datetime.now(KST).strftime("%m/%d %H:%M")

        # 주식 장세 이모지
        s_emoji = "📈" if regime["stock"] == "bull" else ("📉" if regime["stock"] == "bear" else "➡️")
        c_emoji = "📈" if regime["coin"] == "bull" else ("📉" if regime["coin"] == "bear" else "➡️")

        lines = [
            f"📊 *시장 분석 리포트* ({now_str})",
            "",
            f"{s_emoji} 주식: {regime['stock'].upper()} | KOSPI 1일{regime['kospi_ret1d']:+.2f}% 5일{regime['kospi_ret5d']:+.2f}%",
            f"{c_emoji} 코인: {regime['coin'].upper()} | BTC 1일{regime['coin_ret1d']:+.2f}%",
        ]

        if regime["top_rising"]:
            lines.append(f"\n🔺 급등: {', '.join(regime['top_rising'][:3])}")
        if regime["top_falling"]:
            lines.append(f"🔻 급락: {', '.join(regime['top_falling'][:3])}")

        if news_summary:
            lines.append(f"\n📰 뉴스감성: {news_summary}")

        # 매수 차단 통계
        coin_buy = decisions.get("coin_buy", {}).get("count", 0)
        coin_block = decisions.get("coin_sell", {}).get("count", 0) + decisions.get("coin_hold", {}).get("count", 0)
        stock_buy = decisions.get("stock_buy", {}).get("count", 0)
        stock_block = decisions.get("stock_sell", {}).get("count", 0) + decisions.get("stock_hold", {}).get("count", 0)

        lines.append(f"\n🤖 ML게이트(1h): 코인 매수{coin_buy}/차단{coin_block} | 주식 매수{stock_buy}/차단{stock_block}")

        # 7일 실매매 성과
        if perf.get("total", 0) > 0:
            wr = perf["wins"] / perf["total"] * 100
            lines.append(
                f"\n💰 7일 실매매: {perf['total']}건 승률{wr:.0f}% 평균수익{perf['avg_ret']*100:+.2f}%"
                f" (최악{perf['worst']*100:+.2f}% / 최고{perf['best']*100:+.2f}%)"
            )
        else:
            lines.append("\n💰 7일 실매매: 거래 없음")

        # 장세별 전략 힌트
        if regime["coin"] == "bear":
            lines.append("\n⚠️ 코인 약세장 — ML 임계값 완화 모드")
        elif regime["coin"] == "bull":
            lines.append("\n✅ 코인 강세장 — 정상 매수 조건")

        msg = "\n".join(lines)
        await telegram.notify_message(msg)
        logger.info("[MarketAnalyst] 텔레그램 리포트 발송 완료")
    except Exception as e:
        logger.warning("[MarketAnalyst] 텔레그램 리포트 실패: %s", e)


async def analyze_why_no_trades() -> str:
    """거래가 없을 때 원인 분석 문자열 반환."""
    lines = []

    # 최근 24시간 결정 통계
    try:
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT signal, COUNT(*) as cnt, AVG(final_prob) as avg_prob, AVG(avg_thr) as avg_thr_val
                   FROM market_decisions
                   WHERE decided_at >= datetime('now', '-24 hours', 'localtime')
                     AND market='coin'
                   GROUP BY signal""",
            )
            rows = await cur.fetchall()
            for r in rows:
                lines.append(
                    f"  코인 {r['signal']}: {r['cnt']}건 avg_prob={r['avg_prob']:.3f} avg_thr={r['avg_thr_val']:.3f}"
                )
    except Exception:
        pass

    # 에이전트 WF 실패 통계
    try:
        from backend.ml.agents import AGENTS
        failed = [a.agent_id for a in AGENTS.values() if a._model is None]
        trained = [a.agent_id for a in AGENTS.values() if a._model is not None]
        lines.append(f"  학습된 에이전트: {len(trained)}개 / WF실패: {len(failed)}개 ({', '.join(failed[:5])})")
    except Exception:
        pass

    return "\n".join(lines) if lines else "분석 데이터 없음"
