"""포트폴리오 일별 스냅샷 — KIS + 업비트 총 자산 기록 모듈.

매일 16:00 KST(장 마감 30분 후)에 scheduler가 호출.
원금(initial_real_capital):
    .env INITIAL_REAL_CAPITAL > 0  → 해당 값 사용
    0 (기본)                       → 첫 스냅샷 값을 원금으로 자동 설정
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backend.db.database import connect_db

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")


async def take_snapshot() -> dict:
    """KIS + 업비트 잔고 합산 → DB 저장 후 결과 반환.

    두 거래소 잔고 모두 조회 실패(total=0)이면 저장을 건너뜀.
    """
    from backend.api import kis, upbit
    from backend.config import settings

    now_kst = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S")
    kis_cash = 0.0
    kis_stocks = 0.0
    upbit_krw = 0.0
    upbit_coins = 0.0
    any_success = False

    try:
        bal = await kis.get_balance()
        kis_cash = float(bal.get("cash") or 0)
        total_eval = float(bal.get("total_eval") or 0)
        kis_stocks = max(0.0, total_eval - kis_cash)
        any_success = True
    except Exception as e:
        logger.warning("[스냅샷] KIS 잔고 조회 실패: %s", e)

    try:
        if settings.upbit_access_key:
            bal = await upbit.get_balance()
            upbit_krw = float(bal.get("krw") or 0)
            for h in bal.get("holdings", []):
                upbit_coins += float(h.get("current_value") or 0)
            any_success = True
    except Exception as e:
        logger.warning("[스냅샷] 업비트 잔고 조회 실패: %s", e)

    total = kis_cash + kis_stocks + upbit_krw + upbit_coins

    # 모든 잔고가 0이면(API 장애) 잘못된 스냅샷이 원금 기준이 되는 것 방지
    if not any_success or total <= 0:
        logger.warning("[스냅샷] 유효한 잔고 없음(total=%.0f) — 저장 건너뜀", total)
        return {"snapshot_at": now_kst, "total_value": 0, "skipped": True}

    async with connect_db() as db:
        await db.execute(
            """INSERT INTO portfolio_snapshots
               (snapshot_at, kis_cash, kis_stocks, upbit_krw, upbit_coins, total_value)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now_kst, kis_cash, kis_stocks, upbit_krw, upbit_coins, total),
        )
        await db.commit()

    logger.info(
        "[스냅샷] 저장 완료 | KIS %.0f + 주식 %.0f | 업비트 %.0f + 코인 %.0f | 합계 %.0f",
        kis_cash, kis_stocks, upbit_krw, upbit_coins, total,
    )
    return {
        "snapshot_at": now_kst,
        "kis_cash": kis_cash,
        "kis_stocks": kis_stocks,
        "upbit_krw": upbit_krw,
        "upbit_coins": upbit_coins,
        "total_value": total,
    }


async def get_history(days: int = 30) -> list[dict]:
    """최근 N일 스냅샷 목록 (오래된 것부터).

    snapshot_at은 KST 문자열로 저장되므로 서버 TZ에 의존하는
    SQLite localtime 대신 Python에서 KST 기준 cutoff를 계산한다.
    """
    import aiosqlite

    cutoff = (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT snapshot_at, kis_cash, kis_stocks, upbit_krw, upbit_coins, total_value
               FROM portfolio_snapshots
               WHERE snapshot_at >= ?
               ORDER BY snapshot_at ASC""",
            (cutoff,),
        )
        rows = await cursor.fetchall()

    return [dict(r) for r in rows]


async def get_initial_capital() -> float:
    """원금 결정: .env INITIAL_REAL_CAPITAL > 0이면 그것, 아니면 첫 스냅샷 값."""
    from backend.config import settings
    import aiosqlite

    if settings.initial_real_capital > 0:
        return settings.initial_real_capital

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        # total_value > 0인 첫 스냅샷만 원금 기준으로 사용
        cursor = await db.execute(
            "SELECT total_value FROM portfolio_snapshots WHERE total_value > 0 ORDER BY id ASC LIMIT 1"
        )
        row = await cursor.fetchone()

    return float(row["total_value"]) if row else 0.0
