"""거래 기록 모델 및 DB CRUD.

trades 테이블: 모든 매수/매도 실행 기록
positions 테이블: 서버 재시작 후 포지션 복구용 스냅샷
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import aiosqlite

from backend.core.risk_manager import Position
from backend.db.database import connect_db

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    id: int
    symbol: str
    exchange: str       # 'KIS' | 'UPBIT'
    side: str           # 'BUY' | 'SELL'
    qty: float
    price: float
    strategy: str | None
    reason: str | None
    profit_rate: float | None
    created_at: str


# ── 거래 기록 ────────────────────────────────────────────────────


async def insert_trade(
    symbol: str,
    exchange: str,
    side: str,
    qty: float,
    price: float,
    strategy: str | None = None,
    reason: str | None = None,
    profit_rate: float | None = None,
) -> None:
    try:
        async with connect_db() as db:
            await db.execute(
                """INSERT INTO trades (symbol, exchange, side, qty, price, strategy, reason, profit_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, exchange, side, qty, price, strategy, reason, profit_rate),
            )
            await db.commit()
    except Exception:
        logger.exception("[DB] 거래 기록 저장 실패: %s %s %s", exchange, side, symbol)


async def get_trades(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    try:
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception:
        logger.exception("[DB] 거래 기록 조회 실패")
        return []


# ── 포지션 영속성 ────────────────────────────────────────────────


async def upsert_position(symbol: str, exchange: str, pos: Position) -> None:
    try:
        async with connect_db() as db:
            await db.execute(
                """INSERT INTO positions
                       (symbol, exchange, entry_price, qty, stop_loss_price, take_profit_price,
                        highest_price, trailing_stop_price, strategy, opened_at, is_crypto)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol) DO UPDATE SET
                       entry_price=excluded.entry_price,
                       qty=excluded.qty,
                       stop_loss_price=excluded.stop_loss_price,
                       take_profit_price=excluded.take_profit_price,
                       highest_price=excluded.highest_price,
                       trailing_stop_price=excluded.trailing_stop_price,
                       strategy=excluded.strategy,
                       opened_at=excluded.opened_at,
                       is_crypto=excluded.is_crypto""",
                (
                    symbol,
                    exchange,
                    pos.entry_price,
                    pos.qty,
                    pos.stop_loss_price,
                    pos.take_profit_price,
                    pos.highest_price,
                    pos.trailing_stop_price,
                    pos.strategy,
                    pos.opened_at,
                    int(pos.is_crypto),
                ),
            )
            await db.commit()
    except Exception:
        logger.exception("[DB] 포지션 저장 실패: %s", symbol)


async def delete_position(symbol: str) -> None:
    try:
        async with connect_db() as db:
            await db.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            await db.commit()
    except Exception:
        logger.exception("[DB] 포지션 삭제 실패: %s", symbol)


async def load_all_positions() -> dict[str, tuple[str, Position]]:
    """DB에서 포지션 전체 복구. 반환: {symbol: (exchange, Position)}"""
    try:
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions")
            rows = await cursor.fetchall()

        result: dict[str, tuple[str, Position]] = {}
        for r in rows:
            pos = Position(
                symbol=r["symbol"],
                entry_price=r["entry_price"],
                qty=r["qty"],
                stop_loss_price=r["stop_loss_price"],
                take_profit_price=r["take_profit_price"],
                highest_price=r["highest_price"],
                trailing_stop_price=r["trailing_stop_price"],
                strategy=r["strategy"],
                opened_at=r["opened_at"],
                is_crypto=bool(r["is_crypto"]),
            )
            result[r["symbol"]] = (r["exchange"], pos)

        if result:
            logger.info("[DB] 포지션 %d개 복구: %s", len(result), list(result.keys()))
        return result
    except Exception:
        logger.exception("[DB] 포지션 복구 실패")
        return {}
