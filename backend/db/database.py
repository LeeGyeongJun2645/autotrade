"""SQLite DB 초기화 및 연결 관리.

파일 위치: data/autotrade.db (루트 기준)
테이블:
    trades    — 거래 실행 로그 (매수/매도)
    positions — 포지션 영속성 (서버 재시작 후 복구용)
"""

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "autotrade.db"

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    exchange    TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    qty         REAL    NOT NULL,
    price       REAL    NOT NULL,
    strategy    TEXT,
    reason      TEXT,
    profit_rate REAL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
)
"""

_CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    symbol               TEXT    PRIMARY KEY,
    exchange             TEXT    NOT NULL,
    entry_price          REAL    NOT NULL,
    qty                  REAL    NOT NULL,
    stop_loss_price      REAL    NOT NULL,
    take_profit_price    REAL    NOT NULL,
    highest_price        REAL    NOT NULL,
    trailing_stop_price  REAL    NOT NULL,
    strategy             TEXT    NOT NULL,
    opened_at            TEXT    NOT NULL,
    is_crypto            INTEGER NOT NULL DEFAULT 0
)
"""


_CREATE_AGENT_TRADES = """
CREATE TABLE IF NOT EXISTS agent_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    price       REAL    NOT NULL,
    qty         REAL    NOT NULL,
    entry_price REAL,
    profit_rate REAL,
    balance     REAL    NOT NULL,
    traded_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
)
"""

_CREATE_AGENT_POSITIONS = """
CREATE TABLE IF NOT EXISTS agent_positions (
    agent_id    TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    entry_price REAL    NOT NULL,
    qty         REAL    NOT NULL,
    entered_at  TEXT    NOT NULL,
    PRIMARY KEY (agent_id, ticker)
)
"""

_CREATE_AGENT_STATS = """
CREATE TABLE IF NOT EXISTS agent_stats (
    agent_id        TEXT    PRIMARY KEY,
    interval_min    INTEGER NOT NULL,
    label_threshold REAL    NOT NULL,
    buy_threshold   REAL    NOT NULL,
    feature_set     TEXT    NOT NULL,
    total_trades    INTEGER DEFAULT 0,
    win_trades      INTEGER DEFAULT 0,
    win_rate        REAL    DEFAULT 0.0,
    total_return    REAL    DEFAULT 0.0,
    current_balance REAL    DEFAULT 1000000.0,
    is_champion     INTEGER DEFAULT 0,
    updated_at      TEXT
)
"""


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TRADES)
        await db.execute(_CREATE_POSITIONS)
        await db.execute(_CREATE_AGENT_TRADES)
        await db.execute(_CREATE_AGENT_POSITIONS)
        await db.execute(_CREATE_AGENT_STATS)
        await db.commit()
    logger.info("DB 초기화 완료: %s", DB_PATH)


def get_db_path() -> Path:
    return DB_PATH
