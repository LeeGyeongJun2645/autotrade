"""SQLite DB 초기화 및 연결 관리.

파일 위치: data/autotrade.db (루트 기준)
테이블:
    trades    — 거래 실행 로그 (매수/매도)
    positions — 포지션 영속성 (서버 재시작 후 복구용)
"""

import logging
from contextlib import asynccontextmanager
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
    is_active       INTEGER DEFAULT 1,
    updated_at      TEXT
)
"""

_CREATE_PORTFOLIO_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT    NOT NULL,
    kis_cash    REAL    DEFAULT 0.0,
    kis_stocks  REAL    DEFAULT 0.0,
    upbit_krw   REAL    DEFAULT 0.0,
    upbit_coins REAL    DEFAULT 0.0,
    total_value REAL    NOT NULL
)
"""


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        # busy_timeout을 WAL 전환 전에 설정해야 전환 자체의 SQLITE_BUSY도 처리됨
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(_CREATE_TRADES)
        await db.execute(_CREATE_POSITIONS)
        await db.execute(_CREATE_AGENT_TRADES)
        await db.execute(_CREATE_AGENT_POSITIONS)
        await db.execute(_CREATE_AGENT_STATS)
        await db.execute(_CREATE_PORTFOLIO_SNAPSHOTS)
        # 기존 DB migration: 누락 컬럼 추가 (OperationalError = 이미 존재, 그 외는 재발생)
        _migrations = [
            "ALTER TABLE agent_stats ADD COLUMN is_active       INTEGER DEFAULT 1",
            "ALTER TABLE agent_stats ADD COLUMN label_threshold REAL    DEFAULT 0.02",
            "ALTER TABLE agent_stats ADD COLUMN buy_threshold   REAL    DEFAULT 0.55",
            "ALTER TABLE agent_stats ADD COLUMN feature_set     TEXT    DEFAULT 'all'",
            "ALTER TABLE agent_stats ADD COLUMN is_champion     INTEGER DEFAULT 0",
            "ALTER TABLE agent_trades ADD COLUMN entry_price    REAL",
        ]
        for sql in _migrations:
            try:
                await db.execute(sql)
            except aiosqlite.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise
        # 조회 성능용 인덱스 (IF NOT EXISTS로 멱등)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_trades_agent ON agent_trades(agent_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_trades_ticker ON agent_trades(ticker)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_trades_at ON agent_trades(traded_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_at ON portfolio_snapshots(snapshot_at)")
        await db.commit()
    logger.info("DB 초기화 완료: %s", DB_PATH)


def get_db_path() -> Path:
    return DB_PATH


@asynccontextmanager
async def connect_db():
    """busy_timeout=5000 이 적용된 DB 연결 컨텍스트 매니저.

    WAL 모드는 init_db() 에서 DB 파일에 영구 적용되므로 재설정 불필요.
    busy_timeout 은 커넥션별 설정이므로 매번 적용해야 한다.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA busy_timeout=5000")
        yield db
