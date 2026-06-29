"""FastAPI 메인 앱.

엔드포인트 목록:
    GET  /health                 서버·모드 상태
    GET  /balance/kis            KIS 잔고 + 보유 종목
    GET  /balance/upbit          업비트 잔고 + 보유 코인
    GET  /price/kis/{symbol}     KIS 현재가
    GET  /price/upbit/{ticker}   업비트 현재가
    GET  /positions              현재 보유 포지션 목록
    GET  /symbols                감시 종목 목록
    POST /symbols/kis            KIS 감시 종목 추가
    DELETE /symbols/kis/{symbol} KIS 감시 종목 삭제
    POST /symbols/upbit          업비트 감시 티커 추가
    DELETE /symbols/upbit/{ticker} 업비트 감시 티커 삭제
    GET  /scheduler/status       스케줄러 상태
    POST /scheduler/start        스케줄러 시작
    POST /scheduler/stop         스케줄러 중지
    GET  /stream                 SSE — 5초마다 포지션 스냅샷 스트리밍
    GET  /ml/signal/{ticker}     XGBoost ML 신호 조회
    POST /ml/train               ML 모델 학습 트리거
    GET  /ml/status              ML 모델 학습 상태
"""

import asyncio
import json
import logging
import logging.handlers
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.api import kis, telegram, upbit
from backend.backtest.engine import SUPPORTED_STRATEGIES, run_backtest
from backend.config import settings
from backend.core.daily_report import send_daily_report
from backend.core.scheduler import scheduler
from backend.core import sim_log
from backend.db.database import init_db
from backend.models.trade import get_trades

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.handlers.TimedRotatingFileHandler(
    _LOG_DIR / "autotrade.log",
    when="midnight",      # 매일 자정에 새 파일
    backupCount=30,       # 30일치 보관
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)
_file_handler.suffix = "%Y-%m-%d"

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    await init_db()
    await scheduler.restore_positions()
    scheduler.start()
    logger.info("AutoTrade 서버 시작 (모드: %s)", settings.trade_mode)
    await telegram.notify_server_start(settings.trade_mode)
    yield
    scheduler.stop()
    await telegram.notify_server_stop()
    logger.info("AutoTrade 서버 종료")


app = FastAPI(
    title="AutoTrade API",
    version="1.0.0",
    description="주식+코인 자동매매 시스템 (KIS + Upbit)",
    lifespan=lifespan,
)

_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        return FileResponse(_DIST / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 모델 ────────────────────────────────────────────────────

class SymbolRequest(BaseModel):
    symbol: str  # KIS 종목코드 (예: '005930')


class TickerRequest(BaseModel):
    ticker: str  # 업비트 마켓 코드 (예: 'KRW-BTC')


class BacktestRequest(BaseModel):
    strategy: Literal["volatility_breakout", "moving_average", "rsi"]
    symbol: str | None = None
    ticker: str | None = None
    count: int = Field(default=120, ge=1, le=2000)
    initial_cash: float = Field(default=10_000_000, gt=0)


# ── 헬스 체크 ────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """서버 상태 및 거래 모드 확인."""
    return {
        "status": "ok",
        "mode": settings.trade_mode,
        "is_paper": settings.is_paper,
    }


# ── 잔고 조회 ────────────────────────────────────────────────────

@app.get("/balance/kis", tags=["Balance"])
async def get_kis_balance():
    """KIS 주문 가능 예수금 + 보유 종목 조회."""
    try:
        return await kis.get_balance()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/balance/upbit", tags=["Balance"])
async def get_upbit_balance():
    """업비트 KRW 잔고 + 보유 코인 조회."""
    if not settings.upbit_access_key:
        raise HTTPException(status_code=503, detail="업비트 API 키 미설정")
    try:
        return await upbit.get_balance()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ── 현재가 조회 ──────────────────────────────────────────────────

@app.get("/price/kis/{symbol}", tags=["Price"])
async def get_kis_price(symbol: str):
    """KIS 국내 주식 현재가 조회.

    Args:
        symbol: 종목코드 6자리 (예: 005930)
    """
    try:
        return await kis.get_price(symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/price/upbit/{ticker}", tags=["Price"])
async def get_upbit_price(ticker: str):
    """업비트 코인 현재가 조회.

    Args:
        ticker: 마켓 코드 (예: KRW-BTC)
    """
    try:
        return await upbit.get_price(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ── 포지션 조회 ──────────────────────────────────────────────────

@app.get("/positions", tags=["Positions"])
async def get_positions():
    """현재 보유 중인 모든 포지션 반환.

    Returns:
        { "symbol": { entry_price, qty, stop_loss_price, ... } }
    """
    positions = scheduler.get_positions()
    return {symbol: asdict(pos) for symbol, pos in positions.items()}


# ── 감시 종목 관리 ───────────────────────────────────────────────

@app.get("/symbols", tags=["Symbols"])
async def get_symbols():
    """현재 감시 중인 KIS 종목 + 업비트 티커 목록."""
    return {
        "kis": scheduler._kis_symbols,
        "upbit": scheduler._upbit_tickers,
    }


@app.post("/symbols/kis", status_code=201, tags=["Symbols"])
async def add_kis_symbol(body: SymbolRequest):
    """KIS 감시 종목 추가.

    다음 틱부터 전략 분석 및 리스크 관리 대상에 포함된다.
    """
    scheduler.add_kis_symbol(body.symbol)
    return {"added": body.symbol, "kis_symbols": scheduler._kis_symbols}


@app.delete("/symbols/kis/{symbol}", tags=["Symbols"])
async def remove_kis_symbol(symbol: str):
    """KIS 감시 종목 삭제 (보유 포지션은 유지됨)."""
    scheduler.remove_kis_symbol(symbol)
    return {"removed": symbol, "kis_symbols": scheduler._kis_symbols}


@app.post("/symbols/upbit", status_code=201, tags=["Symbols"])
async def add_upbit_ticker(body: TickerRequest):
    """업비트 감시 티커 추가."""
    scheduler.add_upbit_ticker(body.ticker)
    return {"added": body.ticker, "upbit_tickers": scheduler._upbit_tickers}


@app.delete("/symbols/upbit/{ticker}", tags=["Symbols"])
async def remove_upbit_ticker(ticker: str):
    """업비트 감시 티커 삭제 (보유 포지션은 유지됨)."""
    scheduler.remove_upbit_ticker(ticker)
    return {"removed": ticker, "upbit_tickers": scheduler._upbit_tickers}


# ── 스케줄러 제어 ────────────────────────────────────────────────

@app.get("/scheduler/status", tags=["Scheduler"])
async def scheduler_status():
    """스케줄러 실행 상태 조회."""
    return {
        "running": scheduler._scheduler.running,
        "kis_symbols": scheduler._kis_symbols,
        "upbit_tickers": scheduler._upbit_tickers,
        "position_count": len(scheduler.get_positions()),
    }


@app.post("/scheduler/start", tags=["Scheduler"])
async def start_scheduler():
    """스케줄러 시작. 이미 실행 중이면 현재 상태 반환."""
    if scheduler._scheduler.running:
        return {"status": "already_running"}
    scheduler.start()
    return {"status": "started"}


@app.post("/scheduler/stop", tags=["Scheduler"])
async def stop_scheduler():
    """스케줄러 중지. 실행 중인 포지션은 유지됨 (주문 중단만)."""
    scheduler.stop()
    return {"status": "stopped"}


# ── 백테스트 ────────────────────────────────────────────────────

@app.post("/backtest", tags=["Backtest"])
async def backtest(body: BacktestRequest):
    """백테스트 실행.

    symbol(KIS) 또는 ticker(Upbit) 중 하나를 반드시 지정해야 한다.
    Backtrader는 CPU-bound이므로 별도 스레드에서 실행.
    """
    if not body.symbol and not body.ticker:
        raise HTTPException(status_code=400, detail="symbol(KIS) 또는 ticker(Upbit) 중 하나를 입력하세요.")
    if body.symbol and body.ticker:
        raise HTTPException(status_code=400, detail="symbol과 ticker를 동시에 지정할 수 없습니다.")

    try:
        if body.symbol:
            ohlcv = await kis.get_daily_ohlcv(body.symbol, count=body.count)
        else:
            ohlcv = await upbit.get_ohlcv(body.ticker, interval="days", count=min(body.count, 200))  # type: ignore[arg-type]

        result = await asyncio.to_thread(
            run_backtest,
            ohlcv,
            body.strategy,
            body.initial_cash,
            is_crypto=bool(body.ticker),
        )
        return asdict(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ── 리포트 ──────────────────────────────────────────────────────

@app.post("/report/daily", tags=["Report"])
async def trigger_daily_report():
    """일일 백테스트 리포트 즉시 발송 (테스트·수동 트리거용)."""
    await send_daily_report()
    return {"status": "sent"}


# ── 차트 데이터 ──────────────────────────────────────────────────

_UPBIT_INTERVALS = {
    "minutes/1", "minutes/3", "minutes/5", "minutes/10", "minutes/15",
    "minutes/30", "minutes/60", "minutes/240", "days", "weeks", "months",
}

@app.get("/chart/upbit/{ticker}", tags=["Chart"])
async def get_upbit_chart(
    ticker: str,
    interval: str = "days",
    count: int = Query(default=100, ge=1, le=200),
):
    """업비트 OHLCV 차트 데이터 조회."""
    if interval not in _UPBIT_INTERVALS:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 interval: {interval}")
    try:
        return await upbit.get_ohlcv(ticker, interval=interval, count=count)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/chart/kis/{symbol}", tags=["Chart"])
async def get_kis_chart(symbol: str, count: int = Query(default=100, ge=1, le=200)):
    """KIS 일봉 OHLCV 차트 데이터 조회."""
    try:
        return await kis.get_daily_ohlcv(symbol, count=min(count, 200))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/simlog", tags=["Chart"])
async def get_simlog(n: int = Query(default=100, ge=1, le=500)):
    """시뮬레이션 로그 최신 n개 조회."""
    return sim_log.get_logs(n)


# ── 거래 기록 ────────────────────────────────────────────────────

@app.get("/trades", tags=["Trades"])
async def get_trade_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """거래 실행 이력 조회 (최신순).

    Args:
        limit:  최대 반환 건수 (기본 50, 최대 200)
        offset: 페이지 오프셋
    """
    return await get_trades(min(limit, 200), offset)


# ── ML 신호 ────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    tickers: list[str] | None = None


@app.get("/ml/signal/{ticker}", tags=["ML"])
async def get_ml_signal(ticker: str):
    """XGBoost 기반 ML 매수/매도 신호 조회.

    모델이 없으면 404. 먼저 POST /ml/train 호출 필요.
    """
    from dataclasses import asdict as _asdict

    from backend.ml.model import XGBSignalModel

    try:
        ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=200)
        model = XGBSignalModel(ticker)
        result = await model.predict(ohlcv, settings.cryptopanic_token)
        return _asdict(result)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/ml/train", tags=["ML"])
async def trigger_ml_train(body: TrainRequest):
    """XGBoost 모델 학습 트리거.

    tickers 생략 시 기본 5개 코인(BTC/ETH/XRP/SOL/DOGE) 학습.
    학습 완료까지 수십 초 소요 가능.
    """
    from backend.ml.trainer import train_all

    results = await train_all(body.tickers)
    return {"trained": results}


@app.get("/ml/status", tags=["ML"])
async def get_ml_status():
    """ML 모델 학습 상태 (마지막 학습 시각, 정확도 등) 조회."""
    from backend.ml.trainer import get_status

    return get_status()


# ── AI 에이전트 경쟁 ─────────────────────────────────────────────

@app.get("/agents", tags=["Agents"])
async def get_agents():
    """20개 AI 에이전트 현재 상태 조회 (승률, 수익률, 포지션 등)."""
    return scheduler.get_agents_snapshot()


@app.get("/agents/{agent_id}/trades", tags=["Agents"])
async def get_agent_trades(
    agent_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """특정 에이전트 가상 거래 기록 조회."""
    try:
        from backend.db.database import connect_db
        import aiosqlite
        async with connect_db() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM agent_trades WHERE agent_id=? ORDER BY id DESC LIMIT ?",
                (agent_id.upper(), limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ── SSE 실시간 스트리밍 ──────────────────────────────────────────

@app.get("/stream", tags=["Stream"])
async def stream(request: Request):
    """SSE 스트림 — 5초마다 포지션 스냅샷 브로드캐스트.

    Event types:
        positions — 현재 포지션 전체 (entry_price, qty, stop/take 가격 등)
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            positions = scheduler.get_positions()
            yield {
                "event": "positions",
                "data": json.dumps(
                    {s: asdict(p) for s, p in positions.items()},
                    ensure_ascii=False,
                ),
            }

            yield {
                "event": "simlog",
                "data": json.dumps(sim_log.get_logs(50), ensure_ascii=False),
            }

            yield {
                "event": "mlsignal",
                "data": json.dumps(scheduler.get_ml_signals(), ensure_ascii=False),
            }

            yield {
                "event": "agents",
                "data": json.dumps(scheduler.get_agents_snapshot(), ensure_ascii=False),
            }

            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())
