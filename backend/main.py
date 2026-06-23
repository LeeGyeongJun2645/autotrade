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
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.api import kis, telegram, upbit
from backend.backtest.engine import SUPPORTED_STRATEGIES, run_backtest
from backend.config import settings
from backend.core.scheduler import scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
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
    symbol: str | None = None   # KIS 종목코드 (주식 백테스트)
    ticker: str | None = None   # 업비트 티커 (코인 백테스트)
    count: int = 120            # OHLCV 봉 수 (MA: 최소 61, RSI: 최소 15)
    initial_cash: float = 10_000_000


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
        )
        return asdict(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
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
            payload = {symbol: asdict(pos) for symbol, pos in positions.items()}

            yield {
                "event": "positions",
                "data": json.dumps(payload, ensure_ascii=False),
            }

            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())
