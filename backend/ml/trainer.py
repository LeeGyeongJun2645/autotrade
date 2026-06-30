"""ML 모델 자동 학습 스케줄링.

APScheduler에서 매주 월요일 07:00 KST 호출.
단일 티커 학습 또는 기본 5개 코인 전체 학습 지원.
"""

import asyncio
import logging

from backend.api import upbit
from backend.ml.model import XGBSignalModel

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]

# {ticker: {"trained_at": ..., "accuracy": ..., "roc_auc": ..., "n_samples": ...}}
_status: dict[str, dict] = {}


async def train_ticker(ticker: str, count: int = 2000) -> dict:
    """단일 티커 모델 학습."""
    try:
        ohlcv = await upbit.get_ohlcv(ticker, interval="minutes/5", count=min(count, 2000))
        if not ohlcv:
            logger.warning("[ML Trainer] %s: OHLCV 없음", ticker)
            return {"ticker": ticker, "error": "OHLCV 없음"}

        model = XGBSignalModel(ticker)
        result = await asyncio.to_thread(model.train, ohlcv)

        status = {
            "ticker":     result.ticker,
            "trained_at": result.trained_at,
            "accuracy":   result.accuracy,
            "roc_auc":    result.roc_auc,
            "n_samples":  result.n_samples,
        }
        _status[ticker] = status
        logger.info("[ML Trainer] %s 완료: %s", ticker, status)
        return status

    except Exception as e:
        logger.exception("[ML Trainer] %s 학습 실패", ticker)
        return {"ticker": ticker, "error": str(e)}


async def train_all(tickers: list[str] | None = None) -> list[dict]:
    """지정 티커 목록(또는 기본 5종) 순차 학습."""
    targets = tickers or DEFAULT_TICKERS
    results = []
    for ticker in targets:
        r = await train_ticker(ticker)
        results.append(r)
    return results


def get_status() -> dict[str, dict]:
    """마지막 학습 결과 스냅샷 반환."""
    return dict(_status)
