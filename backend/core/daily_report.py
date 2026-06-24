"""매일 08:00 KST 자동 백테스트 리포트 생성 및 텔레그램 발송.

KIS 키 미설정 시 코인(Upbit)만 리포트.
KIS 키 설정 시 주식 + 코인 종합 리포트.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.api import kis, telegram, upbit
from backend.backtest.engine import run_backtest
from backend.config import settings

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

# 기본 리포트 대상 종목 (설정 없을 시)
DEFAULT_KIS_SYMBOLS = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("035720", "카카오"),
]
DEFAULT_UPBIT_TICKERS = ["KRW-BTC", "KRW-ETH"]

STRATEGIES = ["volatility_breakout", "moving_average", "rsi"]
STRATEGY_LABELS = {
    "volatility_breakout": "변동성 돌파",
    "moving_average": "이동평균",
    "rsi": "RSI",
}


def _best_strategy(results: list[dict]) -> dict:
    """수익률 기준 최고 전략 반환."""
    return max(results, key=lambda r: r["total_return_pct"])


def _fmt_result(r: dict) -> str:
    pct = r["total_return_pct"]
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% | 승률 {r['win_rate']:.0f}% | {r['trade_count']}회"


async def _report_upbit(tickers: list[str]) -> str:
    """업비트 코인 백테스트 결과 문자열 생성."""
    lines = ["🪙 <b>Upbit 코인</b>"]
    for ticker in tickers:
        try:
            ohlcv = await upbit.get_ohlcv(ticker, interval="days", count=120)
            results = []
            strategies_to_run = ["moving_average", "rsi"]  # 코인은 VB 제외
            for strat in strategies_to_run:
                r = asdict(run_backtest(ohlcv, strat, 10_000_000, is_crypto=True))
                results.append({"strategy": strat, **r})

            best = _best_strategy(results)
            lines.append(f"\n<b>{ticker}</b>")
            for r in results:
                label = STRATEGY_LABELS[r["strategy"]]
                lines.append(f"  {label}: {_fmt_result(r)}")
            lines.append(f"  ✅ 추천: {STRATEGY_LABELS[best['strategy']]}")
        except Exception:
            logger.exception("[리포트] Upbit %s 백테스트 실패", ticker)
            lines.append(f"\n{ticker}: 데이터 조회 실패")
    return "\n".join(lines)


async def _report_kis(symbols: list[tuple[str, str]]) -> str:
    """KIS 주식 백테스트 결과 문자열 생성."""
    lines = ["🇰🇷 <b>KIS 주식</b>"]
    for code, name in symbols:
        try:
            ohlcv = await kis.get_daily_ohlcv(code, count=120)
            results = []
            for strat in STRATEGIES:
                r = asdict(run_backtest(ohlcv, strat, 10_000_000, is_crypto=False))
                results.append({"strategy": strat, **r})

            best = _best_strategy(results)
            lines.append(f"\n<b>{name} ({code})</b>")
            for r in results:
                label = STRATEGY_LABELS[r["strategy"]]
                lines.append(f"  {label}: {_fmt_result(r)}")
            lines.append(f"  ✅ 추천: {STRATEGY_LABELS[best['strategy']]}")
        except Exception:
            logger.exception("[리포트] KIS %s 백테스트 실패", code)
            lines.append(f"\n{name}: 데이터 조회 실패")
    return "\n".join(lines)


async def send_daily_report(
    kis_symbols: list[tuple[str, str]] | None = None,
    upbit_tickers: list[str] | None = None,
) -> None:
    """일일 백테스트 리포트 생성 및 텔레그램 발송.

    Args:
        kis_symbols:   (종목코드, 종목명) 튜플 리스트. None이면 기본 종목.
        upbit_tickers: 업비트 마켓 코드 리스트. None이면 기본 티커.
    """
    now = datetime.now(KST)
    date_str = now.strftime("%Y-%m-%d")
    sections: list[str] = [f"📊 <b>일일 백테스트 리포트</b> ({date_str})"]

    # KIS: 실제 키가 있을 때만
    if settings.kis_app_key and settings.kis_app_key != "test":
        symbols = kis_symbols or DEFAULT_KIS_SYMBOLS
        try:
            kis_section = await _report_kis(symbols)
            sections.append(kis_section)
        except Exception:
            logger.exception("[리포트] KIS 섹션 생성 실패")
    else:
        logger.info("[리포트] KIS 키 미설정 — 코인 리포트만 발송")

    # Upbit: 항상 실행 (공개 API)
    tickers = upbit_tickers or DEFAULT_UPBIT_TICKERS
    try:
        upbit_section = await _report_upbit(tickers)
        sections.append(upbit_section)
    except Exception:
        logger.exception("[리포트] Upbit 섹션 생성 실패")

    message = "\n\n".join(sections)
    await telegram.notify_message(message)
    logger.info("[리포트] 일일 리포트 발송 완료 (%s)", date_str)
