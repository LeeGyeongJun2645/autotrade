"""텔레그램 봇 알림 모듈.

매수·매도 체결, 오류, 서버 시작/종료 시 텔레그램 메시지 발송.
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 시 모든 함수는 no-op.
"""

import logging

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org"
_TIMEOUT = 10


def _is_configured() -> bool:
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


async def _send(text: str) -> None:
    """텔레그램 sendMessage API 호출 (내부 공통 함수).

    오류 발생 시 예외를 삼키고 로그만 남긴다.
    알림 실패가 매매 로직을 중단시켜선 안 된다.
    """
    if not _is_configured():
        return

    url = f"{_BASE}/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except Exception:
        logger.exception("텔레그램 메시지 발송 실패")


# ── 공개 알림 함수 ────────────────────────────────────────────────

async def notify_server_start(mode: str) -> None:
    """서버 시작 알림."""
    emoji = "🟡" if mode == "PAPER" else "🟢"
    await _send(f"{emoji} <b>AutoTrade 시작</b>\n모드: <code>{mode}</code>")


async def notify_server_stop() -> None:
    """서버 종료 알림."""
    await _send("🔴 <b>AutoTrade 종료</b>")


async def notify_buy(
    symbol: str,
    qty: float,
    price: float,
    strategy: str,
    reason: str,
    is_crypto: bool = False,
) -> None:
    """매수 체결 알림.

    Args:
        symbol:    종목코드 또는 마켓 코드
        qty:       체결 수량
        price:     체결 단가
        strategy:  신호 발생 전략명
        reason:    전략 판단 근거 (StrategyResult.reason)
        is_crypto: 코인 여부 (수량 포맷 변경)
    """
    qty_str = f"{qty:.8f}" if is_crypto else f"{qty:.0f}주"
    price_str = f"{price:,.0f}원"
    await _send(
        f"📈 <b>매수 체결</b>\n"
        f"종목: <code>{symbol}</code>\n"
        f"수량: {qty_str} @ {price_str}\n"
        f"전략: {strategy}\n"
        f"사유: {reason}"
    )


async def notify_sell(
    symbol: str,
    qty: float,
    price: float,
    profit_rate: float,
    reason: str,
    is_crypto: bool = False,
) -> None:
    """매도 체결 알림.

    Args:
        symbol:      종목코드 또는 마켓 코드
        qty:         체결 수량
        price:       체결 단가
        profit_rate: 수익률 (소수, 예: -0.02 = -2%)
        reason:      청산 사유 (STOP_LOSS / TAKE_PROFIT / TRAILING_STOP / 전략명)
        is_crypto:   코인 여부
    """
    qty_str = f"{qty:.8f}" if is_crypto else f"{qty:.0f}주"
    price_str = f"{price:,.0f}원"
    pnl_pct = profit_rate * 100
    emoji = "🔴" if profit_rate < 0 else "🟢"

    await _send(
        f"{emoji} <b>매도 체결</b>\n"
        f"종목: <code>{symbol}</code>\n"
        f"수량: {qty_str} @ {price_str}\n"
        f"수익률: {pnl_pct:+.2f}%\n"
        f"사유: {reason}"
    )


async def notify_error(context: str, error: Exception) -> None:
    """오류 발생 알림.

    Args:
        context: 오류가 발생한 위치/상황 설명
        error:   발생한 예외 객체
    """
    await _send(
        f"⚠️ <b>오류 발생</b>\n"
        f"위치: {context}\n"
        f"내용: {type(error).__name__}: {error}"
    )


async def notify_ml_signal(
    symbol: str,
    signal: str,
    buy_prob: float,
    news_score: float,
    prev_signal: str | None = None,
) -> None:
    """ML 신호 변화 알림."""
    _LABEL = {
        "strong_buy":  "강력매수 🟢🟢",
        "buy":         "매수 🟢",
        "hold":        "관망 ⚪",
        "sell":        "매도 🔴",
        "strong_sell": "강력매도 🔴🔴",
    }
    cur = _LABEL.get(signal, signal)
    prev_str = f"{_LABEL.get(prev_signal, prev_signal)} → " if prev_signal else ""
    news_arrow = "📈" if news_score > 0.05 else ("📉" if news_score < -0.05 else "➖")

    await _send(
        f"🤖 <b>ML 신호</b>\n"
        f"종목: <code>{symbol}</code>\n"
        f"신호: {prev_str}<b>{cur}</b>\n"
        f"매수확률: {buy_prob:.1%}\n"
        f"뉴스감성: {news_arrow} {news_score:+.3f}"
    )


async def notify_message(text: str) -> None:
    """커스텀 메시지 발송 (디버그·수동 알림용)."""
    await _send(text)
