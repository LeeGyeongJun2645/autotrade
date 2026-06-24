"""시뮬레이션 로그 인메모리 버퍼.

스케줄러 틱마다 전략 신호·체결·리스크 이벤트를 기록.
SSE /stream 을 통해 프론트엔드에 실시간 스트리밍.
"""

from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

_buffer: deque[dict] = deque(maxlen=300)


def push(symbol: str, message: str, level: str = "INFO") -> None:
    """로그 항목 추가.

    Args:
        symbol:  종목코드 또는 마켓 (예: '005930', 'KRW-BTC')
        message: 로그 메시지
        level:   'BUY' | 'SELL' | 'INFO' | 'WARN' | 'RISK'
    """
    _buffer.appendleft({
        "time": datetime.now(KST).strftime("%H:%M:%S"),
        "symbol": symbol,
        "message": message,
        "level": level,
    })


def get_logs(n: int = 100) -> list[dict]:
    """최신 n개 로그 반환."""
    return list(_buffer)[:n]


def clear() -> None:
    _buffer.clear()
