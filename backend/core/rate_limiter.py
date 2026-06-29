"""API Rate Limiter — 토큰 버킷 알고리즘으로 초당 호출 수 제한.

KIS API 공식 제한: 초당 20건.
settings.kis_rate_limit_per_sec=18 로 여유 확보해 사용.
"""

import asyncio
import time


class RateLimiter:
    """비동기 토큰 버킷 Rate Limiter.

    Args:
        max_calls: period 동안 허용되는 최대 호출 수
        period:    측정 구간 (초), 기본 1.0
    """

    def __init__(self, max_calls: int, period: float = 1.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """호출 허용 슬롯이 생길 때까지 대기 후 진입.

        Lock 외부에서 sleep: sleep 동안 다른 요청이 lock을 획득할 수 있어
        직렬 대기 없이 각자 대기 후 슬롯을 공정하게 획득.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
                if len(self._calls) < self._max_calls:
                    self._calls.append(time.monotonic())
                    return
                sleep_for = self._period - (now - self._calls[0])
            # Lock 해제 후 sleep — 다른 코루틴이 lock 획득 가능
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
