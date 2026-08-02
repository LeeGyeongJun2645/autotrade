"""펀딩레이트 차익거래 전략 (Funding Rate Arbitrage).

바이비트 선물 펀딩비가 높은 코인에 대해:
  현물 매수 + 선물 숏 → 가격 방향 리스크 없이 펀딩비만 수취

진입 조건:
  - 연환산 펀딩비 >= ARB_MIN_APR (기본 20%)
  - 과거 3회 펀딩비 평균 >= 단회 임계값 (연속성 확인)
  - 동시 포지션 수 <= MAX_POSITIONS

청산 조건:
  - 연환산 펀딩비 < ARB_EXIT_APR (기본 5%)
  - 현물+선물 합산 손익 < -ARB_STOP_LOSS (기본 -3%)
  - 보유 기간 >= ARB_MAX_HOLD_HOURS (기본 168시간 = 1주)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.api import bybit as _bybit

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

# ── 전략 파라미터 ────────────────────────────────────────────────
ARB_MIN_APR       = 0.20   # 진입: 연환산 펀딩비 최소 20%
ARB_EXIT_APR      = 0.05   # 청산: 연환산 펀딩비 5% 미만
ARB_STOP_LOSS     = 0.03   # 강제청산: 포지션 손실 -3%
ARB_MAX_HOLD_HRS  = 168    # 최대 보유시간 (7일)
MAX_POSITIONS     = 3      # 동시 최대 포지션 수
POSITION_USDT     = 50.0   # 포지션당 투자금액 (USDT)
FUNDING_PER_YEAR  = 3 * 365  # 8시간 주기 × 연365일 = 1095회

# 거래 가능한 최소 유동성 (일 거래대금 기준, USDT)
MIN_VOLUME_USDT = 5_000_000


@dataclass
class ArbPosition:
    """펀딩 차익 포지션 상태."""
    symbol:       str           # 예: "BTCUSDT"
    spot_qty:     float         # 현물 보유 수량
    fut_qty:      float         # 선물 숏 수량
    entry_price:  float         # 진입 시점 가격
    entry_time:   datetime      = field(default_factory=lambda: datetime.now(KST))
    collected_fr: float         = 0.0   # 누적 수취 펀딩비 (USDT)
    spot_avg:     float         = 0.0   # 현물 평균 매수가

    @property
    def age_hours(self) -> float:
        return (datetime.now(KST) - self.entry_time).total_seconds() / 3600

    def pnl_rate(self, current_price: float) -> float:
        """현물 가격 변동 손익률 (선물 숏이 상쇄하므로 이론상 0 근처)."""
        if self.spot_avg <= 0:
            return 0.0
        spot_pnl = (current_price / self.spot_avg - 1.0) * self.spot_qty * self.spot_avg
        fut_pnl  = (self.entry_price - current_price) * self.fut_qty
        total_cost = self.spot_qty * self.spot_avg
        return (spot_pnl + fut_pnl + self.collected_fr) / total_cost if total_cost > 0 else 0.0


class FundingArbManager:
    """펀딩레이트 차익거래 매니저."""

    def __init__(self) -> None:
        self._positions: dict[str, ArbPosition] = {}
        self._lock = asyncio.Lock()

    # ── 공개 인터페이스 ────────────────────────────────────────────

    async def scan_and_enter(self) -> None:
        """펀딩비 스캔 후 조건 충족 시 포지션 진입."""
        if not _bybit._available():
            logger.debug("[FundingArb] API 키 미설정, 스캔 스킵")
            return

        async with self._lock:
            if len(self._positions) >= MAX_POSITIONS:
                logger.debug("[FundingArb] 최대 포지션(%d) 도달, 스캔 스킵", MAX_POSITIONS)
                return

        candidates = await self._find_candidates()
        if not candidates:
            return

        async with self._lock:
            for sym, apr, price in candidates:
                if sym in self._positions:
                    continue
                if len(self._positions) >= MAX_POSITIONS:
                    break
                await self._enter(sym, apr, price)

    async def check_and_exit(self) -> None:
        """보유 포지션 청산 조건 체크."""
        if not self._positions:
            return

        all_rates = await _bybit.get_all_funding_rates()
        rate_map  = {r["symbol"]: r for r in all_rates}

        async with self._lock:
            to_exit = []
            for sym, pos in self._positions.items():
                info    = rate_map.get(sym, {})
                fr      = float(info.get("fundingRate") or 0)
                apr     = fr * FUNDING_PER_YEAR
                price   = float(info.get("lastPrice") or pos.entry_price)
                pnl     = pos.pnl_rate(price)
                reason  = None

                if apr < ARB_EXIT_APR:
                    reason = f"펀딩비 하락 (연{apr:.1%})"
                elif pnl < -ARB_STOP_LOSS:
                    reason = f"손실 ({pnl:.2%})"
                elif pos.age_hours >= ARB_MAX_HOLD_HRS:
                    reason = f"최대 보유시간 ({pos.age_hours:.0f}h)"

                if reason:
                    to_exit.append((sym, reason, pnl))

            for sym, reason, pnl in to_exit:
                await self._exit(sym, reason, pnl)

    async def update_collected_funding(self) -> None:
        """펀딩비 지급 시점(UTC 0/8/16시)에 누적 수취 펀딩비 업데이트."""
        if not self._positions:
            return
        all_rates = await _bybit.get_all_funding_rates()
        rate_map  = {r["symbol"]: r for r in all_rates}
        async with self._lock:
            for sym, pos in self._positions.items():
                fr    = float(rate_map.get(sym, {}).get("fundingRate") or 0)
                earned = fr * pos.fut_qty * pos.entry_price  # 숏 포지션 수취
                pos.collected_fr += earned
                if abs(earned) > 0.001:
                    logger.info(
                        "[FundingArb] %s 펀딩비 수취 %.4f USDT (누적 %.4f USDT, 연%.1f%%)",
                        sym, earned, pos.collected_fr, fr * FUNDING_PER_YEAR * 100,
                    )

    def status(self) -> list[dict]:
        """현재 포지션 상태 딕셔너리 리스트."""
        return [
            {
                "symbol":       sym,
                "spot_qty":     pos.spot_qty,
                "fut_qty":      pos.fut_qty,
                "entry_price":  pos.entry_price,
                "entry_time":   pos.entry_time.isoformat(),
                "age_hours":    round(pos.age_hours, 1),
                "collected_fr": round(pos.collected_fr, 4),
            }
            for sym, pos in self._positions.items()
        ]

    # ── 내부 로직 ──────────────────────────────────────────────────

    async def _find_candidates(self) -> list[tuple[str, float, float]]:
        """연환산 펀딩비가 높은 심볼 반환: [(symbol, apr, price), ...]"""
        all_rates = await _bybit.get_all_funding_rates()
        if not all_rates:
            return []

        candidates = []
        for item in all_rates:
            sym   = item["symbol"]
            fr    = item["fundingRate"]
            price = item["lastPrice"]
            if price <= 0:
                continue
            apr = fr * FUNDING_PER_YEAR

            if apr < ARB_MIN_APR:
                continue
            if sym in self._positions:
                continue

            # 과거 3회 펀딩비 평균 확인 (연속성)
            history = await _bybit.get_funding_rate_history(sym, limit=3)
            if len(history) < 2:
                continue
            avg_fr = sum(float(h.get("fundingRate") or 0) for h in history) / len(history)
            if avg_fr * FUNDING_PER_YEAR < ARB_MIN_APR * 0.7:  # 과거 평균이 임계값 70% 미만이면 제외
                continue

            candidates.append((sym, apr, price))

        candidates.sort(key=lambda x: -x[1])  # APR 내림차순
        return candidates[:MAX_POSITIONS]

    async def _enter(self, symbol: str, apr: float, price: float) -> None:
        """현물 매수 + 선물 숏 동시 진입."""
        qty = round(POSITION_USDT / price, 4)
        if qty <= 0:
            return

        logger.warning(
            "[FundingArb] 진입 시도: %s 연%.1f%% | 현물매수+선물숏 %.4f개 (%.0f USDT)",
            symbol, apr * 100, qty, POSITION_USDT,
        )

        # 양방향 모드 확인 (선물 Hedge Mode)
        await _bybit.set_position_mode(symbol)

        # 현물 매수
        spot_result = await _bybit.spot_market_buy(symbol, qty)
        if spot_result is None:
            logger.error("[FundingArb] %s 현물 매수 실패, 진입 중단", symbol)
            return

        # 선물 숏
        fut_result = await _bybit.futures_open_short(symbol, qty)
        if fut_result is None:
            logger.error("[FundingArb] %s 선물 숏 실패, 현물 매도로 롤백", symbol)
            await _bybit.spot_market_sell(symbol, qty)
            return

        self._positions[symbol] = ArbPosition(
            symbol=symbol,
            spot_qty=qty,
            fut_qty=qty,
            entry_price=price,
            spot_avg=price,
        )
        expected_8h = qty * price * apr / FUNDING_PER_YEAR
        logger.warning(
            "[FundingArb] 진입 완료: %s %.4f개 | 연%.1f%% | 예상 8h수익 %.4f USDT",
            symbol, qty, apr * 100, expected_8h,
        )

    async def _exit(self, symbol: str, reason: str, pnl: float) -> None:
        """현물 매도 + 선물 숏 청산."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        logger.warning(
            "[FundingArb] 청산 시도: %s (%s) PnL=%.2f%%",
            symbol, reason, pnl * 100,
        )

        # 선물 청산 먼저 (리스크 제거 우선)
        fut_ok = await _bybit.futures_close_short(symbol, pos.fut_qty)
        spot_ok = await _bybit.spot_market_sell(symbol, pos.spot_qty)

        if fut_ok and spot_ok:
            total_earned = pos.collected_fr
            del self._positions[symbol]
            logger.warning(
                "[FundingArb] 청산 완료: %s 누적펀딩수익 %.4f USDT (보유 %.1fh)",
                symbol, total_earned, pos.age_hours,
            )
        else:
            logger.error("[FundingArb] %s 청산 실패 (fut=%s, spot=%s)", symbol, fut_ok, spot_ok)


# ── 싱글톤 ────────────────────────────────────────────────────────
funding_arb = FundingArbManager()
