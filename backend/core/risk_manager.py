"""리스크 관리 모듈.

포지션 개설 시 손절가·익절가를 자동 계산하고,
매 틱마다 현재가를 검사해 청산 신호를 반환한다.

손절  : 매수가 × (1 + stop_loss_rate)    예) -2% → 매수가 × 0.98
익절  : 매수가 × (1 + take_profit_rate)   예) +4% → 매수가 × 1.04
트레일링 스탑: 고점 갱신 시 trailing_stop_price = 고점 × (1 - trailing_stop_rate)
             → 이익 구간에서 고점에서 1% 하락 시 청산
포지션 한도  : 종목당 총 자본의 max_position_ratio 이내
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal

from backend.config import settings

ExitReason = Literal["STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", ""]


@dataclass
class Position:
    """보유 포지션 정보.

    스케줄러가 인메모리로 관리하고, 체결 시 DB(trade.py)에 기록한다.

    Attributes:
        symbol:              종목코드 or 마켓 (예: '005930', 'KRW-BTC')
        entry_price:         매수 체결 평균가
        qty:                 보유 수량 (주식: 정수, 코인: 소수)
        stop_loss_price:     손절가 (매수가 × (1 + stop_loss_rate))
        take_profit_price:   익절가 (매수가 × (1 + take_profit_rate))
        highest_price:       매수 이후 기록된 최고가 (트레일링 스탑 기준)
        trailing_stop_price: 트레일링 스탑 현재 기준가 (고점 갱신 시 자동 상향)
        strategy:            신호를 발생시킨 전략명
        opened_at:           포지션 개설 시각 (ISO 8601)
        is_crypto:           코인 여부 (수량 계산 단위 구분)
    """

    symbol: str
    entry_price: float
    qty: float
    stop_loss_price: float
    take_profit_price: float
    highest_price: float
    trailing_stop_price: float
    strategy: str
    opened_at: str = field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul")).isoformat())
    is_crypto: bool = False

    @property
    def entry_value(self) -> float:
        """매수 시 투자 금액 (entry_price × qty)."""
        return self.entry_price * self.qty


@dataclass
class ExitSignal:
    """청산 판단 결과.

    Attributes:
        should_exit:   True면 청산 주문 실행
        reason:        청산 사유 ("STOP_LOSS" | "TAKE_PROFIT" | "TRAILING_STOP" | "")
        current_price: 판단 시점 현재가
        profit_rate:   현재 수익률 (소수, 예: -0.02 = -2%)
    """

    should_exit: bool
    reason: ExitReason
    current_price: float
    profit_rate: float


class RiskManager:
    """리스크 관리 정적 메서드 집합.

    인스턴스화 불필요 — 모든 메서드는 @staticmethod.
    """

    @staticmethod
    def open_position(
        symbol: str,
        entry_price: float,
        qty: float,
        strategy: str,
        is_crypto: bool = False,
        stop_loss_rate: float | None = None,
        take_profit_rate: float | None = None,
        trailing_stop_rate: float | None = None,
    ) -> Position:
        """포지션 개설 — 손절가·익절가·트레일링 스탑 초기값 계산.

        Args:
            symbol:            종목 코드 or 마켓
            entry_price:       매수 체결가
            qty:               매수 수량
            strategy:          전략명 (예: 'volatility_breakout')
            is_crypto:         코인 여부
            stop_loss_rate:    손절 비율 (음수, 기본 config 값)
            take_profit_rate:  익절 비율 (양수, 기본 config 값)
            trailing_stop_rate: 트레일링 스탑 비율 (양수, 기본 config 값)

        Returns:
            초기화된 Position 인스턴스
        """
        sl_rate = stop_loss_rate if stop_loss_rate is not None else settings.stop_loss_rate
        tp_rate = take_profit_rate if take_profit_rate is not None else settings.take_profit_rate
        ts_rate = trailing_stop_rate if trailing_stop_rate is not None else settings.trailing_stop_rate

        stop_loss_price = entry_price * (1 + sl_rate)      # sl_rate 음수이므로 자동 하락
        take_profit_price = entry_price * (1 + tp_rate)
        # 트레일링 스탑 초기값 = 진입가 기준 (이익 구간 진입 전에는 stop_loss가 우선)
        trailing_stop_price = entry_price * (1 - ts_rate)

        return Position(
            symbol=symbol,
            entry_price=entry_price,
            qty=qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            highest_price=entry_price,
            trailing_stop_price=trailing_stop_price,
            strategy=strategy,
            is_crypto=is_crypto,
        )

    @staticmethod
    def calculate_qty(
        total_cash: float,
        price: float,
        max_position_ratio: float | None = None,
        is_crypto: bool = False,
    ) -> float:
        """매수 가능 수량 계산.

        Args:
            total_cash:          주문 가능 현금
            price:               현재 매수가
            max_position_ratio:  1종목 최대 비율 (기본 config 값 20%)
            is_crypto:           True면 소수점 수량, False면 정수(주)

        Returns:
            매수 수량. 0이면 자금 부족.
        """
        if price <= 0:
            return 0.0

        ratio = max_position_ratio if max_position_ratio is not None else settings.max_position_ratio
        investment = total_cash * ratio

        if investment <= 0:
            return 0.0

        raw_qty = investment / price
        return raw_qty if is_crypto else float(math.floor(raw_qty))

    @staticmethod
    def update_trailing_stop(
        position: Position,
        current_price: float,
        trailing_stop_rate: float | None = None,
    ) -> None:
        """현재가로 트레일링 스탑 갱신 (인플레이스 수정).

        고점을 갱신했을 때만 trailing_stop_price 를 상향 조정.
        트레일링 스탑은 절대 하향하지 않는다.

        Args:
            position:           현재 포지션
            current_price:      실시간 현재가
            trailing_stop_rate: 트레일링 스탑 비율 (기본 config 값)
        """
        ts_rate = trailing_stop_rate if trailing_stop_rate is not None else settings.trailing_stop_rate

        if current_price > position.highest_price:
            position.highest_price = current_price
            new_ts = current_price * (1 - ts_rate)
            # 트레일링 스탑은 올라가기만 한다
            if new_ts > position.trailing_stop_price:
                position.trailing_stop_price = new_ts

    @staticmethod
    def check_exit(position: Position, current_price: float) -> ExitSignal:
        """현재가 기준 청산 여부 판단.

        판단 우선순위:
            1. 손절 (stop_loss_price 이하) — 무조건 최우선
            2. TP 도달 → 즉시매도 아님, 타이트 트레일링 전환 (추가 상승 포착)
            3. 트레일링 스탑 (이익 구간에서 고점 대비 1% 하락 시 청산)

        Args:
            position:      현재 포지션
            current_price: 실시간 현재가

        Returns:
            ExitSignal (should_exit, reason, current_price, profit_rate)
        """
        if position.entry_price == 0:
            return ExitSignal(should_exit=False, reason="", current_price=current_price, profit_rate=0.0)
        profit_rate = (current_price - position.entry_price) / position.entry_price

        # 1. 손절
        if current_price <= position.stop_loss_price:
            return ExitSignal(
                should_exit=True,
                reason="STOP_LOSS",
                current_price=current_price,
                profit_rate=profit_rate,
            )

        # 2. TP 도달 → 즉시매도 대신 타이트 트레일링 전환 (추가 상승 홀딩 후 하락 시 청산)
        if current_price >= position.take_profit_price:
            ts_rate = settings.trailing_stop_rate
            new_trail = current_price * (1 - ts_rate)
            if new_trail > position.trailing_stop_price:
                position.trailing_stop_price = new_trail
            position.take_profit_price = float("inf")  # 재트리거 방지

        # 3. 트레일링 스탑 — 이익 구간(trailing_stop > entry)에서 고점 대비 하락 시 청산
        if (
            position.trailing_stop_price > position.entry_price
            and current_price < position.trailing_stop_price
        ):
            return ExitSignal(
                should_exit=True,
                reason="TRAILING_STOP",
                current_price=current_price,
                profit_rate=profit_rate,
            )

        return ExitSignal(
            should_exit=False,
            reason="",
            current_price=current_price,
            profit_rate=profit_rate,
        )

    @staticmethod
    def tick(
        position: Position,
        current_price: float,
        trailing_stop_rate: float | None = None,
    ) -> ExitSignal:
        """매 틱(현재가 수신)마다 호출하는 통합 메서드.

        update_trailing_stop → check_exit 순서를 보장한다.
        스케줄러는 개별 메서드 대신 이 메서드만 사용한다.

        Args:
            position:           현재 포지션
            current_price:      실시간 현재가
            trailing_stop_rate: 트레일링 스탑 비율 (기본 config 값)

        Returns:
            ExitSignal — should_exit=True 면 즉시 청산 주문 실행
        """
        RiskManager.update_trailing_stop(position, current_price, trailing_stop_rate)
        return RiskManager.check_exit(position, current_price)

    @staticmethod
    def validate_order(
        cash: float,
        price: float,
        qty: float,
        is_crypto: bool = False,
    ) -> tuple[bool, str]:
        """주문 실행 전 최종 유효성 검사.

        Args:
            cash:      주문 가능 현금
            price:     주문가
            qty:       주문 수량
            is_crypto: 코인 여부

        Returns:
            (True, "") — 주문 가능
            (False, "사유") — 주문 불가 사유
        """
        if qty <= 0:
            return False, "주문 수량이 0 이하입니다."
        if not is_crypto and qty < 1:
            return False, "주식 최소 주문 수량은 1주입니다."
        if price <= 0:
            return False, "주문가가 0 이하입니다."
        required = price * qty
        if required > cash:
            return False, f"잔금 부족 (필요: {required:,.0f}원, 보유: {cash:,.0f}원)"
        return True, ""
