"""20개 AI 에이전트 경쟁 시뮬레이션 모듈.

각 에이전트는 서로 다른 (인터벌, 레이블 기준, 매수 임계값, 피처 세트) 조합으로
독립적인 가상 포트폴리오를 운영한다.

승률이 가장 높은 에이전트가 챔피언으로 선정되며,
해당 전략 결과를 실제 ML 모델 참고에 활용한다.
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.features import FEATURE_NAMES, compute_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL    = 10_000_000.0  # 에이전트당 초기 가상 자금 (1000만원)
POSITION_RATIO     = 0.5            # 잔액의 50%씩 사용
MAX_OPEN_POSITIONS = 3              # 에이전트당 동시 최대 포지션 수

# ── 피처 세트 정의 ────────────────────────────────────────────────

FEATURE_SETS: dict[str, list[str]] = {
    "all": FEATURE_NAMES,
    "momentum": [
        "rsi_9", "macd_diff", "stoch_rsi", "williams_r",
        "mfi_14", "roc_10", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        # 래그 피처 (단기 모멘텀 전환 포착)
        "rsi_lag_1", "rsi_lag_2", "macd_diff_lag_1",
        "ret_lag_1", "ret_lag_2", "ret_lag_3",
    ],
    "trend": [
        "ma5_ratio", "ma20_ratio", "ma60_ratio",
        "ma5_cross_ma20", "ma20_cross_ma60",
        "adx_14", "adx_pos", "adx_neg",
        "trix_15", "dpo_20", "vortex_diff",
        "ema9_cross_ema21", "vwap_ratio", "vwap_cross",
        "bb_pband_lag_1",  # 볼린저밴드 직전 위치
    ],
    "volume": [
        "vol_ratio", "obv_change", "cmf_20",
        "mfi_14", "ret_1d", "ret_5d", "ret_20d",
        "vol_ratio_lag_1",  # 직전 거래량 비율
    ],
}

# ── 20개 에이전트 설정 (전체 5분봉, 중복 전략 없음) ─────────────
# (agent_id, interval_min, label_threshold, buy_threshold, feature_set, market, lookahead)
# lookahead: 3=단기(15분), 5=중기(25분), 8=장기(40분) — 앙상블 분산 극대화

AGENT_CONFIGS: list[tuple] = [
    # ── 코인 AI01~AI10 ── lookahead 3/5/8 순환으로 시간 지평 다양화
    ("AI01",  5, 0.001, 0.55, "all",      "coin",  3),  # 단기 공격형
    ("AI02",  5, 0.001, 0.62, "momentum", "coin",  5),
    ("AI03",  5, 0.002, 0.55, "trend",    "coin",  8),  # 장기 추세형
    ("AI04",  5, 0.002, 0.60, "volume",   "coin",  3),
    ("AI05",  5, 0.003, 0.58, "all",      "coin",  5),
    ("AI06",  5, 0.003, 0.65, "momentum", "coin",  8),
    ("AI07",  5, 0.004, 0.60, "trend",    "coin",  3),
    ("AI08",  5, 0.002, 0.60, "volume",   "coin",  5),
    ("AI09",  5, 0.005, 0.62, "all",      "coin",  8),
    ("AI10",  5, 0.002, 0.62, "momentum", "coin",  3),
    # ── 주식 AI11~AI20 ── 동일 구조
    ("AI11",  5, 0.001, 0.55, "all",      "stock", 3),
    ("AI12",  5, 0.001, 0.62, "trend",    "stock", 5),
    ("AI13",  5, 0.002, 0.55, "momentum", "stock", 8),
    ("AI14",  5, 0.002, 0.60, "volume",   "stock", 3),
    ("AI15",  5, 0.003, 0.58, "all",      "stock", 5),
    ("AI16",  5, 0.003, 0.65, "trend",    "stock", 8),
    ("AI17",  5, 0.004, 0.60, "momentum", "stock", 3),
    ("AI18",  5, 0.004, 0.68, "volume",   "stock", 5),
    ("AI19",  5, 0.005, 0.62, "all",      "stock", 8),
    ("AI20",  5, 0.005, 0.70, "trend",    "stock", 3),
]



@dataclass
class AgentPosition:
    ticker: str
    entry_price: float
    qty: float
    entered_at: str


@dataclass
class AgentTrade:
    agent_id: str
    ticker: str
    action: str          # "BUY" | "SELL"
    price: float
    qty: float
    entry_price: float | None
    profit_rate: float | None
    balance: float
    traded_at: str


class SimAgent:
    """단일 AI 에이전트 — 독립 가상 포트폴리오 + 전용 ML 모델."""

    def __init__(
        self,
        agent_id: str,
        interval_min: int,
        label_threshold: float,
        buy_threshold: float,
        feature_set: str,
        market: str = "coin",  # "coin" | "stock"
        lookahead: int = 5,    # 레이블링 시 몇 봉 앞 수익 기준 (3=15분/5=25분/8=40분)
    ) -> None:
        self.agent_id = agent_id
        self.interval_min = interval_min
        self.label_threshold = label_threshold
        self.buy_threshold = buy_threshold
        self.feature_set = feature_set
        self.market = market
        self.lookahead = lookahead
        self.feature_names = FEATURE_SETS[feature_set]

        self._balance = INITIAL_CAPITAL
        self._positions: dict[str, AgentPosition] = {}
        self._model: XGBClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._trained_at: str | None = None
        self._model_path = MODEL_DIR / f"xgb_agent_{agent_id}.pkl"

        self.total_trades = 0
        self.win_trades = 0
        self.is_champion = False
        self.is_active = True  # False면 매수 스킵 (자동 임계값 조정에서 관리)
        self.recent_trades: list[dict] = []
        self._last_position_values: dict[str, float] = {}  # ticker → 현재 평가액
        self._last_atr_pct: float = 0.0  # ATR 기반 동적 손익용 (predict()에서 업데이트)

    # ── 프로퍼티 ────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        return self.win_trades / self.total_trades if self.total_trades > 0 else 0.0

    def update_position_values(self, prices: dict[str, float]) -> None:
        """매 틱마다 포지션 평가액 갱신 — 가격 없으면 마지막 알려진 값 유지."""
        for ticker, pos in self._positions.items():
            if ticker in prices:
                self._last_position_values[ticker] = pos.qty * prices[ticker]
            elif ticker not in self._last_position_values:
                self._last_position_values[ticker] = pos.qty * pos.entry_price
        # 청산된 포지션 제거
        self._last_position_values = {
            t: v for t, v in self._last_position_values.items()
            if t in self._positions
        }

    @property
    def position_value(self) -> float:
        """현재 포지션 총 평가액. 가격 데이터 없으면 매수가 기준 반환 (0원 버그 방지)."""
        total = 0.0
        for ticker, pos in self._positions.items():
            total += self._last_position_values.get(ticker, pos.entry_price * pos.qty)
        return total

    @property
    def total_return(self) -> float:
        """현재 잔액 + 포지션 평가액 기준 총 수익률."""
        return (self._balance + self.position_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    @property
    def _kelly_ratio(self) -> float:
        """Half-Kelly 포지션 사이징. 승률 낮을 때 자동으로 베팅 크기 줄임."""
        if self.total_trades < 10:
            return POSITION_RATIO
        p = max(0.3, min(0.8, self.win_rate))
        b = 1.5   # 평균 이익 / 평균 손실 추정 (보수적)
        kelly = (p * b - (1 - p)) / b
        return max(0.1, min(POSITION_RATIO, kelly * 0.5))  # Half-Kelly, 10~50% 범위

    def _sharpe_weight(self) -> float:
        """최근 거래 수익률 기반 샤프 비율 가중치.

        꾸준히 수익 내는 에이전트를 우대. 운으로 한 번 크게 번 에이전트보다
        안정적으로 조금씩 버는 에이전트가 앙상블에서 더 큰 발언권을 가짐.
        거래 5개 미만이면 승률로 폴백.
        """
        sell_trades = [
            t["profit_rate"] for t in self.recent_trades
            if t.get("action") == "SELL" and t.get("profit_rate") is not None
        ]
        if len(sell_trades) < 5:
            return max(self.win_rate, 0.1)
        mean_r = float(np.mean(sell_trades))
        std_r  = float(np.std(sell_trades))
        if std_r < 1e-9:
            return max(mean_r * 10 + 0.5, 0.1)
        sharpe = mean_r / std_r
        return max(sharpe + 0.5, 0.1)  # 음수 방지 (최소 0.1 보장)

    @property
    def interval_str(self) -> str:
        return f"minutes/{self.interval_min}"

    # ── 모델 저장/로드 ───────────────────────────────────────────

    def load_model(self) -> bool:
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            stored_feats = data.get("feature_names", [])
            if stored_feats and stored_feats != self.feature_names:
                return False  # 피처 불일치 → 재학습
            self._model = data["model"]
            self._scaler = data["scaler"]
            self._trained_at = data.get("trained_at")
            return True
        except Exception:
            self._model_path.unlink(missing_ok=True)  # 손상 파일 삭제 → 다음 틱 재학습
            return False

    def save_model(self) -> None:
        with open(self._model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "trained_at": self._trained_at,
                    "feature_names": self.feature_names,
                },
                f,
            )

    # ── 학습 (동기 — asyncio.to_thread 로 호출) ─────────────────

    def train(self, ohlcv_list: list[dict], funding_rates: list[dict] | None = None) -> bool:
        """분봉 OHLCV로 전용 모델 학습."""
        try:
            feat_df = compute_features(ohlcv_list, funding_rates=funding_rates)
            feat_df = feat_df[[c for c in self.feature_names if c in feat_df.columns]].dropna()
            if len(feat_df) < 50:
                return False

            # feat_df와 동일한 인덱스 기반으로 close 추출 (인덱스 불일치 방지)
            feat_df = feat_df.reset_index(drop=True)
            close_vals = [float(c["close"]) for c in reversed(ohlcv_list)]
            close = pd.Series(close_vals)
            # feat_df 행 수에 맞게 close 뒤에서 자름
            n = len(feat_df)
            close = close.iloc[-n:].reset_index(drop=True)

            # ── 노이즈 제거 레이블링 ─────────────────────────────────
            LOOKAHEAD   = self.lookahead  # 에이전트별 시간 지평 (3/5/8봉)
            NOISE_FLOOR = 0.001           # ±0.1% 이내는 잡음 → 학습 제외

            next_close = close.shift(-LOOKAHEAD)
            ret   = next_close / close - 1
            label = (ret >= self.label_threshold).astype(int)

            # 마지막 LOOKAHEAD 행 제거 (next_close NaN)
            feat_df = feat_df.iloc[:-LOOKAHEAD]
            label   = label.iloc[:-LOOKAHEAD]
            ret     = ret.iloc[:-LOOKAHEAD]

            # 길이 보정
            min_len = min(len(feat_df), len(label))
            feat_df = feat_df.iloc[:min_len]
            label   = label.iloc[:min_len]
            ret     = ret.iloc[:min_len]

            # 모호한 레이블 제거 (±NOISE_FLOOR 이내 수익은 잡음)
            clear_mask = ret.abs() >= NOISE_FLOOR
            feat_df = feat_df[clear_mask.values]
            label   = label[clear_mask.values]

            if len(feat_df) < 30 or len(set(label.values)) < 2:
                return False

            X = feat_df.values
            y = label.values.astype(int)

            # 최근 500샘플 2배 가중치 (최신 시장 환경 우선 반영)
            weights = np.ones(len(X))
            if len(X) > 500:
                weights[-500:] = 2.0

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # 클래스 불균형 자동 보정 — buy 레이블이 적은 하락장에서도 공정한 학습
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos,  # buy/sell 불균형 자동 보정
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_s, y, sample_weight=weights, verbose=False)

            self._model = clf
            self._scaler = scaler
            self._trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_model()
            logger.info("[%s] 모델 학습 완료 (%d샘플)", self.agent_id, len(X))
            return True
        except Exception as e:
            logger.warning("[%s] 학습 실패: %s", self.agent_id, e)
            return False

    # ── 예측 ────────────────────────────────────────────────────

    def predict(self, ohlcv_list: list[dict]) -> tuple[str, float]:
        """(signal, buy_prob) 반환. 모델 없으면 ('hold', 0.5)."""
        if self._model is None and not self.load_model():
            return "hold", 0.5
        try:
            full_df = compute_features(ohlcv_list)  # 전체 피처
            if full_df.empty:
                return "hold", 0.5
            # ATR 캐싱 — _agent_execute에서 동적 손익 계산에 사용
            if "atr_pct" in full_df.columns:
                self._last_atr_pct = float(full_df["atr_pct"].iloc[-1])
            # ADX 레짐 필터 — 횡보장(ADX<20)은 예측 신뢰도 낮으므로 스킵
            if full_df["adx_14"].iloc[-1] < 20:
                return "hold", 0.5
            feat_df = full_df[[c for c in self.feature_names if c in full_df.columns]].dropna()
            if feat_df.empty:
                return "hold", 0.5
            X_last = feat_df.iloc[[-1]].values
            X_scaled = self._scaler.transform(X_last)  # type: ignore[union-attr]
            prob = float(self._model.predict_proba(X_scaled)[0, 1])  # type: ignore[union-attr]
        except Exception:
            return "hold", 0.5

        if prob >= self.buy_threshold:
            return "buy", round(prob, 4)
        if prob <= (1.0 - self.buy_threshold):
            return "sell", round(prob, 4)
        return "hold", round(prob, 4)

    async def predict_live(self, ohlcv_list: list[dict], ticker: str | None = None) -> tuple[str, float]:
        """실시간 보정 포함 예측 — 실매매 게이트 전용.

        기본 predict() 확률에 공포탐욕·김치프리미엄·펀딩비·호가창·기관외국인을 실시간 반영.
        """
        _, prob = self.predict(ohlcv_list)

        if self.market == "coin" and ticker:
            # 공포탐욕 지수 (탐욕→과열 하향, 공포→저평가 상향)
            try:
                from backend.ml.model import _get_fear_greed
                fg = await _get_fear_greed()
                prob = max(0.01, min(0.99, prob - fg * 0.04))
            except Exception:
                pass

            # 김치프리미엄 + 펀딩비
            try:
                coin_sym = ticker.replace("KRW-", "") + "USDT"
                current_price = float(ohlcv_list[0]["close"])
                from backend.api.binance import get_funding_rate, get_kimchi_premium
                kimchi, funding = await asyncio.gather(
                    get_kimchi_premium(current_price, coin_sym),
                    get_funding_rate(coin_sym),
                )
                if kimchi > 3.0:
                    prob = max(0.01, prob - min((kimchi - 3.0) * 0.01, 0.05))
                elif kimchi < -1.0:
                    prob = min(0.99, prob + min((-kimchi - 1.0) * 0.01, 0.03))
                if abs(funding) > 0.001:
                    fund_adj = min(abs(funding) * 20, 0.04) * (1 if funding > 0 else -1)
                    prob = max(0.01, min(0.99, prob - fund_adj))
            except Exception:
                pass

            # 업비트 호가창 buy_pressure
            try:
                from backend.api.upbit import get_orderbook
                ob = await get_orderbook(ticker)
                prob = max(0.01, min(0.99, prob + (ob.get("buy_pressure", 0.5) - 0.5) * 0.08))
            except Exception:
                pass

        elif self.market == "stock" and ticker:
            # KIS 호가창 + 기관/외국인 순매수
            try:
                from backend.api.kis import get_investor_trend, get_orderbook_kis
                ob, inv = await asyncio.gather(
                    get_orderbook_kis(ticker),
                    get_investor_trend(ticker),
                )
                prob = max(0.01, min(0.99, prob + (ob.get("buy_pressure", 0.5) - 0.5) * 0.06))
                net = inv.get("institution_net_buy", 0) + inv.get("foreign_net_buy", 0)
                if net > 0:
                    prob = min(0.99, prob + 0.02)
                elif net < 0:
                    prob = max(0.01, prob - 0.02)
            except Exception:
                pass

        if prob >= self.buy_threshold:
            return "buy", round(prob, 4)
        if prob <= (1.0 - self.buy_threshold):
            return "sell", round(prob, 4)
        return "hold", round(prob, 4)

    # ── 가상 매수/매도 ───────────────────────────────────────────

    def virtual_buy(self, ticker: str, price: float) -> AgentTrade | None:
        if not self.is_active:
            return None  # 비활성화된 에이전트는 매수 스킵
        if ticker in self._positions:
            return None
        if len(self._positions) >= MAX_OPEN_POSITIONS:
            return None  # 동시 포지션 최대 3개 초과 차단
        amount = self._balance * self._kelly_ratio  # Half-Kelly 자동 사이징
        if amount < 5_000:
            return None
        actual_price = price * 1.0005  # 슬리피지 0.05% 반영 (실제 체결가 불리 보정)
        qty = amount / actual_price
        price = actual_price  # 이하 price 변수를 실제 체결가로 통일
        self._balance -= amount
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._positions[ticker] = AgentPosition(ticker, price, qty, now)
        trade = AgentTrade(self.agent_id, ticker, "BUY", price, qty, None, None, self._balance, now)
        self._push_recent(trade)
        return trade

    def virtual_sell(self, ticker: str, price: float) -> AgentTrade | None:
        pos = self._positions.pop(ticker, None)
        if pos is None:
            return None
        proceeds = pos.qty * price
        profit_rate = (price - pos.entry_price) / pos.entry_price
        self._balance += proceeds
        self.total_trades += 1
        if profit_rate > 0:
            self.win_trades += 1
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        trade = AgentTrade(self.agent_id, ticker, "SELL", price, pos.qty, pos.entry_price, round(profit_rate, 4), self._balance, now)
        self._push_recent(trade)
        return trade

    def _push_recent(self, trade: AgentTrade) -> None:
        self.recent_trades = ([self._trade_to_dict(trade)] + self.recent_trades)[:30]

    @staticmethod
    def _trade_to_dict(t: AgentTrade) -> dict:
        return {
            "agent_id":   t.agent_id,
            "ticker":     t.ticker,
            "action":     t.action,
            "price":      t.price,
            "qty":        t.qty,
            "entry_price": t.entry_price,
            "profit_rate": t.profit_rate,
            "balance":    t.balance,
            "traded_at":  t.traded_at,
        }

    # ── DB 복구 ─────────────────────────────────────────────────

    def restore_from_db(self, stats: dict, positions: list[dict], trades: list[dict]) -> None:
        """서버 재시작 시 DB에서 상태 복구."""
        self._balance      = stats.get("current_balance", INITIAL_CAPITAL)
        self.total_trades  = stats.get("total_trades", 0)
        self.win_trades    = stats.get("win_trades", 0)
        self.is_champion   = bool(stats.get("is_champion", 0))
        self.is_active     = bool(stats.get("is_active", 1))
        # 자동 조정된 임계값 복구 (기본값은 AGENT_CONFIGS 기준)
        if "buy_threshold" in stats and stats["buy_threshold"]:
            self.buy_threshold = float(stats["buy_threshold"])
        for p in positions:
            pos = AgentPosition(
                ticker=p["ticker"],
                entry_price=p["entry_price"],
                qty=p["qty"],
                entered_at=p["entered_at"],
            )
            self._positions[p["ticker"]] = pos
            # 재시작 시 매수가 기준으로 초기화 → 다음 틱에 실시간 가격으로 갱신
            self._last_position_values[p["ticker"]] = pos.entry_price * pos.qty
        self.recent_trades = trades[:30]

    # ── 직렬화 ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        pos_value = self.position_value
        total_value = self._balance + pos_value
        positions_detail = {}
        for ticker, pos in self._positions.items():
            cost = pos.entry_price * pos.qty
            cur_val = self._last_position_values.get(ticker, cost)
            positions_detail[ticker] = {
                "entry_price":         pos.entry_price,
                "qty":                 pos.qty,
                "entered_at":          pos.entered_at,
                "current_value":       round(cur_val, 0),
                "unrealized_pnl_pct":  round((cur_val / cost - 1) * 100, 2) if cost > 0 else 0.0,
            }
        return {
            "agent_id":         self.agent_id,
            "market":           self.market,
            "interval_min":     self.interval_min,
            "label_threshold":  self.label_threshold,
            "buy_threshold":    self.buy_threshold,
            "feature_set":      self.feature_set,
            "balance":          round(self._balance, 0),
            "position_value":   round(pos_value, 0),
            "total_value":      round(total_value, 0),
            "total_return_pct": round(self.total_return * 100, 2),
            "win_rate":         round(self.win_rate * 100, 1),
            "total_trades":     self.total_trades,
            "win_trades":       self.win_trades,
            "is_champion":      self.is_champion,
            "is_active":        self.is_active,
            "trained_at":       self._trained_at,
            "positions":        positions_detail,
            "recent_trades":    self.recent_trades[:20],
        }


# ── 싱글톤 에이전트 풀 ────────────────────────────────────────────

def build_agents() -> dict[str, SimAgent]:
    return {cfg[0]: SimAgent(*cfg) for cfg in AGENT_CONFIGS}


AGENTS: dict[str, SimAgent] = build_agents()


def get_champion(market: str | None = None) -> SimAgent | None:
    """마켓별 총 자산(잔액+포지션) 1위 에이전트 반환 (최소 10거래)."""
    candidates = [a for a in AGENTS.values() if a.total_trades >= 10]
    if market:
        candidates = [a for a in candidates if a.market == market]
    return max(candidates, key=lambda a: a.total_return) if candidates else None


async def predict_ensemble(
    ohlcv_list: list[dict],
    ticker: str,
    market: str,
) -> tuple[str, float]:
    """가중 앙상블 게이트 — 전 에이전트 동적 투표 + 실시간 보정.

    가중치 = 승률 × (1 + 총수익률).
    지금 잘하는 전략이 자동으로 발언권이 커져서 장세에 자동 적응.
    """
    candidates = [
        a for a in AGENTS.values()
        if a.market == market and a._model is not None
    ]
    if not candidates:
        return "hold", 0.5

    # ── 가중 예측 집계 (샤프 비율 × 총수익률) ─────────────────────
    weighted_prob = 0.0
    total_weight  = 0.0
    for agent in candidates:
        ret    = max(agent.total_return, -0.5)      # 큰 손실 에이전트 하한 제한
        sharpe = agent._sharpe_weight()              # 꾸준한 수익 에이전트 우대
        weight = max(sharpe * max(1.0 + ret, 0.1), 0.1)  # 최소 발언권 0.1 보장
        _, prob = agent.predict(ohlcv_list)
        weighted_prob += prob * weight
        total_weight  += weight

    final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5

    # ── 실시간 보정 (시장 데이터 1회씩만 조회) ──────────────────
    if market == "coin" and ticker:
        try:
            from backend.ml.model import _get_fear_greed
            fg = await _get_fear_greed()
            final_prob = max(0.01, min(0.99, final_prob - fg * 0.04))
        except Exception:
            pass
        try:
            coin_sym = ticker.replace("KRW-", "") + "USDT"
            current_price = float(ohlcv_list[0]["close"])
            from backend.api.binance import get_funding_rate, get_kimchi_premium
            kimchi, funding = await asyncio.gather(
                get_kimchi_premium(current_price, coin_sym),
                get_funding_rate(coin_sym),
            )
            if kimchi > 3.0:
                final_prob = max(0.01, final_prob - min((kimchi - 3.0) * 0.01, 0.05))
            elif kimchi < -1.0:
                final_prob = min(0.99, final_prob + min((-kimchi - 1.0) * 0.01, 0.03))
            if abs(funding) > 0.001:
                fund_adj = min(abs(funding) * 20, 0.04) * (1 if funding > 0 else -1)
                final_prob = max(0.01, min(0.99, final_prob - fund_adj))
        except Exception:
            pass
        try:
            from backend.api.upbit import get_orderbook
            ob = await get_orderbook(ticker)
            final_prob = max(0.01, min(0.99, final_prob + (ob.get("buy_pressure", 0.5) - 0.5) * 0.08))
        except Exception:
            pass

    elif market == "stock" and ticker:
        try:
            from backend.api.kis import get_investor_trend, get_orderbook_kis
            ob, inv = await asyncio.gather(
                get_orderbook_kis(ticker),
                get_investor_trend(ticker),
            )
            final_prob = max(0.01, min(0.99, final_prob + (ob.get("buy_pressure", 0.5) - 0.5) * 0.06))
            net = inv.get("institution_net_buy", 0) + inv.get("foreign_net_buy", 0)
            if net > 0:
                final_prob = min(0.99, final_prob + 0.02)
            elif net < 0:
                final_prob = max(0.01, final_prob - 0.02)
        except Exception:
            pass

    # ── 임계값: 학습된 에이전트 buy_threshold 가중 평균 ─────────
    avg_thr = sum(a.buy_threshold for a in candidates) / len(candidates)

    if final_prob >= avg_thr:
        return "buy", round(final_prob, 4)
    if final_prob <= (1.0 - avg_thr):
        return "sell", round(final_prob, 4)
    return "hold", round(final_prob, 4)


def refresh_champion_flags() -> None:
    """마켓별 총자산 1위 에이전트에 is_champion=True 표시 (UI 전용, 실시간 갱신).

    게이트는 앙상블 투표 방식이므로 챔피언 플래그는 순위 표시 목적만 가짐.
    _agent_tick 끝에서 5분마다 호출.
    """
    for market in ("coin", "stock"):
        market_agents = [a for a in AGENTS.values() if a.market == market]
        for a in market_agents:
            a.is_champion = False
        active = [a for a in market_agents if a.total_trades > 0]
        if active:
            max(active, key=lambda a: a.total_return).is_champion = True
