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

INITIAL_CAPITAL = 1_000_000.0   # 에이전트당 초기 가상 자금 (100만원)
POSITION_RATIO  = 0.5            # 잔액의 50%씩 사용

# ── 피처 세트 정의 ────────────────────────────────────────────────

FEATURE_SETS: dict[str, list[str]] = {
    "all": FEATURE_NAMES,
    "momentum": [
        "rsi_14", "macd_diff", "stoch_rsi", "williams_r",
        "mfi_14", "roc_10", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    ],
    "trend": [
        "ma5_ratio", "ma20_ratio", "ma60_ratio",
        "ma5_cross_ma20", "ma20_cross_ma60",
        "adx_14", "adx_pos", "adx_neg",
        "trix_15", "dpo_20", "vortex_diff", "cci_20",
    ],
    "volume": [
        "vol_ratio", "obv_change", "cmf_20",
        "mfi_14", "ret_1d", "ret_5d", "ret_20d",
    ],
}

# ── 20개 에이전트 설정 ───────────────────────────────────────────
# (agent_id, interval_min, label_threshold, buy_threshold, feature_set, market)
# market: "coin" → 업비트 24/7 | "stock" → KIS 장중만

AGENT_CONFIGS: list[tuple[str, int, float, float, str, str]] = [
    # ── 코인 전담 AI01~AI10 (업비트 거래대금 상위 50개, 24/7) ──
    ("AI01",  1, 0.001, 0.55, "all",      "coin"),
    ("AI02",  1, 0.002, 0.60, "momentum", "coin"),
    ("AI03",  1, 0.003, 0.65, "volume",   "coin"),
    ("AI04",  1, 0.005, 0.55, "trend",    "coin"),
    ("AI05",  5, 0.001, 0.60, "all",      "coin"),
    ("AI06",  5, 0.002, 0.55, "momentum", "coin"),
    ("AI07",  5, 0.003, 0.65, "volume",   "coin"),
    ("AI08",  5, 0.005, 0.60, "trend",    "coin"),
    ("AI09",  5, 0.001, 0.65, "all",      "coin"),
    ("AI10", 15, 0.002, 0.55, "all",      "coin"),
    # ── 주식 전담 AI11~AI20 (KIS 거래량 상위 50개, 장중만) ──────
    ("AI11",  1, 0.001, 0.55, "all",      "stock"),
    ("AI12",  1, 0.002, 0.60, "momentum", "stock"),
    ("AI13",  5, 0.001, 0.55, "all",      "stock"),
    ("AI14",  5, 0.002, 0.60, "volume",   "stock"),
    ("AI15",  5, 0.003, 0.65, "trend",    "stock"),
    ("AI16",  5, 0.005, 0.55, "all",      "stock"),
    ("AI17", 15, 0.001, 0.60, "all",      "stock"),
    ("AI18", 15, 0.002, 0.55, "momentum", "stock"),
    ("AI19", 15, 0.003, 0.65, "volume",   "stock"),
    ("AI20", 15, 0.005, 0.55, "all",      "stock"),
]

# 에이전트가 감시하는 기본 코인 티커
DEFAULT_TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]


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
    ) -> None:
        self.agent_id = agent_id
        self.interval_min = interval_min
        self.label_threshold = label_threshold
        self.buy_threshold = buy_threshold
        self.feature_set = feature_set
        self.market = market  # 코인 전담 or 주식 전담
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
        self.recent_trades: list[dict] = []

    # ── 프로퍼티 ────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        return self.win_trades / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def total_return(self) -> float:
        """현재 잔액 기준 총 수익률 (포지션 미실현 손익 제외)."""
        return (self._balance - INITIAL_CAPITAL) / INITIAL_CAPITAL

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

    def train(self, ohlcv_list: list[dict]) -> bool:
        """분봉 OHLCV로 전용 모델 학습."""
        try:
            feat_df = compute_features(ohlcv_list)
            feat_df = feat_df[[c for c in self.feature_names if c in feat_df.columns]].dropna()
            if len(feat_df) < 50:
                return False

            df_raw = pd.DataFrame(list(reversed(ohlcv_list)))
            df_raw["date"] = pd.to_datetime(df_raw["date"].astype(str).str[:19])
            df_raw = df_raw.set_index("date").sort_index()
            close = df_raw["close"].astype(float)

            next_close = close.shift(-1)
            label = ((next_close / close - 1) >= self.label_threshold).astype(int)
            label = label.reindex(feat_df.index).dropna()
            feat_df = feat_df.loc[label.index].iloc[:-1]
            label = label.iloc[:-1]

            if len(feat_df) < 30 or len(set(label.values)) < 2:
                return False

            X = feat_df.values
            y = label.values.astype(int)

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            clf = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_s, y, verbose=False)

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
            feat_df = compute_features(ohlcv_list)
            feat_df = feat_df[[c for c in self.feature_names if c in feat_df.columns]].dropna()
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

    # ── 가상 매수/매도 ───────────────────────────────────────────

    def virtual_buy(self, ticker: str, price: float) -> AgentTrade | None:
        if ticker in self._positions:
            return None
        amount = self._balance * POSITION_RATIO
        if amount < 5_000:
            return None
        qty = amount / price
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
        for p in positions:
            self._positions[p["ticker"]] = AgentPosition(
                ticker=p["ticker"],
                entry_price=p["entry_price"],
                qty=p["qty"],
                entered_at=p["entered_at"],
            )
        self.recent_trades = trades[:30]

    # ── 직렬화 ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "agent_id":        self.agent_id,
            "market":          self.market,
            "interval_min":    self.interval_min,
            "label_threshold": self.label_threshold,
            "buy_threshold":   self.buy_threshold,
            "feature_set":     self.feature_set,
            "balance":         round(self._balance, 0),
            "total_return_pct": round(self.total_return * 100, 2),
            "win_rate":        round(self.win_rate * 100, 1),
            "total_trades":    self.total_trades,
            "win_trades":      self.win_trades,
            "is_champion":     self.is_champion,
            "trained_at":      self._trained_at,
            "positions": {
                ticker: {
                    "entry_price": pos.entry_price,
                    "qty":         pos.qty,
                    "entered_at":  pos.entered_at,
                }
                for ticker, pos in self._positions.items()
            },
            "recent_trades": self.recent_trades[:10],
        }


# ── 싱글톤 에이전트 풀 ────────────────────────────────────────────

def build_agents() -> dict[str, SimAgent]:
    return {
        cfg[0]: SimAgent(*cfg)
        for cfg in AGENT_CONFIGS
    }


AGENTS: dict[str, SimAgent] = build_agents()


def get_champion() -> SimAgent | None:
    """승률 최고 에이전트 반환 (최소 10거래 이상)."""
    candidates = [a for a in AGENTS.values() if a.total_trades >= 10]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.win_rate)


def update_champion() -> str | None:
    """챔피언 재선정 후 agent_id 반환."""
    for a in AGENTS.values():
        a.is_champion = False
    champ = get_champion()
    if champ:
        champ.is_champion = True
        return champ.agent_id
    return None
