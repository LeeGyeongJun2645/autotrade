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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.features import FEATURE_NAMES, compute_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL    = 1_000_000.0   # 에이전트당 초기 가상 자금 (100만원)
POSITION_RATIO     = 0.15           # 잔액의 15%씩 사용 (0.5→0.15: 리스크 70% 축소)
MAX_OPEN_POSITIONS = 2              # 에이전트당 동시 최대 포지션 수 (3→2)

# ── 피처 세트 정의 ────────────────────────────────────────────────

FEATURE_SETS: dict[str, list[str]] = {
    "all": FEATURE_NAMES,
    "momentum": [
        "rsi_9", "macd_diff", "stoch_rsi", "williams_r",
        "mfi_14", "roc_10", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        "rsi_lag_1", "rsi_lag_2", "macd_diff_lag_1",
        "ret_lag_1", "ret_lag_2", "ret_lag_3",
        "regime_state", "mtf_ema_bull",
        "hour_sin", "hour_cos", "is_opening_hour",
        "btc_corr_20", "btc_ret_5",
        "taker_buy_ratio",
        "ha_color", "ha_bull_streak",
        # OFI + MTF (모멘텀 방향성 강화)
        "ofi_5", "ofi_20", "cvd_ratio",
        "mtf_ret_15m", "mtf_ret_1h", "mtf_align",
        # 장중 세션 모멘텀 (주식 전용, 코인=중립)
        "session_phase", "ret_since_open", "intraday_reversal",
        # 이중바닥 + 매도소진 (모멘텀 반전 신호)
        "double_bottom_signal", "exhaustion_bounce",
        "stoch_rsi_k", "stoch_rsi_d", "stoch_rsi_cross",
    ],
    "trend": [
        "ma5_ratio", "ma20_ratio", "ma60_ratio",
        "ma5_cross_ma20", "ma20_cross_ma60",
        "adx_14", "adx_pos", "adx_neg",
        "trix_15", "vortex_diff",
        "ema9_cross_ema21", "vwap_ratio", "vwap_cross",
        "bb_pband_lag_1",
        "regime_state", "anchored_vwap_ratio", "anchored_vwap_cross", "mtf_ema_bull",
        "hour_sin", "hour_cos", "is_closing_hour",
        "kospi_ret_5", "kospi_rel_str",
        "oi_price_diverge",
        "ichi_above_cloud", "ichi_cloud_green", "ichi_tenkan_kijun_bull",
        "above_pp", "dist_to_r1",
        # MTF 모멘텀 + GK 변동성 (추세 강도 보조)
        "mtf_ret_1h", "mtf_ret_4h", "mtf_align",
        "gk_vol",
        # ORB 돌파 (주식 지지/저항 기반 추세 확인, 코인=중립)
        "orb_position", "orb_breakout", "orb_high_pct", "session_phase",
        # 정배열/역배열 + 피보나치 + 지지선 기울기 (추세 구조 강화)
        "ma_bull_align", "ma_bear_align",
        "fib_382_support", "fib_50_support", "fib_618_support",
        "support_slope", "double_bottom_signal",
    ],
    "volume": [
        "vol_ratio", "cmf_20",
        "mfi_14", "ret_1d", "ret_5d", "ret_20d",
        "vol_ratio_lag_1",
        "regime_state", "bb_squeeze",
        "hour_sin", "hour_cos", "is_lunch_hour",
        "btc_ret_5", "kospi_ret_5",
        "oi_change_pct", "taker_buy_ratio",
        "dist_to_pp", "dist_to_s1",
        # OFI + GK (거래량 품질 강화)
        "ofi_5", "ofi_20", "cvd_ratio",
        "gk_vol",
        "session_phase", "ret_since_open",
        # 매도소진 + 가짜돌파 필터 (거래량 신호 품질 강화)
        "exhaustion_bounce", "false_breakout_warn", "vol_reversal_signal",
    ],
}

# ── 20개 에이전트 설정 ────────────────────────────────────────────
# 코인: 60분봉 (1h bars) — 5분봉 대비 노이즈:신호 비율 대폭 개선, EV 음수 구간 탈피 목표
# 주식: 15분봉 — 장중 6.5시간 기준 26봉 확보, 5분봉 대비 노이즈 감소
# lookahead: 코인 3=3시간/5=5시간/8=8시간, 주식 3=45분/5=75분/8=2시간

AGENT_CONFIGS: list[tuple] = [
    # (agent_id, interval_min, label_threshold, buy_threshold, feature_set, market, lookahead, model_type)
    # label_threshold: 60분봉 기준 3~8시간 내 수익 기준 (5분봉 0.006~0.012 → 1h 0.008~0.015)
    # 코인 홀수 → LightGBM / 짝수 → XGBoost (앙상블 다양성 극대화)
    # archive 2주치 실적: AI01(-43%) AI04(-52%) AI08(-96%) AI10(-79%) 모두 60분봉 최악
    # → AI16(+25%)/AI12(+7%) 15분봉 trend 패턴으로 교체 (archive 기반 데이터 결정)
    ("AI01", 15, 0.006, 0.60, "trend",    "coin",  8, "lgbm"),  # AI16 패턴 코인 버전 (1위)
    ("AI02", 60, 0.009, 0.63, "momentum", "coin",  5, "xgb"),
    ("AI03", 60, 0.010, 0.62, "trend",    "coin",  8, "lgbm"),  # 장기 추세형
    ("AI04", 15, 0.006, 0.62, "trend",    "coin",  5, "xgb"),   # AI12 패턴 코인 버전 (2위)
    ("AI05", 60, 0.012, 0.60, "all",      "coin",  5, "lgbm"),
    ("AI06", 60, 0.010, 0.65, "momentum", "coin",  8, "xgb"),
    ("AI07", 60, 0.012, 0.62, "trend",    "coin",  3, "lgbm"),
    ("AI08", 15, 0.005, 0.60, "momentum", "coin",  5, "xgb"),   # 15분봉 모멘텀
    ("AI09", 60, 0.015, 0.65, "all",      "coin",  8, "lgbm"),
    ("AI10", 15, 0.006, 0.65, "volume",   "coin",  6, "xgb"),   # 15분봉 거래량
    # 주식: 15분봉 — 장중 충분한 데이터 확보 + 노이즈 감소
    ("AI11", 15, 0.007, 0.58, "all",      "stock", 3, "lgbm"),
    ("AI12", 15, 0.006, 0.60, "trend",    "stock", 5, "lgbm"),
    ("AI13", 15, 0.009, 0.58, "momentum", "stock", 8, "lgbm"),
    ("AI14", 15, 0.006, 0.60, "volume",   "stock", 3, "lgbm"),
    ("AI15", 15, 0.011, 0.60, "all",      "stock", 5, "lgbm"),
    ("AI16", 15, 0.006, 0.60, "trend",    "stock", 8, "lgbm"),
    ("AI17", 15, 0.011, 0.62, "momentum", "stock", 3, "lgbm"),
    ("AI18", 15, 0.007, 0.60, "volume",   "stock", 5, "lgbm"),
    ("AI19", 15, 0.010, 0.65, "all",      "stock", 8, "lgbm"),
    ("AI20", 15, 0.008, 0.60, "trend",    "stock", 6, "lgbm"),
    # ── 신규 코인 에이전트 (AI21-AI24): 상위 전략(AI01 WR=71%, AI03 WR=80%) 변형 확장
    ("AI21", 60, 0.012, 0.67, "trend",    "coin",  8, "lgbm"),  # AI03 변형 — 60분봉 추세, 임계값↑
    ("AI22", 15, 0.005, 0.70, "trend",    "coin",  5, "lgbm"),  # AI01 변형 — 15분봉 추세, 높은 임계값
    ("AI23", 60, 0.010, 0.66, "momentum", "coin",  5, "lgbm"),  # 60분봉 모멘텀 lgbm (AI02=xgb 대칭)
    ("AI24", 15, 0.007, 0.68, "all",      "coin",  8, "lgbm"),  # 15분봉 전체 피처 코인 lgbm
    # ── 신규 주식 에이전트 (AI25-AI28): AI16(WR=100%) 패턴 기반 확장
    ("AI25", 15, 0.005, 0.63, "trend",    "stock", 8, "lgbm"),  # AI16 변형 — 낮은 label, 잦은 신호
    ("AI26", 15, 0.008, 0.65, "trend",    "stock", 5, "lgbm"),  # 중간 label 추세 (AI16/AI20 사이)
    ("AI27", 15, 0.005, 0.63, "volume",   "stock", 3, "lgbm"),  # 거래량 기반 단기 주식
    ("AI28", 60, 0.010, 0.65, "trend",    "stock", 8, "lgbm"),  # 60분봉 추세 주식 (새 타임프레임)
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
        market: str = "coin",   # "coin" | "stock"
        lookahead: int = 5,     # 레이블링 시 몇 봉 앞 수익 기준 (3=15분/5=25분/8=40분)
        model_type: str = "xgb",  # "xgb" | "lgbm"
    ) -> None:
        self.agent_id = agent_id
        self.interval_min = interval_min
        self.label_threshold = label_threshold
        self.buy_threshold = buy_threshold
        self.feature_set = feature_set
        self.market = market
        self.lookahead = lookahead
        self.model_type = model_type
        self.feature_names = FEATURE_SETS[feature_set]

        self._balance = INITIAL_CAPITAL
        self._positions: dict[str, AgentPosition] = {}
        self._model: XGBClassifier | LGBMClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._trained_at: str | None = None
        self._model_path = MODEL_DIR / f"{model_type}_agent_{agent_id}_{interval_min}m.pkl"
        self._train_lock = asyncio.Lock()   # 동시 재학습 race condition 방지

        self.total_trades = 0
        self.win_trades = 0
        self.is_champion = False
        self.is_active = True  # False면 매수 스킵 (자동 임계값 조정에서 관리)
        self.recent_trades: list[dict] = []
        self._last_position_values: dict[str, float] = {}  # ticker → 현재 평가액
        self._last_atr_pct: float = 0.0          # ATR 기반 동적 손익용 (predict()에서 업데이트)
        self._last_vol_ratio: float = 1.0        # 최근 거래량 비율 (vol/ma20) — RVOL 필터용
        self._last_adx_14: float = 0.0           # ADX(14) — BUY 필터: 횡보장 진입 차단용
        self._last_hold_reason: str = ""         # 마지막 hold 반환 이유 (앙상블 원인분석용)
        self._meta_model: LGBMClassifier | None = None   # Meta-Labeling 2차 필터 모델
        self._meta_scaler: StandardScaler | None = None
        self._meta_path = MODEL_DIR / f"meta_agent_{agent_id}_{interval_min}m.pkl"
        self._cached_funding_rates: list[dict] = []  # 재학습 시 업데이트, predict()에서 사용
        self._cached_oi_hist: list[dict] = []         # BTC OI 히스토리 캐시 (코인 전용)
        self._cached_taker_hist: list[dict] = []      # BTC Taker 비율 히스토리 캐시 (코인 전용)
        self._cached_ls_hist: list[dict] = []         # BTC 글로벌 L/S 비율 히스토리 캐시 (코인 전용)
        self._last_ohlcv_cache: dict[str, list[dict]] = {}  # 종목별 최신 OHLCV (continuation_score 용, 최대 100종목)
        self._OHLCV_CACHE_MAXSIZE: int = 100  # 에이전트당 캐시 최대 종목 수 (22에이전트 × 100 × 200봉 ≈ 430MB 상한)
        self._peak_price: dict[str, float] = {}  # 트레일링 스탑용 최고가 추적
        self._trailing_mode: set[str] = set()
        # ── MFE/MAE 기반 동적 TP 배수 ─────────────────────────────────
        self._dynamic_tp_mult: float = 3.0       # R비율 분석 후 재학습 시 자동 갱신 (2.5~3.5)
        # ── 부분 청산 Scale-out (ATR×1.5 도달 시 40% 매도) ────────────
        self._partial_tp_price: dict[str, float] = {}  # 종목별 1차 TP 목표가
        self._partial_tp_done: set[str] = set()         # 1차 청산 완료 종목 추적
        # ── SHAP 피처 가지치기 ──────────────────────────────────────────
        self._low_importance_feats: set[str] = set()   # 중요도 0.5% 미만 피처 (재학습 시 갱신)
        self._trained_feature_names: list[str] = []   # 현재 모델이 학습된 피처 목록 (predict 일관성)
        self._consecutive_losses: int = 0        # 연속 손실 카운터 → 5회 시 재학습 트리거
        self.needs_retrain: bool = False          # 즉시 재학습 요청 플래그
        self._daily_retrain_count: int = 0       # 당일 비상재학습 횟수 (최대 3회)
        self._last_retrain_date: str = ""        # 마지막 재학습 날짜 (YYYY-MM-DD)
        self._cooldown_tickers: dict[str, str] = {}  # ticker → 쿨다운 만료 시각 (ISO), 손절 후 30분 재매수 금지
        self._today_buy_count: int = 0               # 당일 BUY 횟수 (00:00 KST 리셋)
        self._today_date: str = ""                   # 리셋 기준 날짜 (YYYY-MM-DD KST)
        self.day_start_balance: float = 0.0          # 당일 00:00 잔액 스냅샷 (일일 P&L 계산용)
        self._ticker_blacklist: set[str] = set()     # 성과 기반 동적 블랙리스트 (WR<35%+20건 이상)
        # ── 손실 분석 / 패턴 학습 ────────────────────────────────────────
        self._last_signal_snapshot: dict = {}         # 마지막 predict() 신호값 캐시
        self._buy_context: dict[str, dict] = {}       # 매수 시점 신호 스냅샷 (소급 분석용)
        self._loss_patterns: dict[tuple, int] = {}    # (wrong_signal_조합) → 손실 횟수
        self._loss_analysis_log: list[dict] = []      # 최근 30건 손실 분석 결과

    # ── 프로퍼티 ────────────────────────────────────────────────

    @property
    def adaptive_buy_threshold(self) -> float:
        """연속 손실 횟수에 따라 buy_threshold 동적 상향.

        연속 손실이 쌓일수록 진입 기준이 올라가 과매매를 자동 억제.
        재학습 완료 후 _consecutive_losses=0 리셋 → 임계값도 정상 복귀.
        """
        base = self.buy_threshold
        n = self._consecutive_losses
        if n >= 10:
            return min(base + 0.10, 0.88)
        elif n >= 5:
            return min(base + 0.05, 0.82)
        elif n >= 3:
            return min(base + 0.03, 0.80)
        return base

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
        return max(0.05, min(POSITION_RATIO, kelly * 0.5))  # Half-Kelly, 5~15% 범위

    def _sharpe_weight(self) -> float:
        """최근 거래 수익률 기반 Sortino 비율 가중치.

        Sortino = mean_return / downside_deviation (손실분만 분모).
        Sharpe 대비 손실에 더 가혹한 패널티 → 승률 최대화에 유리.
        거래 5개 미만이면 승률로 폴백.
        """
        sell_trades = [
            t["profit_rate"] for t in self.recent_trades
            if t.get("action") == "SELL" and t.get("profit_rate") is not None
        ]
        if len(sell_trades) < 5:
            return max(self.win_rate, 0.1)
        mean_r    = float(np.mean(sell_trades))
        neg_rets  = [r for r in sell_trades if r < 0.0]
        if not neg_rets:
            return max(mean_r * 10 + 0.5, 0.1)  # 전승: 높은 가중치
        downside  = float(np.sqrt(np.mean([r ** 2 for r in neg_rets])))
        if downside < 1e-9:
            return max(mean_r * 10 + 0.5, 0.1)
        sortino   = mean_r / downside
        return max(sortino + 0.5, 0.1)  # 음수 방지

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
            stored_feats = data.get("feature_names", None)
            if stored_feats is None or stored_feats != self.feature_names:
                if stored_feats is None:
                    logger.warning("[%s] 피처 정보 없는 구 모델 → 삭제 (재학습)", self.agent_id)
                else:
                    added   = [f for f in self.feature_names if f not in stored_feats]
                    removed = [f for f in stored_feats if f not in self.feature_names]
                    logger.warning(
                        "[%s] 피처 불일치 → 모델 삭제 (추가=%s, 제거=%s)",
                        self.agent_id, added[:5], removed[:5],
                    )
                self._model_path.unlink(missing_ok=True)
                return False
            loaded = data["model"]
            loaded_type = "lgbm" if isinstance(loaded, LGBMClassifier) else "xgb"
            if loaded_type != self.model_type:
                logger.warning("[%s] 모델 타입 불일치 (%s→%s) → 삭제", self.agent_id, loaded_type, self.model_type)
                self._model_path.unlink(missing_ok=True)
                return False
            self._model = loaded
            self._scaler = data["scaler"]
            self._trained_at = data.get("trained_at")
            self._trained_feature_names = data.get("trained_feature_names", [])
            self._low_importance_feats = set(data.get("low_importance_feats", []))
            self.load_meta_model()  # meta 모델도 함께 복원
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
                    "trained_feature_names": self._trained_feature_names,
                    "low_importance_feats": list(self._low_importance_feats),
                },
                f,
            )

    def load_meta_model(self) -> bool:
        if not self._meta_path.exists():
            return False
        try:
            with open(self._meta_path, "rb") as f:
                data = pickle.load(f)
            self._meta_model = data["model"]
            self._meta_scaler = data["scaler"]
            return True
        except Exception:
            self._meta_path.unlink(missing_ok=True)
            return False

    def train_meta(self, trade_results: list[dict]) -> bool:
        """BUY 시점 컨텍스트 + 결과로 Meta-Labeling 2차 필터 모델 학습.

        피처: buy_prob, hour_sin, hour_cos, buy_adx, buy_vol_ratio
        레이블: profit_rate > 0 → 1 (성공), else 0 (실패)
        데이터: agent_trades WHERE action='BUY' AND buy_prob IS NOT NULL
        """
        valid = [
            t for t in trade_results
            if t.get("buy_prob") is not None and t.get("profit_rate") is not None
        ]
        if len(valid) < 30:
            return False
        try:
            X_rows = []
            for t in valid:
                h = 12
                try:
                    h = int(t.get("traded_at", "2000-01-01T12:00:00")[11:13])
                except Exception:
                    pass
                X_rows.append([
                    float(t["buy_prob"]),
                    np.sin(2 * np.pi * h / 24),
                    np.cos(2 * np.pi * h / 24),
                    float(t.get("buy_adx", 25.0) or 25.0),
                    float(t.get("buy_vol_ratio", 1.0) or 1.0),
                ])
            X = np.array(X_rows)
            y = np.array([1 if (t.get("profit_rate") or 0) > 0 else 0 for t in valid])
            if len(set(y)) < 2:
                return False
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            clf = LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                scale_pos_weight=max(1.0, (y == 0).sum() / max((y == 1).sum(), 1)),
                random_state=42, verbose=-1,
            )
            clf.fit(X_s, y)
            self._meta_model = clf
            self._meta_scaler = scaler
            with open(self._meta_path, "wb") as f:
                pickle.dump({"model": clf, "scaler": scaler}, f)
            logger.info("[%s] Meta-Labeling 모델 학습 완료 (%d샘플, WR=%.1f%%)",
                        self.agent_id, len(valid), y.mean() * 100)
            return True
        except Exception as e:
            logger.debug("[%s] Meta 학습 실패: %s", self.agent_id, e)
            return False

    # ── 학습 (동기 — asyncio.to_thread 로 호출) ─────────────────

    def train(
        self,
        ohlcv_list: list[dict],
        funding_rates: list[dict] | None = None,
        btc_ohlcv: list[dict] | None = None,
        kospi_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
        trade_results: list[dict] | None = None,
        ls_hist: list[dict] | None = None,
    ) -> bool:
        """분봉 OHLCV로 전용 모델 학습."""
        try:
            _ls = ls_hist or (self._cached_ls_hist if self.market == "coin" else None)
            feat_df = compute_features(
                ohlcv_list,
                funding_rates=funding_rates,
                btc_ohlcv=btc_ohlcv,
                kospi_ohlcv=kospi_ohlcv,
                oi_hist=oi_hist,
                taker_hist=taker_hist,
                ls_hist=_ls,
                interval_min=self.interval_min,
            )
            # ATR을 피처 필터링 전에 미리 추출 — 레이블 생성 시 실제 손익 기준과 정합하기 위해
            _atr_full = feat_df["atr_pct"].copy() if "atr_pct" in feat_df.columns else None
            # ADX 필터: 코인만 추세장 구간 학습 — predict() BUY 필터(>=20)와 동일 임계값
            _adx_mask = (
                feat_df["adx_14"] >= 20
                if (self.market == "coin" and "adx_14" in feat_df.columns)
                else None
            )
            _train_cols = [c for c in self.feature_names
                           if c in feat_df.columns and c not in self._low_importance_feats]
            feat_df = feat_df[_train_cols].dropna()
            if _adx_mask is not None:
                feat_df = feat_df[_adx_mask.reindex(feat_df.index, fill_value=False)]
            # ATR도 feat_df 인덱스에 맞춰 정렬
            _atr_series = _atr_full.reindex(feat_df.index) if _atr_full is not None else None
            if len(feat_df) < 50:
                logger.warning("[%s] 학습 스킵: feat_df %d행 < 50 (입력 %d봉, 피처수 %d)",
                               self.agent_id, len(feat_df), len(ohlcv_list), len(self.feature_names))
                return False

            # close_all: ohlcv_list 전체(N_raw)를 DatetimeIndex로 구성
            # compute_features 내부 dropna 때문에 feat_df.index (N_feat) < len(ohlcv_list) (N_raw)
            # → _full_dt_index = feat_df.index 를 그대로 Series index로 쓰면 길이 불일치 ValueError
            # ohlcv_list 날짜 기반 DatetimeIndex(N_raw)로 close_all 구성 후 reindex로 정렬
            _ohlcv_dates = pd.to_datetime([str(c["date"])[:19] for c in reversed(ohlcv_list)])
            close_all = pd.Series(
                [float(c["close"]) for c in reversed(ohlcv_list)],
                index=_ohlcv_dates,
            )
            close_all = close_all[~close_all.index.duplicated(keep="last")]
            close = close_all.reindex(feat_df.index).reset_index(drop=True)
            if close.isna().any():
                logger.warning("[%s] close reindex NaN 발생 — 날짜 불일치. 학습 스킵.", self.agent_id)
                return False
            feat_df = feat_df.reset_index(drop=True)
            if _atr_series is not None:
                _atr_series = _atr_series.reset_index(drop=True)

            # ── 트리플 배리어 레이블링 ────────────────────────────────
            # TP/SL을 봉별 ATR에 맞춰 동적으로 계산 — 실제 매매 손익 기준(ATR×3.0/1.5)과 정합
            # ATR 미존재 시 label_threshold 폴백
            LOOKAHEAD   = self.lookahead
            NOISE_FLOOR = 0.001

            raw_labels = np.zeros(len(close), dtype=int)
            for i in range(len(close) - LOOKAHEAD):
                entry = close.iloc[i]
                # _agent_execute 실제 손익 기준과 정합 (TP=ATR×2.5, SL=ATR×1.5)
                # TP 2.0→2.5: 실행 TP 변경에 맞춰 학습 레이블 상향 (수익률 우선 전략)
                if _atr_series is not None and pd.notna(_atr_series.iloc[i]) and float(_atr_series.iloc[i]) > 0:
                    _atr = float(_atr_series.iloc[i])
                    tp_pct = max(min(_atr * 3.0, 0.10), self.label_threshold)
                    sl_pct = max(min(_atr * 1.5, 0.05), self.label_threshold * 0.5)
                else:
                    tp_pct = self.label_threshold
                    sl_pct = self.label_threshold * 0.5
                tp, sl = entry * (1 + tp_pct), entry * (1 - sl_pct)
                for j in range(1, LOOKAHEAD + 1):
                    p = close.iloc[i + j]
                    if p >= tp:
                        raw_labels[i] = 1
                        break
                    if p <= sl:
                        break  # 손절 터치 → 0 유지
            label = pd.Series(raw_labels, index=range(len(close)))

            # ATR 기반 변동성 필터 (lookahead 없음: 미래 수익률 대신 현재 ATR 사용)
            if _atr_series is not None:
                _vol_series = _atr_series.fillna(NOISE_FLOOR * 2)
            else:
                _vol_series = close.pct_change().abs().rolling(3).mean().fillna(NOISE_FLOOR * 2)
            _vol_series = _vol_series.reset_index(drop=True)

            # 마지막 LOOKAHEAD 행 제거
            feat_df    = feat_df.iloc[:-LOOKAHEAD]
            label      = label.iloc[:-LOOKAHEAD]
            _vol_series = _vol_series.iloc[:-LOOKAHEAD]

            # 길이 보정
            min_len = min(len(feat_df), len(label), len(_vol_series))
            feat_df    = feat_df.iloc[:min_len]
            label      = label.iloc[:min_len]
            _vol_series = _vol_series.iloc[:min_len]

            # 노이즈 필터: label=0이고 ATR이 하위 20% 미만(= 초저변동성 횡보) → 제외
            # NOISE_FLOOR(0.001)는 ATR비율 대비 너무 작아 아무것도 필터링 안 됨 → 분위수 기반으로 변경
            _atr_noise_thr = max(float(np.percentile(_vol_series.values, 20)), NOISE_FLOOR)
            clear_mask = (label == 1) | (_vol_series.values >= _atr_noise_thr)
            feat_df = feat_df[clear_mask]
            label   = label[clear_mask]

            if len(feat_df) < 30 or len(set(label.values)) < 2:
                logger.warning("[%s] 학습 스킵: 레이블 후 %d행 / 클래스수 %d (양성레이블 %d개)",
                               self.agent_id, len(feat_df), len(set(label.values)), int(label.sum()))
                return False

            X = feat_df.values
            y = label.values.astype(int)

            # ── 2-창 Purged Walk-Forward: 이전 레짐(창A) + 최신 레짐(창B) ──────
            # 데이터: [───훈련───][GAP][──창A──][GAP][──창B──]
            # 창A/B 모두 훈련셋 밖 → 데이터 리케이지 없음
            GAP = self.lookahead
            if len(X) >= 400:
                # 충분한 데이터: 2창 WF + 넉넉한 검증셋
                VAL_SIZE = min(200, max(len(X) // 6, 30))
                _two_win = len(X) > 2 * VAL_SIZE + 2 * GAP + 100
            else:
                # 소규모(OTF 200봉 등): 검증셋 최소화, 2창 비활성화 — 학습셋 확보 우선
                VAL_SIZE = max(15, len(X) // 8)
                _two_win = False
            if _two_win:
                _t_end   = len(X) - 2 * VAL_SIZE - 2 * GAP
                X_train  = X[:_t_end];    y_train  = y[:_t_end]
                _a0 = _t_end + GAP;       _a1 = _a0 + VAL_SIZE
                X_val_a  = X[_a0:_a1];   y_val_a  = y[_a0:_a1]   # 창A — 이전 레짐
                X_val    = X[_a1 + GAP:]; y_val    = y[_a1 + GAP:]  # 창B — 최신 레짐
            elif len(X) > VAL_SIZE + GAP + 50:
                X_train  = X[:-(VAL_SIZE + GAP)]; y_train  = y[:-(VAL_SIZE + GAP)]
                X_val    = X[-VAL_SIZE:];          y_val    = y[-VAL_SIZE:]
                X_val_a  = None;                   y_val_a  = None
            else:
                X_train  = X;    y_train  = y
                X_val    = None; y_val    = None
                X_val_a  = None; y_val_a  = None
                _two_win = False

            # 최근 500샘플 2배 가중치 (최신 시장 환경 우선 반영)
            weights = np.ones(len(X_train))
            if len(X_train) > 500:
                weights[-500:] = 2.0

            # 에이전트 거래 결과 기반 sample_weight 조정
            # 수익 낸 매수 시점 봉 → 가중치 ↑ (최대 3배)
            # 손실 낸 매수 시점 봉 → 가중치 ↓ (최소 0.3배)
            if trade_results:
                try:
                    _round_freq = f"{self.interval_min}min"
                    ohlcv_dates = pd.to_datetime(
                        [d["date"] for d in reversed(ohlcv_list)]
                    ).round(_round_freq)
                    ohlcv_dates_arr = ohlcv_dates.values
                    n_train = len(X_train)
                    for tr in trade_results:
                        try:
                            buy_dt = pd.Timestamp(tr["buy_at"]).round(_round_freq).to_datetime64()
                            idx = int(np.searchsorted(ohlcv_dates_arr, buy_dt))
                            if idx >= n_train:
                                continue
                            profit = float(tr["profit_rate"])
                            if profit > 0:
                                boost = min(1.0 + profit * 10, 3.0)
                                weights[idx] *= boost
                            elif profit < 0:
                                decay = max(0.3, 1.0 + profit * 5)
                                weights[idx] *= decay
                        except Exception:
                            continue
                    absorbed = sum(1 for tr in trade_results if tr.get("profit_rate", 0) > 0)
                    logger.info(
                        "[%s] 거래 결과 흡수: 수익 %d건 / 전체 %d건",
                        self.agent_id, absorbed, len(trade_results),
                    )
                except Exception as e:
                    logger.debug("[%s] 거래 결과 weight 적용 실패 (무시): %s", self.agent_id, e)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            # 클래스 불균형 자동 보정 — buy 레이블이 적은 하락장에서도 공정한 학습
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

            if self.model_type == "lgbm":
                # min_child_samples: 소규모 학습 데이터(OTF 200봉)에서도 분할 가능하도록 적응형
                _mcs = max(10, min(30, len(X_train) // 10))
                clf = LGBMClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=_mcs,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
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
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )
            fit_kwargs: dict = {"sample_weight": weights}
            if self.model_type != "lgbm":
                fit_kwargs["verbose"] = False
            clf.fit(X_train_s, y_train, **fit_kwargs)

            # ── SHAP 피처 중요도 분석 (built-in importances, shap 라이브러리 불필요) ──
            try:
                _active_feats = [c for c in self.feature_names if c in feat_df.columns]
                _imp  = clf.feature_importances_
                _tot  = _imp.sum()
                if _tot > 0 and len(_imp) == len(_active_feats):
                    _imp_pct = _imp / _tot
                    _imp_map = dict(zip(_active_feats, _imp_pct))
                    _sorted_imp = sorted(_imp_map.items(), key=lambda x: x[1], reverse=True)
                    logger.info("[%s] 피처중요도 TOP5: %s | BOTTOM5: %s",
                                self.agent_id,
                                [(k, f"{v:.1%}") for k, v in _sorted_imp[:5]],
                                [(k, f"{v:.2%}") for k, v in _sorted_imp[-5:]])
                    # 0.3% 미만 피처 → 저중요도 목록 갱신 (최소 30개 남기도록 제한)
                    # 0.5%→0.3%: FVG·VSA 등 희귀 이진 피처가 과도하게 pruning되는 문제 방지
                    _low = {k for k, v in _imp_map.items() if v < 0.003}
                    if len(_active_feats) - len(_low) >= 30:
                        self._low_importance_feats = _low
                        if _low:
                            logger.debug("[%s] 저중요도 피처 %d개 다음 학습 제외 예정",
                                         self.agent_id, len(_low))
            except Exception:
                pass

            # Purged Walk-Forward 검증 — 정밀도(승률) 기반으로 평가
            if X_val is not None and len(X_val) > 0:
                _val_s    = scaler.transform(X_val)
                _val_prob = clf.predict_proba(_val_s)[:, 1]
                val_acc   = clf.score(_val_s, y_val)
                _val_pred = (_val_prob >= self.buy_threshold).astype(int)
                from sklearn.metrics import precision_score as _ps2, recall_score as _rs2
                _val_prec = _ps2(y_val, _val_pred, zero_division=0)
                _val_rec  = _rs2(y_val, _val_pred, zero_division=0)
                _n_val_buy = int(_val_pred.sum())
                # n_buy=0: 검증셋에서 buy 예측 0개 → threshold 너무 높거나 완전 보수 모델
                # 이런 모델은 실거래에서도 threshold 근처 확률이 불안정 → 차단
                _wf_fail = val_acc < 0.50 or (_val_prec < 0.40 and _n_val_buy > 5) or (
                    _n_val_buy < 2 and len(X_val) > 20
                )
                if _wf_fail:
                    logger.warning(
                        "[%s] WF검증 실패 acc=%.1f%% prec=%.1f%% rec=%.1f%% (thr=%.2f n_buy=%d)",
                        self.agent_id, val_acc*100, _val_prec*100, _val_rec*100,
                        self.buy_threshold, _n_val_buy,
                    )
                    return False

                from sklearn.metrics import precision_score as _ps
                _init_thr = self.buy_threshold

                def _find_best_thr(prob_arr: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
                    best_thr, best_prec = _init_thr, 0.0
                    for _t in np.arange(0.50, 0.75, 0.01):  # 상한 0.85→0.75: 과도한 임계값 방지
                        _p = (prob_arr >= _t).astype(int)
                        if _p.sum() < max(5, len(y_true) // 20):
                            continue
                        if (y_true == 1).sum() > 0 and float(_p[y_true == 1].mean()) < 0.10:
                            continue
                        _prec = _ps(y_true, _p, zero_division=0)
                        if _prec > best_prec:
                            best_prec, best_thr = _prec, float(_t)
                    return best_thr, best_prec

                # 창B (최신 레짐) 임계값
                thr_b, prec_b = _find_best_thr(_val_prob, y_val)

                if _two_win and X_val_a is not None and len(X_val_a) > 0:
                    # 창A (이전 레짐) 임계값 — 훈련셋 밖, 리케이지 없음
                    _va_prob = clf.predict_proba(scaler.transform(X_val_a))[:, 1]
                    thr_a, _ = _find_best_thr(_va_prob, y_val_a)
                    # 최종: 최신 60% + 이전 40% 가중 평균
                    if prec_b > 0:
                        _combined = 0.6 * thr_b + 0.4 * thr_a
                        self.buy_threshold = round(min(max(_combined, 0.65), 0.75), 2)
                    logger.debug("[%s] 2창WF %.1f%% | 창A %.2f + 창B %.2f → %.2f",
                                 self.agent_id, val_acc * 100, thr_a, thr_b, self.buy_threshold)
                else:
                    if prec_b > 0:
                        self.buy_threshold = round(min(max(thr_b, 0.65), 0.75), 2)
                    logger.info(
                        "[%s] WF검증 acc=%.1f%% prec=%.1f%% rec=%.1f%% | thr=%.2f | n_train=%d n_val=%d",
                        self.agent_id, val_acc*100, prec_b*100, _val_rec*100,
                        self.buy_threshold, len(X_train), len(X_val),
                    )

            # 최소 임계값 — 수익률 중심: 주식 0.60, 코인 0.65 (낮은 임계값이 손실의 주원인)
            _min_thr = 0.60 if self.market == "stock" else 0.65
            self.buy_threshold = min(max(self.buy_threshold, _min_thr), 0.75)
            self._model = clf
            self._scaler = scaler
            self._trained_feature_names = list(feat_df.columns)  # predict() 피처 일관성 보장
            self._trained_at = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            self.save_model()
            logger.warning("[%s] 모델 학습 완료 (%d샘플, X_train=%d, thr=%.2f)", self.agent_id, len(X), len(X_train), self.buy_threshold)
            return True
        except Exception as e:
            logger.warning("[%s] 학습 실패: %s", self.agent_id, e)
            return False

    def _build_labeled_segment(
        self,
        ohlcv_list: list[dict],
        kospi_ohlcv: list[dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """단일 종목 OHLCV → (X, y, feature_cols) 배열 반환. 실패/샘플 부족 시 None."""
        try:
            feat_df = compute_features(ohlcv_list, kospi_ohlcv=kospi_ohlcv, interval_min=self.interval_min)
            _atr_full = feat_df["atr_pct"].copy() if "atr_pct" in feat_df.columns else None
            _seg_cols = [c for c in self.feature_names
                         if c in feat_df.columns and c not in self._low_importance_feats]
            if not _seg_cols:
                logger.debug("[%s] _build_labeled_segment: _seg_cols 없음 (feat_df cols=%d)", self.agent_id, len(feat_df.columns))
                return None
            feat_df = feat_df[_seg_cols].dropna()
            if len(feat_df) < 30:
                logger.debug("[%s] _build_labeled_segment: dropna 후 %d행 < 30", self.agent_id, len(feat_df))
                return None
            _atr_series = _atr_full.reindex(feat_df.index) if _atr_full is not None else None

            _ohlcv_dates = pd.to_datetime([str(c["date"])[:19] for c in reversed(ohlcv_list)])
            close_all = pd.Series(
                [float(c["close"]) for c in reversed(ohlcv_list)], index=_ohlcv_dates,
            )
            close_all = close_all[~close_all.index.duplicated(keep="last")]
            close = close_all.reindex(feat_df.index).reset_index(drop=True)
            if close.isna().any():
                logger.debug("[%s] _build_labeled_segment: close NaN %d개 (날짜 불일치)", self.agent_id, close.isna().sum())
                return None
            feat_df = feat_df.reset_index(drop=True)
            if _atr_series is not None:
                _atr_series = _atr_series.reset_index(drop=True)

            LOOKAHEAD   = self.lookahead
            NOISE_FLOOR = 0.001
            raw_labels  = np.zeros(len(close), dtype=int)
            for i in range(len(close) - LOOKAHEAD):
                entry = close.iloc[i]
                if _atr_series is not None and pd.notna(_atr_series.iloc[i]) and float(_atr_series.iloc[i]) > 0:
                    _atr   = float(_atr_series.iloc[i])
                    # ATR×2.0: 기존 3.0 대비 낮춰 레이블 생성량 증가 (주식 단기봉 ATR이 작아 레이블 부족 방지)
                    tp_pct = max(min(_atr * 2.0, 0.08), self.label_threshold)
                    sl_pct = max(min(_atr * 1.5, 0.05), self.label_threshold * 0.5)
                else:
                    tp_pct = self.label_threshold
                    sl_pct = self.label_threshold * 0.5
                tp, sl = entry * (1 + tp_pct), entry * (1 - sl_pct)
                for j in range(1, LOOKAHEAD + 1):
                    p = close.iloc[i + j]
                    if p >= tp:
                        raw_labels[i] = 1
                        break
                    if p <= sl:
                        break

            label      = pd.Series(raw_labels)
            next_close = close.shift(-LOOKAHEAD)
            ret        = (next_close / close - 1).fillna(0)

            feat_df = feat_df.iloc[:-LOOKAHEAD]
            label   = label.iloc[:-LOOKAHEAD]
            ret     = ret.iloc[:-LOOKAHEAD]
            min_len = min(len(feat_df), len(label))
            feat_df = feat_df.iloc[:min_len]
            label   = label.iloc[:min_len]
            ret     = ret.iloc[:min_len]

            clear_mask = (label == 1) | (ret.abs() >= NOISE_FLOOR)
            feat_df    = feat_df[clear_mask.values]
            label      = label[clear_mask.values]

            _n_pos = int(label.sum())
            if len(feat_df) < 30 or len(set(label.values)) < 2:
                logger.debug("[%s] _build_labeled_segment: 최종 %d행 / 양성 %d개 → 스킵", self.agent_id, len(feat_df), _n_pos)
                return None
            return feat_df.values, label.values.astype(int), list(feat_df.columns)
        except Exception as e:
            logger.warning("[%s] _build_labeled_segment 실패: %s", self.agent_id, e)
            return None

    def train_multi(
        self,
        ohlcv_by_symbol: dict[str, list[dict]],
        kospi_ohlcv: list[dict] | None = None,
    ) -> bool:
        """여러 종목 OHLCV를 풀링해 단일 모델 학습 (주식 에이전트 전용)."""
        try:
            all_X: list[np.ndarray] = []
            all_y: list[np.ndarray] = []
            _actual_cols: list[str] = []
            for sym, ohlcv_list in ohlcv_by_symbol.items():
                if len(ohlcv_list) < 50:
                    continue
                seg = self._build_labeled_segment(ohlcv_list, kospi_ohlcv)
                if seg is None:
                    continue
                all_X.append(seg[0])
                all_y.append(seg[1])
                if not _actual_cols:
                    _actual_cols = seg[2]  # 첫 세그먼트 컬럼 기록 (실제 학습 피처)
                logger.debug("[%s] train_multi: %s %d샘플", self.agent_id, sym, len(seg[0]))

            if not all_X:
                logger.warning("[%s] train_multi: 유효 종목 없음 (입력 %d종목, 각 봉수: %s)",
                               self.agent_id, len(ohlcv_by_symbol),
                               {s: len(v) for s, v in ohlcv_by_symbol.items()})
                return False

            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)

            if len(X) < 50 or len(set(y.tolist())) < 2:
                logger.warning("[%s] train_multi 스킵: 풀링 후 %d샘플 / 클래스 %d",
                               self.agent_id, len(X), len(set(y.tolist())))
                return False

            GAP = self.lookahead
            if len(X) >= 400:
                VAL_SIZE = min(200, max(len(X) // 6, 30))
                _two_win = len(X) > 2 * VAL_SIZE + 2 * GAP + 100
            else:
                VAL_SIZE = max(15, len(X) // 8)
                _two_win = False
            if _two_win:
                _t_end  = len(X) - 2 * VAL_SIZE - 2 * GAP
                X_train = X[:_t_end];    y_train = y[:_t_end]
                _a0     = _t_end + GAP;  _a1     = _a0 + VAL_SIZE
                X_val_a = X[_a0:_a1];   y_val_a = y[_a0:_a1]
                X_val   = X[_a1 + GAP:]; y_val   = y[_a1 + GAP:]
            elif len(X) > VAL_SIZE + GAP + 50:
                X_train = X[:-(VAL_SIZE + GAP)]; y_train = y[:-(VAL_SIZE + GAP)]
                X_val   = X[-VAL_SIZE:];          y_val   = y[-VAL_SIZE:]
                X_val_a = None;                   y_val_a = None
            else:
                X_train = X;    y_train = y
                X_val   = None; y_val   = None
                X_val_a = None; y_val_a = None
                _two_win = False

            weights = np.ones(len(X_train))
            if len(X_train) > 500:
                weights[-500:] = 2.0

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            n_neg     = int((y_train == 0).sum())
            n_pos     = int((y_train == 1).sum())
            scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

            _mcs = max(10, min(30, len(X_train) // 10))
            if self.model_type == "lgbm":
                clf = LGBMClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=_mcs,
                    reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=scale_pos,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
            else:
                clf = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    gamma=0.1, reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=scale_pos,
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
            fit_kwargs: dict = {"sample_weight": weights}
            if self.model_type != "lgbm":
                fit_kwargs["verbose"] = False
            clf.fit(X_train_s, y_train, **fit_kwargs)

            # ── SHAP 피처 중요도 분석 (train_multi 주식 에이전트용) ──
            try:
                _tm_imp  = clf.feature_importances_
                _tm_tot  = _tm_imp.sum()
                if _tm_tot > 0 and _actual_cols and len(_tm_imp) == len(_actual_cols):
                    _tm_pct  = _tm_imp / _tm_tot
                    _tm_map  = dict(zip(_actual_cols, _tm_pct))
                    _tm_srt  = sorted(_tm_map.items(), key=lambda x: x[1], reverse=True)
                    logger.info("[%s] 피처중요도(multi) TOP5: %s | BOTTOM5: %s",
                                self.agent_id,
                                [(k, f"{v:.1%}") for k, v in _tm_srt[:5]],
                                [(k, f"{v:.2%}") for k, v in _tm_srt[-5:]])
                    _tm_low = {k for k, v in _tm_map.items() if v < 0.003}
                    if len(_actual_cols) - len(_tm_low) >= 30:
                        self._low_importance_feats = _tm_low
            except Exception:
                pass

            if X_val is not None and len(X_val) > 0:
                _val_s    = scaler.transform(X_val)
                _val_prob = clf.predict_proba(_val_s)[:, 1]
                val_acc   = clf.score(_val_s, y_val)
                _val_pred    = (_val_prob >= self.buy_threshold).astype(int)
                from sklearn.metrics import precision_score as _ps, recall_score as _rs
                _val_prec    = _ps(y_val, _val_pred, zero_division=0)
                _n_val_buy   = int(_val_pred.sum())
                _tm_wf_fail  = val_acc < 0.50 or (_val_prec < 0.40 and _n_val_buy > 5) or (
                    _n_val_buy < 2 and len(X_val) > 20
                )
                if _tm_wf_fail:
                    logger.warning(
                        "[%s] train_multi WF검증 실패 acc=%.1f%% prec=%.1f%% (thr=%.2f n_buy=%d)",
                        self.agent_id, val_acc * 100, _val_prec * 100,
                        self.buy_threshold, _n_val_buy,
                    )
                    return False

                from sklearn.metrics import precision_score as _ps
                _init_thr = self.buy_threshold

                def _find_best_thr(prob_arr: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
                    best_thr, best_prec = _init_thr, 0.0
                    for _t in np.arange(0.50, 0.75, 0.01):
                        _p = (prob_arr >= _t).astype(int)
                        if _p.sum() < max(5, len(y_true) // 20):
                            continue
                        if (y_true == 1).sum() > 0 and float(_p[y_true == 1].mean()) < 0.10:
                            continue
                        _prec = _ps(y_true, _p, zero_division=0)
                        if _prec > best_prec:
                            best_prec, best_thr = _prec, float(_t)
                    return best_thr, best_prec

                thr_b, prec_b = _find_best_thr(_val_prob, y_val)
                if _two_win and X_val_a is not None and len(X_val_a) > 0:
                    _va_prob = clf.predict_proba(scaler.transform(X_val_a))[:, 1]
                    thr_a, _ = _find_best_thr(_va_prob, y_val_a)
                    if prec_b > 0:
                        _combined = 0.6 * thr_b + 0.4 * thr_a
                        self.buy_threshold = round(min(max(_combined, 0.60), 0.75), 2)
                else:
                    if prec_b > 0:
                        self.buy_threshold = round(min(max(thr_b, 0.60), 0.75), 2)

            # 수익률 중심: 주식 최소 0.60 (기존 0.55에서 상향)
            _min_thr = 0.60 if self.market == "stock" else 0.65
            self.buy_threshold = min(max(self.buy_threshold, _min_thr), 0.75)
            self._model   = clf
            self._scaler  = scaler
            self._trained_feature_names = _actual_cols  # 실제 학습에 사용된 컬럼
            self._trained_at = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            self.save_model()
            logger.warning("[%s] train_multi 완료 (%d종목 풀링, %d샘플, thr=%.2f)",
                           self.agent_id, len(all_X), len(X), self.buy_threshold)
            return True
        except Exception as e:
            logger.warning("[%s] train_multi 실패: %s", self.agent_id, e)
            return False

    # ── 보유 포지션 지속성 분석 ──────────────────────────────────

    def continuation_score(
        self,
        ohlcv_list: list[dict],
        btc_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
        ticker: str = "",
    ) -> tuple[float, list[str]]:
        """보유 포지션의 상승 지속 가능성 종합 평가.

        모든 보유 가능한 지표(기술적·모멘텀·거래량·시장구조·캔들)를 활용해
        '더 오를지(양수) / 반전할지(음수)' 판단.

        Returns:
            (score, reasons): score -1.0~+1.0, reasons = 판단 근거 목록
        """
        try:
            full_df = compute_features(
                ohlcv_list,
                funding_rates=self._cached_funding_rates or None,
                btc_ohlcv=btc_ohlcv,
                oi_hist=oi_hist or (self._cached_oi_hist or None),
                taker_hist=taker_hist or (self._cached_taker_hist or None),
                ticker=ticker,
                interval_min=self.interval_min,
            )
        except Exception:
            return 0.0, []

        if full_df.empty or len(full_df) < 5:
            return 0.0, []

        row   = full_df.iloc[-1]
        prev  = full_df.iloc[-2] if len(full_df) >= 2 else row
        prev3 = full_df.iloc[-4] if len(full_df) >= 4 else prev

        score: float = 0.0
        reasons: list[str] = []

        def _g(col: str, default: float = 0.0) -> float:
            try:
                v = row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)
                return float(v) if v is not None and not np.isnan(float(v)) else default
            except Exception:
                return default

        # ── 1. 모멘텀: RSI ─────────────────────────────────────
        rsi = _g("rsi_9")
        if rsi > 80:
            score -= 0.30; reasons.append(f"RSI과매수({rsi:.0f}>80 극단)")
        elif rsi > 75:
            score -= 0.15; reasons.append(f"RSI과매수({rsi:.0f}>75)")
        elif 50 < rsi <= 65:
            score += 0.15; reasons.append(f"RSI모멘텀({rsi:.0f} 50~65 적정)")
        elif rsi < 35:
            score -= 0.20; reasons.append(f"RSI모멘텀소실({rsi:.0f}<35)")

        # ── 2. MACD 방향 ────────────────────────────────────────
        macd_now  = _g("macd_diff")
        macd_prev = float(prev.get("macd_diff", 0) if hasattr(prev, "get") else 0)
        if macd_now > 0 and macd_now > macd_prev:
            score += 0.20; reasons.append(f"MACD상승지속({macd_now:.4f}↑)")
        elif macd_now > 0 and macd_now < macd_prev:
            score -= 0.10; reasons.append("MACD모멘텀약화(양→감소)")
        elif macd_prev > 0 and macd_now <= 0:
            score -= 0.25; reasons.append("MACD음전환(모멘텀반전)")

        # ── 3. ADX: 추세 강도 ────────────────────────────────────
        adx = _g("adx_14")
        if adx > 30:
            score += 0.15; reasons.append(f"ADX강한추세({adx:.0f}>30)")
        elif adx < 15:
            score -= 0.15; reasons.append(f"ADX횡보({adx:.0f}<15 추세없음)")

        # ── 4. 거래량: 모멘텀 참여도 ─────────────────────────────
        vol_ratio = _g("vol_ratio", 1.0)
        if vol_ratio > 2.0:
            score += 0.15; reasons.append(f"RVOL강(x{vol_ratio:.1f} 기관참여)")
        elif vol_ratio > 1.3:
            score += 0.08; reasons.append(f"RVOL양호(x{vol_ratio:.1f})")
        elif vol_ratio < 0.5:
            score -= 0.20; reasons.append(f"RVOL급감(x{vol_ratio:.1f} 모멘텀소멸)")
        elif vol_ratio < 0.8:
            score -= 0.10; reasons.append(f"RVOL약(x{vol_ratio:.1f})")

        # ── 5. 멀티타임프레임 정렬 ────────────────────────────────
        mtf_align = _g("mtf_align")
        if mtf_align >= 3:
            score += 0.20; reasons.append("MTF3개타임프레임강세정렬")
        elif mtf_align == 2:
            score += 0.10; reasons.append("MTF2개타임프레임강세")
        elif mtf_align == 0:
            score -= 0.20; reasons.append("MTF전타임프레임하락")

        # ── 6. 헤이킨 아시 캔들 ──────────────────────────────────
        ha_streak = _g("ha_bull_streak")
        ha_no_lower = _g("ha_no_lower_shadow")
        ha_color = _g("ha_color")
        if ha_streak >= 3 and ha_no_lower:
            score += 0.15; reasons.append(f"HA강세지속({ha_streak:.0f}봉연속양봉·아래꼬리없음)")
        elif ha_streak >= 2:
            score += 0.08; reasons.append(f"HA양봉지속({ha_streak:.0f}봉)")
        elif ha_color == 0 and ha_no_lower == 0:  # 음봉 + 아래꼬리 없음 → 하락세
            score -= 0.15; reasons.append("HA음봉·아래꼬리없음(하락지속)")

        # ── 7. 볼린저 밴드 위치 ───────────────────────────────────
        bb_pband = _g("bb_pband", 0.5)
        if bb_pband > 0.95:
            score -= 0.20; reasons.append(f"BB상단초과({bb_pband:.2f} 과매수)")
        elif bb_pband > 0.80:
            score -= 0.08; reasons.append(f"BB상단근접({bb_pband:.2f})")
        elif 0.4 < bb_pband <= 0.7:
            score += 0.08; reasons.append(f"BB중심권({bb_pband:.2f} 상승여력)")

        # ── 8. OI + Taker (코인 전용) ─────────────────────────────
        oi_chg = _g("oi_change_pct")
        taker  = _g("taker_buy_ratio", 0.5)
        if oi_chg > 0.01 and taker > 0.55:
            score += 0.15; reasons.append(f"OI증가({oi_chg*100:.1f}%)+매수우세({taker:.2f})")
        elif oi_chg < -0.02:
            score -= 0.10; reasons.append(f"OI감소({oi_chg*100:.1f}% 롱청산)")
        if taker < 0.42:
            score -= 0.10; reasons.append(f"Taker매도우세({taker:.2f})")

        # ── 9. 펀딩비 (코인 전용) ────────────────────────────────
        funding = _g("funding_rate")
        if funding > 0.003:
            score -= 0.20; reasons.append(f"펀딩비과열(+{funding*100:.2f}% 롱쏠림)")
        elif funding < -0.002:
            score += 0.10; reasons.append(f"펀딩비역(−{abs(funding)*100:.2f}% 숏스퀴즈가능)")

        # ── 10. BTC 상관 (알트 전용) ─────────────────────────────
        btc_ret5 = _g("btc_ret_5")
        if btc_ret5 < -0.02 and ticker and not ticker.startswith("KRW-BTC"):
            score -= 0.15; reasons.append(f"BTC급락({btc_ret5*100:.1f}% 알트동조하락)")
        elif btc_ret5 > 0.015:
            score += 0.08; reasons.append(f"BTC강세({btc_ret5*100:.1f}% 알트견인)")

        # ── 11. 이치모쿠 클라우드 구조 ───────────────────────────
        above_cloud = _g("ichi_above_cloud")
        cloud_green = _g("ichi_cloud_green")
        if above_cloud == 1 and cloud_green == 1:
            score += 0.10; reasons.append("이치모쿠구름위+상승구름(강세구조)")
        elif above_cloud == 0:
            score -= 0.10; reasons.append("이치모쿠구름아래(약세구조)")

        # ── 12. 캔들 패턴 ────────────────────────────────────────
        shooting_star  = _g("is_shooting_star")
        bearish_engulf = _g("is_bearish_engulf")
        hammer         = _g("is_hammer")
        bull_engulf    = _g("is_bullish_engulf")
        if shooting_star:
            score -= 0.15; reasons.append("슈팅스타(고점반전캔들)")
        if bearish_engulf:
            score -= 0.15; reasons.append("하락장악형(반전캔들)")
        if hammer:
            score += 0.10; reasons.append("망치형(반등캔들)")
        if bull_engulf:
            score += 0.10; reasons.append("상승장악형(강세캔들)")

        # ── 13. VWAP 위치 ────────────────────────────────────────
        vwap_cross = _g("vwap_cross")
        vwap_ratio = _g("vwap_ratio")
        if vwap_cross == 1 and vwap_ratio > 0.005:
            score += 0.08; reasons.append(f"VWAP위+괴리({vwap_ratio*100:.1f}% 기관매집)")
        elif vwap_cross == 0:
            score -= 0.10; reasons.append("VWAP아래(기관기준선이탈)")

        # ── 14. 피봇 포인트 (R1 도달) ────────────────────────────
        dist_r1 = _g("dist_to_r1")
        if 0 < dist_r1 < 0.005:
            score -= 0.10; reasons.append(f"R1저항선근접({dist_r1*100:.2f}% 이내)")

        score = float(np.clip(score, -1.0, 1.0))
        return score, reasons

    # ── 예측 ────────────────────────────────────────────────────

    def predict(
        self,
        ohlcv_list: list[dict],
        btc_ohlcv: list[dict] | None = None,
        kospi_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
        ls_hist: list[dict] | None = None,
        ticker: str = "",
        obi_snaps: list[float] | None = None,
    ) -> tuple[str, float]:
        """(signal, buy_prob) 반환. 모델 없으면 ('hold', 0.5)."""
        if self._model is None and not self.load_model():
            return "hold", 0.5
        if ticker and ohlcv_list:
            # LRU 제한: maxsize 초과 시 가장 오래된 항목 삭제 (메모리 누수 방지)
            if len(self._last_ohlcv_cache) >= self._OHLCV_CACHE_MAXSIZE and ticker not in self._last_ohlcv_cache:
                oldest = next(iter(self._last_ohlcv_cache))
                del self._last_ohlcv_cache[oldest]
            self._last_ohlcv_cache[ticker] = ohlcv_list
        try:
            full_df = compute_features(
                ohlcv_list,
                funding_rates=self._cached_funding_rates if self._cached_funding_rates else None,
                btc_ohlcv=btc_ohlcv,
                kospi_ohlcv=kospi_ohlcv,
                oi_hist=oi_hist or (self._cached_oi_hist or None),
                taker_hist=taker_hist or (self._cached_taker_hist or None),
                ticker=ticker if ticker else "",
                ls_hist=ls_hist or (self._cached_ls_hist or None),
                obi_snaps=obi_snaps,
                interval_min=self.interval_min,
            )
            if full_df.empty:
                return "hold", 0.5
            # ATR / RVOL / ADX 캐싱 — _agent_execute에서 동적 손익 및 BUY 필터에 사용
            if "atr_pct" in full_df.columns:
                self._last_atr_pct = float(full_df["atr_pct"].iloc[-1])
            if "vol_ratio" in full_df.columns:
                self._last_vol_ratio = float(full_df["vol_ratio"].iloc[-1])
            if "adx_14" in full_df.columns:
                self._last_adx_14 = float(full_df["adx_14"].iloc[-1])
            # 손실 분석용 신호 스냅샷 캐시 (매수 시점 컨텍스트 보존)
            self._last_signal_snapshot = {
                "rsi_9":       float(full_df["rsi_9"].iloc[-1])       if "rsi_9"       in full_df.columns else 50.0,
                "macd_diff":   float(full_df["macd_diff"].iloc[-1])   if "macd_diff"   in full_df.columns else 0.0,
                "adx_14":      float(full_df["adx_14"].iloc[-1])      if "adx_14"      in full_df.columns else 20.0,
                "bb_pband":    float(full_df["bb_pband"].iloc[-1])    if "bb_pband"    in full_df.columns else 0.5,
                "mtf_align":   float(full_df["mtf_align"].iloc[-1])   if "mtf_align"   in full_df.columns else 1.5,
                "regime_state":float(full_df["regime_state"].iloc[-1])if "regime_state" in full_df.columns else 1.0,
                "vol_ratio":   float(full_df["vol_ratio"].iloc[-1])   if "vol_ratio"   in full_df.columns else 1.0,
            }
            # 학습 시 사용된 피처 목록 우선 사용 (SHAP 가지치기 일관성)
            _pred_cols = (self._trained_feature_names
                          if self._trained_feature_names
                          else [c for c in self.feature_names if c in full_df.columns])
            feat_df = full_df[[c for c in _pred_cols if c in full_df.columns]].dropna()
            if feat_df.empty:
                return "hold", 0.5
            X_last_df = feat_df.iloc[[-1]]
            expected = getattr(self._scaler, "n_features_in_", None)
            if expected is not None and X_last_df.shape[1] != expected:
                logger.warning(
                    "[%s] 피처 수 불일치: 현재 %d개, 스케일러 %d개 — hold 반환",
                    self.agent_id, X_last_df.shape[1], expected,
                )
                return "hold", 0.5
            # scaler는 numpy로 fit됐으므로 numpy로 transform (DataFrame 전달 시 경고 발생)
            X_scaled = self._scaler.transform(X_last_df.values)  # type: ignore[union-attr]
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_last_df.columns)
            prob = float(self._model.predict_proba(X_scaled_df)[0, 1])  # type: ignore[union-attr]
        except Exception:
            logger.debug("[%s] predict 예외 — hold 반환", self.agent_id, exc_info=True)
            return "hold", 0.5

        # ── 하락 레짐 하드 억제: regime_state==0 (하락추세) 시 확률 대폭 감소 ──
        # 이전: ADX>25 = 추세(1)로 분류 → 하락 추세도 BUY 허용 (근본 문제)
        # 수정: +DI/-DI 기반으로 0=하락/2=상승 분리 → 하락추세에서 BUY 사실상 차단
        try:
            if "regime_state" in full_df.columns:
                _regime_val = float(full_df["regime_state"].iloc[-1])
                if _regime_val == 0.0:   # 하락 추세 → 확률 대폭 억제
                    prob = max(0.01, prob * 0.30)
                elif _regime_val == 2.0: # 상승 추세 → 약간 강화
                    prob = min(0.99, prob * 1.05)
        except Exception:
            pass

        # ── MTF 정렬 필터: 3개 타임프레임 모두 하락 정렬 시 차단 ───
        # mtf_align==0: 15분·1시간·4시간 전부 하락 정렬 → 매수 억제
        # 단, ICT 반전 신호(FVG 상승갭, VSA Stopping Volume) 동반 시 소프트 억제로 완화
        # 이유: ICT Power of 3에서 런던 스윕 직후 mtf_align=0이어도 NY 반전 유효
        try:
            if "mtf_align" in full_df.columns:
                _align = float(full_df["mtf_align"].iloc[-1])
                if _align == 0.0:
                    # 3개 TF 전부 하락 → 하드차단 (소프트 억제는 regime_state=2 부스트 0.945*0.7=0.661로 임계값 통과 가능)
                    return "hold", round(prob * 0.70, 4)
                elif _align == 3.0:  # 전부 상승 정렬 → 강화
                    prob = max(0.01, min(0.99, prob * 1.08))
        except Exception:
            pass

        # ── 펀딩레이트 극단값 하드게이트 (코인 전용) ──────────────────────
        # >0.1%(0.001): 롱 과열 → 롱 진입 차단. <-0.05%(-0.0005): 숏 과포화 → 롱 유리
        try:
            if "funding_rate" in full_df.columns:
                _fr = float(full_df["funding_rate"].iloc[-1])
                if _fr > 0.001:   # 0.1% 이상 = 롱 과열 → 신규 롱 차단
                    return "hold", round(prob * 0.4, 4)
                elif _fr < -0.0005:  # -0.05% 이하 = 숏 과포화 → 롱 우호적
                    prob = min(0.99, prob * 1.05)
        except Exception:
            pass

        # ── 보유 중 종목 차트 모니터링: RSI 과매수 + MACD 반전 → 조기 익절 ────
        # 연구: RSI>75 + MACD histogram 음전환 = 모멘텀 소진, 수익률 하락 신호
        # 매수 후 차트를 계속 보면서 "다 올라왔다" 싶으면 파는 전략
        if ticker and ticker in self._positions:
            try:
                _held_pos = self._positions[ticker]
                _now_kst  = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
                _entered  = datetime.fromisoformat(_held_pos.entered_at).replace(tzinfo=None)
                _held_min = (_now_kst - _entered).total_seconds() / 60

                # ── 주식 전용: 15:00 이후 오버나이트 방지 강제 청산 ────────────
                if self.market == "stock" and len(full_df) >= 5:
                    try:
                        _now_k = datetime.now(ZoneInfo("Asia/Seoul"))
                        if _now_k.hour == 15 and _now_k.minute >= 0:
                            from backend.core import sim_log
                            sim_log.push(self.agent_id, f"[차트모니터링] {ticker} 주식마감청산(15시이후)", "SELL")
                            return "sell", round(max(prob, 0.60), 4)
                    except Exception:
                        pass

                if _held_min >= 30 and len(full_df) >= 5:
                    _rsi_now   = float(full_df["rsi_9"].iloc[-1])
                    _rsi_prev  = float(full_df["rsi_9"].iloc[-3])
                    _macd_now  = float(full_df["macd_diff"].iloc[-1])
                    _macd_prev = float(full_df["macd_diff"].iloc[-2])

                    # RSI 과매수 구간(>75) + MACD 히스토그램 음전환 → 모멘텀 소진
                    _rsi_overbought = _rsi_now > 75
                    _macd_turning   = _macd_prev > 0 and _macd_now < 0  # 양→음 전환

                    # RSI 다이버전스: 가격 고점 갱신하는데 RSI는 오히려 낮아짐
                    _rsi_diverge = _rsi_now > 70 and _rsi_now < _rsi_prev - 3

                    if _rsi_overbought and _macd_turning:
                        from backend.core import sim_log
                        sim_log.push(self.agent_id, f"[차트모니터링] {ticker} RSI={_rsi_now:.1f}(과매수)+MACD반전→조기익절신호", "SELL")
                        return "sell", round(max(prob, 0.6), 4)

                    if _rsi_diverge and _macd_turning:
                        from backend.core import sim_log
                        sim_log.push(self.agent_id, f"[차트모니터링] {ticker} RSI다이버전스({_rsi_prev:.1f}→{_rsi_now:.1f})+MACD반전→익절신호", "SELL")
                        return "sell", round(max(prob, 0.55), 4)

                    # ── 저항선 근처 + 약세 캔들 → 익절 (저항선에서 팔아야 함) ──
                    # 실전: 저항선 2% 이내에 도달하면 매도 준비. 약세 봉 출현 시 즉시 익절
                    try:
                        _near_res = bool(full_df["near_resistance_2pct"].iloc[-1])
                        _bear_cnt = float(full_df["bearish_count_3"].iloc[-1])
                        _bear_engulf = bool(full_df["is_bearish_engulf"].iloc[-1]) if "is_bearish_engulf" in full_df.columns else False
                        _shooting   = bool(full_df["is_shooting_star"].iloc[-1]) if "is_shooting_star" in full_df.columns else False
                        if _near_res and (_bear_cnt >= 2 or _bear_engulf or _shooting):
                            from backend.core import sim_log
                            sim_log.push(self.agent_id, f"[차트모니터링] {ticker} 저항선근처({float(full_df['swing_near_resistance'].iloc[-1])*100:.1f}%)+약세봉→익절신호", "SELL")
                            return "sell", round(max(prob, 0.55), 4)
                    except Exception:
                        pass

                    # ── 3봉 연속 음봉 + 거래량 증가 → 추세 전환, 즉시 익절 ──────
                    # 실전: 산 뒤 계속 차트 보다가 3봉 연속 음봉이면 추세 꺾인 것
                    try:
                        _bear3 = float(full_df["bearish_count_3"].iloc[-1])
                        _vol_now = float(full_df["vol_ratio"].iloc[-1])
                        if _bear3 >= 3 and _vol_now >= 1.2:
                            from backend.core import sim_log
                            sim_log.push(self.agent_id, f"[차트모니터링] {ticker} 3봉연속음봉+거래량급증({_vol_now:.1f}x)→추세전환 익절", "SELL")
                            return "sell", round(max(prob, 0.52), 4)
                    except Exception:
                        pass

                    # ── 주식 전용: ORB 하단이탈 손절 + 장중반전 익절 ────────────
                    if self.market == "stock":
                        try:
                            _orb_br = float(full_df["orb_breakout"].iloc[-1]) if "orb_breakout" in full_df.columns else 0.0
                            if _orb_br == -1:
                                from backend.core import sim_log
                                sim_log.push(self.agent_id, f"[차트모니터링] {ticker} ORB하단이탈→주식손절", "SELL")
                                return "sell", round(max(prob, 0.52), 4)
                        except Exception:
                            pass
                        try:
                            _rev_s = bool(full_df["intraday_reversal"].iloc[-1]) if "intraday_reversal" in full_df.columns else False
                            _sess_s = int(full_df["session_phase"].iloc[-1]) if "session_phase" in full_df.columns else 0
                            if _rev_s and _sess_s >= 2:
                                from backend.core import sim_log
                                sim_log.push(self.agent_id, f"[차트모니터링] {ticker} 장중반전신호(오전강세→오후약세)→주식익절", "SELL")
                                return "sell", round(max(prob, 0.52), 4)
                        except Exception:
                            pass
            except Exception:
                pass

        # ── ADX 횡보장 필터: 코인 전용 (ETH/BTC 15-22 구간 손실 집중 데이터 기반)
        # 주식은 ADX 특성이 달라 적용 제외 — 주식 hold 원인은 chart analysis에서 관리
        try:
            _adx_val = float(full_df.get("adx_14", pd.Series([25.0])).iloc[-1]) if "adx_14" in full_df.columns else 25.0
            if _adx_val < 18.0 and self.market == "coin":
                _strong_reversal = (
                    bool(full_df.get("double_bottom_signal", pd.Series([0])).iloc[-1])
                    or bool(full_df.get("exhaustion_bounce", pd.Series([0])).iloc[-1])
                    or bool(full_df.get("fib_618_support", pd.Series([0])).iloc[-1])
                    or bool(full_df.get("near_support_3pct", pd.Series([0])).iloc[-1])
                ) if not full_df.empty else False
                if not _strong_reversal:
                    self._last_hold_reason = f"adx_low:{_adx_val:.1f}"
                    prob = min(prob, 0.52)  # buy_threshold 통상 0.65+ → 사실상 hold
        except Exception:
            pass

        # ── 차트 종합 분석 진입 필터 (캔들 + 지지선 + 거래량 + 다중봉 확인) ────────
        # 실전 원칙: 캔들 하나만 보지 말고 ① 지지선 위치 ② 거래량 ③ 여러 봉의 흐름 동시 확인
        # 연구 결과: 최소 3번 터치된 수평 지지선 근처 + 거래량 급증 + 2봉 이상 양봉 = 고승률 진입
        try:
            _last_row = full_df.iloc[-1]
            def _cf(col: str) -> bool:
                return bool(_last_row.get(col, 0))
            def _fv(col: str, default: float = 0.0) -> float:
                return float(_last_row.get(col, default))

            # ── ① 강세 캔들 패턴 (단봉 시그널) ───────────────────────────
            _bullish_candle = (
                _cf("is_hammer") or            # 망치형: 아래꼬리 지지 반등
                _cf("is_inverted_hammer") or   # 역망치형: 저점 반전 시도
                _cf("is_bullish_engulf") or    # 상승장악형: 이전 음봉 완전 덮음
                _cf("is_morning_star") or      # 아침별형: 3봉 반전 패턴
                _cf("is_piercing") or          # 관통형: 이전 음봉 50% 이상 회복
                _cf("is_marubozu_bull")        # 마루보쥬 양봉: 꼬리없는 강한 양봉
            )
            # ── ② 일목균형표 조건 (기준선/구름대 기반 추세 확인) ──────────
            _ichi_ok = _cf("ichi_tenkan_kijun_bull") or _cf("ichi_above_cloud")
            # ── ③ MA 정배열 + HA 연속 양봉 (추세 지속 모멘텀) ────────────
            _trend_ok = _cf("ma20_cross_ma60") and _fv("ha_bull_streak") >= 3
            # ── ④ 눌림목: 정배열에서 MA20 근처 눌린 후 반등 (실전 핵심 매수 자리) ──
            _pullback_ok = _cf("is_pullback_to_ma20")
            # ── ⑤ BB 하단 반등: 과매도 구간 (BB 25% 이하) ───────────────
            _bb_bounce = _fv("bb_pband", 0.5) < 0.25
            # ── ⑥ 거래량 반전 패턴: 감소→급증 (매도 소진, 반등 임박) ───────
            _vol_reversal = _cf("vol_reversal_signal")
            # ── ⑦ MA20/MA60 회복봉: 이평선 아래서 위로 돌파 (지지선 재확인) ─
            _ma_reclaim = _cf("close_reclaimed_ma20") or _cf("close_reclaimed_ma60")
            # ── ⑧ 동적 지지선 근처: 최근 20봉 저점 3% 이내 (수평 지지선 근접) ─
            _near_support = _cf("near_support_3pct")
            # ── ⑨ 3봉 방향 확인: 최근 3봉 중 양봉 2개 이상 (3틱룰) ──────
            _multi_bull = _fv("bullish_count_3") >= 2
            # ── ⑩ 이중바닥 W 패턴 완성 (40봉 내 두 유사저점+넥라인 돌파) ─
            _double_bottom = _cf("double_bottom_signal")
            # ── ⑪ 피보나치 지지 근접 (61.8% = 가장 강력, 38.2%/50% 포함) ──
            _fib_support = _cf("fib_618_support") or _cf("fib_50_support") or _cf("fib_382_support")
            # ── ⑫ 완전 정배열 (MA5>MA20>MA60 동시) ─────────────────────
            _ma_align_bull = _cf("ma_bull_align")
            # ── ⑬ 매도소진 고신뢰 반등 (거래량반전+RSI<35) ───────────────
            _exhaustion_buy = _cf("exhaustion_bounce")
            # ── ⑭ 지지선 기울기 양수 (상승 지지선 구조 확인) ────────────
            _sup_slope_up = _fv("support_slope") > 0.0

            # ── 약세 캔들 → 신규 진입 직접 차단 (저항선 근처 약세는 더 강하게)
            _bearish_candle = _cf("is_shooting_star") or _cf("is_bearish_engulf")
            _near_resistance = _cf("near_resistance_2pct")
            _ma_align_bear  = _cf("ma_bear_align")
            if _bearish_candle:
                _penalty = 0.30 if _near_resistance else 0.40
                self._last_hold_reason = "bearish_candle" + ("+near_res" if _near_resistance else "")
                return "hold", round(prob * _penalty, 4)

            # ── 완전 역배열 + 가짜돌파 경고 → 강한 매수 차단 ──────────
            if _ma_align_bear and _cf("false_breakout_warn"):
                self._last_hold_reason = "ma_bear+false_brk"
                return "hold", round(prob * 0.25, 4)

            # ── 저항선 2% 이내 + 음봉 추세 → 매수 차단 (벽 앞에서 들어가면 안됨)
            if _near_resistance and _fv("bearish_count_3") >= 2:
                self._last_hold_reason = "near_res+bear3"
                return "hold", round(prob * 0.45, 4)

            # ── 가짜 돌파 경고 단독 → 확률 약화 ────────────────────────
            if _cf("false_breakout_warn") and not (_near_support or _fib_support):
                self._last_hold_reason = "false_brk"
                return "hold", round(prob * 0.60, 4)

            # ── 상승 근거가 하나도 없으면 매수 차단 (캔들 하나만 보면 안됨)
            _has_signal = (
                _bullish_candle or _ichi_ok or _trend_ok or
                _pullback_ok or _bb_bounce or _vol_reversal or
                _ma_reclaim or _double_bottom or _fib_support or
                _exhaustion_buy
            )
            if not _has_signal:
                self._last_hold_reason = "no_signal"
                return "hold", round(prob * 0.55, 4)

            # ── 이중바닥 + 매도소진 → 확률 강화 (고승률 조합) ──────────
            if _double_bottom and _exhaustion_buy:
                prob = min(prob * 1.12, 0.95)
            elif _double_bottom or _exhaustion_buy:
                prob = min(prob * 1.06, 0.95)

            # ── 완전 정배열 + 피보나치 지지 + 양봉 → 최고 신뢰 진입 ────
            if _ma_align_bull and _fib_support and _bullish_candle:
                prob = min(prob * 1.10, 0.95)

            # ── 신호는 있지만 지지선 근처도 아니고 거래량도 없으면 약화 ──
            # 실전: 지지선과 거래량이 없는 캔들 패턴은 속임수일 가능성 높음
            _vol_ok = _fv("vol_ratio", 1.0) >= 1.1
            _context_ok = (
                _near_support or _multi_bull or _vol_ok or _vol_reversal or
                _fib_support or _double_bottom or _sup_slope_up
            )
            if _bullish_candle and not _context_ok:
                return "hold", round(prob * 0.70, 4)

            # ── 거래량 미동반 캔들만 단독: 신호 강도 약화 ─────────────────
            if _bullish_candle and not _vol_ok and not _near_support and not _fib_support:
                return "hold", round(prob * 0.75, 4)

        except Exception:
            pass

        # ── 주식 전용 진입 필터: ORB + VWAP + 세션 + KOSPI 상대강도 ─────────────
        # 실전 원칙: ① ORB 상단 돌파 ② VWAP 위 ③ KOSPI 강세 종목 ④ 마감 전 신규 비매수
        if self.market == "stock":
            try:
                _last_s = full_df.iloc[-1]
                def _ssf(col: str, default: float = 0.0) -> float:
                    return float(_last_s.get(col, default))
                def _ssb(col: str) -> bool:
                    return bool(_last_s.get(col, 0))

                _sess = int(_ssf("session_phase"))

                # 마감 구간 (session_phase==3: 14:30~) 신규 매수 완전 차단
                if _sess == 3:
                    return "hold", round(prob * 0.25, 4)

                # 개장 초변동 (session_phase==0: 09:00~09:30): 범위 확정 전 확률 약화
                if _sess == 0:
                    prob = prob * 0.75

                # ORB 하단 돌파 중 → 하락 압력, 신규 진입 완전 차단
                _orb_break = _ssf("orb_breakout")
                if _orb_break == -1:
                    return "hold", round(prob * 0.20, 4)

                _orb_bull = (_orb_break == 1)           # ORB 상단 돌파 (강세 추세)
                _orb_vol  = _ssb("orb_vol_confirm")     # ORB + 거래량 1.5배 동반

                # VWAP 기준선: 기관의 매수 기준선, 아래면 분위기 나쁨
                _above_vwap = _ssb("anchored_vwap_cross")

                # KOSPI 상대강도 + 갭 방향
                _kospi_rel = _ssf("kospi_rel_str")
                _ret_open  = _ssf("ret_since_open")

                # VWAP 아래 + ORB 돌파 없음 → 추세 역방향 진입 차단
                if not _above_vwap and not _orb_bull:
                    return "hold", round(prob * 0.40, 4)

                # KOSPI 역행 + 갭다운 → 시장 전체 약세, 개별주 진입 차단
                if _kospi_rel < -0.005 and _ret_open < -0.01:
                    return "hold", round(prob * 0.45, 4)

                # ORB 상단돌파 + 거래량 확인 = 주식 단타 최고 신뢰 진입 신호
                if _orb_bull and _orb_vol:
                    prob = min(prob * 1.15, 0.95)

                # VWAP 위 + ORB 돌파 = 추세 방향 더블 확인
                if _above_vwap and _orb_bull:
                    prob = min(prob * 1.08, 0.95)

            except Exception:
                pass

        if prob >= self.adaptive_buy_threshold:
            # ── 손실 패턴 패널티: 과거 동일 신호 조합에서 반복 손실 → 확률 하향 ──
            _lp = self._loss_penalty()
            if _lp > 0:
                prob = max(0.01, prob * (1 - _lp))
                logger.debug("[%s] 손실패턴 패널티 %.0f%% → prob=%.3f", self.agent_id, _lp * 100, prob)
                if prob < self.adaptive_buy_threshold:
                    return "hold", round(prob, 4)
            # ── Meta-Labeling 2차 필터 ───────────────────────────────
            # 1차 BUY 신호를 실제로 거래할지 2차 모델이 판단 (False Positive 감소)
            if self._meta_model is not None and self._meta_scaler is not None:
                try:
                    _h = datetime.now(ZoneInfo("Asia/Seoul")).hour
                    _meta_feat = np.array([[
                        prob,
                        np.sin(2 * np.pi * _h / 24),
                        np.cos(2 * np.pi * _h / 24),
                        self._last_adx_14,
                        self._last_vol_ratio,
                    ]])
                    _meta_prob = float(
                        self._meta_model.predict_proba(self._meta_scaler.transform(_meta_feat))[0, 1]
                    )
                    if _meta_prob < 0.5:
                        return "hold", round(prob, 4)  # 메타 모델이 거부
                except Exception:
                    pass
            return "buy", round(prob, 4)
        if prob <= (1.0 - self.buy_threshold):
            return "sell", round(prob, 4)
        self._last_hold_reason = f"prob:{prob:.3f}<thr:{self.buy_threshold}"
        return "hold", round(prob, 4)

    async def predict_live(self, ohlcv_list: list[dict], ticker: str | None = None) -> tuple[str, float]:
        """실시간 보정 포함 예측 — 실매매 게이트 전용.

        기본 predict() 확률에 공포탐욕·김치프리미엄·펀딩비·호가창·기관외국인을 실시간 반영.
        """
        if not ohlcv_list:
            return "hold", 0.5

        _, prob = self.predict(
            ohlcv_list,
            oi_hist=self._cached_oi_hist or None,
            taker_hist=self._cached_taker_hist or None,
            obi_snaps=None,
        )

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
                # 펀딩비: 양수(롱 쏠림) → 확률 하향 / 강한 음수(숏 쏠림) → 반등 가능성 → 소폭 상향
                # 연구: 펀딩 >+50% APR 3일+ = 롱 과열 징후, <-50% APR = 바닥 신호
                if funding > 0.002:  # +0.2% 이상 롱 쏠림
                    prob = max(0.01, prob - min(funding * 15, 0.04))
                elif funding < -0.003:  # -0.3% 이하 강한 숏 쏠림 → 반등 가능
                    prob = min(0.99, prob + min(abs(funding) * 8, 0.02))
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
                _inst = inv.get("institution_net_buy", 0)
                _frgn = inv.get("foreign_net_buy", 0)
                _net = _inst + _frgn
                _total = abs(_inst) + abs(_frgn) + 1
                _net_ratio = max(-1.0, min(1.0, _net / _total))
                prob = max(0.01, min(0.99, prob + _net_ratio * 0.04))
            except Exception:
                pass

        if prob >= self.adaptive_buy_threshold:
            return "buy", round(prob, 4)
        if prob <= (1.0 - self.adaptive_buy_threshold):
            return "sell", round(prob, 4)
        return "hold", round(prob, 4)

    # ── 가상 매수/매도 ───────────────────────────────────────────

    def virtual_buy(self, ticker: str, price: float, portion: float = 1.0) -> AgentTrade | None:
        if not self.is_active:
            return None
        if ticker in self._positions:
            return None
        if len(self._positions) >= MAX_OPEN_POSITIONS:
            return None
        amount = self._balance * self._kelly_ratio * min(max(portion, 0.1), 1.0)  # DCA 비율 반영
        if amount < 5_000:
            return None
        actual_price = price * 1.0005  # 슬리피지 0.05% 반영 (실제 체결가 불리 보정)
        qty = amount / actual_price
        price = actual_price  # 이하 price 변수를 실제 체결가로 통일
        self._balance -= amount
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")
        self._positions[ticker] = AgentPosition(ticker, price, qty, now)
        trade = AgentTrade(self.agent_id, ticker, "BUY", price, qty, price, None, self._balance, now)
        self._push_recent(trade)
        return trade

    def virtual_sell(self, ticker: str, price: float) -> AgentTrade | None:
        pos = self._positions.pop(ticker, None)
        if pos is None:
            return None
        actual_sell = price * 0.9995  # 매도 수수료 0.05% (Upbit/KIS 기준)
        proceeds = pos.qty * actual_sell
        profit_rate = (actual_sell - pos.entry_price) / pos.entry_price
        self._balance += proceeds
        self.total_trades += 1
        if profit_rate > 0:
            self.win_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 5:
                today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
                if self._last_retrain_date != today:
                    self._daily_retrain_count = 0
                    self._last_retrain_date = today
                if self._daily_retrain_count < 3:
                    self.needs_retrain = True  # 5연속 손실 → 즉시 재학습 (하루 3회 제한)
        self._peak_price.pop(ticker, None)
        self._trailing_mode.discard(ticker)
        self._partial_tp_price.pop(ticker, None)
        self._partial_tp_done.discard(ticker)
        self._buy_context.pop(ticker, None)  # 수익/손실 불문 매도 시 컨텍스트 정리 (메모리 누수 방지)
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")
        trade = AgentTrade(self.agent_id, ticker, "SELL", price, pos.qty, pos.entry_price, round(profit_rate, 4), self._balance, now)
        self._push_recent(trade)
        return trade

    def record_buy_context(self, symbol: str) -> None:
        """매수 시 신호 스냅샷 저장 (이후 손실 발생 시 소급 분석용)."""
        self._buy_context[symbol] = dict(self._last_signal_snapshot)

    def analyze_loss(self, symbol: str, profit_rate: float, held_min: float) -> dict:
        """손실 거래 소급 분석: 매수 당시 어떤 신호/전략이 실패했는지 판단.

        Returns:
            분석 딕셔너리 (wrong_signals, causes, pattern_repeat 등)
        """
        ctx = self._buy_context.pop(symbol, {})
        rsi  = ctx.get("rsi_9",       50.0)
        adx  = ctx.get("adx_14",      20.0)
        bb   = ctx.get("bb_pband",     0.5)
        mtf  = ctx.get("mtf_align",    1.5)
        reg  = ctx.get("regime_state", 1.0)
        vol  = ctx.get("vol_ratio",    1.0)
        macd = ctx.get("macd_diff",    0.0)

        wrong: list[str] = []
        cause: list[str] = []

        if rsi > 65:
            wrong.append("rsi_overbought")
            cause.append(f"RSI={rsi:.0f} 과매수 진입 (반전 위험)")
        if adx < 18:
            wrong.append("low_adx")
            cause.append(f"ADX={adx:.0f} 추세 약함 (횡보장 매수)")
        if mtf < 2.0:
            wrong.append("mtf_misalign")
            cause.append(f"MTF정렬={mtf:.1f} (다중TF 하락 불일치)")
        if bb > 0.85:
            wrong.append("bb_upper")
            cause.append(f"BB위치={bb:.2f} 볼린저 상단 돌파")
        if vol < 0.8:
            wrong.append("low_volume")
            cause.append(f"거래량비율={vol:.2f} (약한 신호 신뢰도)")
        if reg == 0.0:
            wrong.append("bear_regime")
            cause.append("레짐=하락장 (매수 억제 구간 진입)")
        if macd < 0:
            wrong.append("macd_negative")
            cause.append(f"MACD={macd:.5f} 음수 (모멘텀 약세)")

        if held_min < 30:
            cause.append(f"보유{held_min:.0f}분 → 노이즈 신호 손절")
        elif held_min > 240:
            cause.append(f"보유{held_min:.0f}분 → 알파소멸 후 물림")

        _key = frozenset(wrong)  # frozenset: hashable + set 연산 직접 가능 (tuple+set변환 불필요)
        self._loss_patterns[_key] = self._loss_patterns.get(_key, 0) + 1

        analysis = {
            "agent_id":       self.agent_id,
            "symbol":         symbol,
            "loss_pct":       round(profit_rate * 100, 2),
            "held_min":       round(held_min, 1),
            "buy_ctx":        ctx,
            "wrong_signals":  wrong,
            "causes":         cause,
            "pattern_repeat": self._loss_patterns[_key],
        }
        self._loss_analysis_log = ([analysis] + self._loss_analysis_log)[:30]
        return analysis

    def _loss_penalty(self) -> float:
        """현재 신호 조합이 과거 손실 패턴과 2개 이상 겹치면 패널티 반환 (0.0 ~ 0.50)."""
        if not self._loss_patterns or not self._last_signal_snapshot:
            return 0.0
        snap = self._last_signal_snapshot
        current: set[str] = set()
        if snap.get("rsi_9",       50.0) > 65:   current.add("rsi_overbought")
        if snap.get("adx_14",      20.0) < 18:   current.add("low_adx")
        if snap.get("mtf_align",    1.5) < 2.0:  current.add("mtf_misalign")
        if snap.get("bb_pband",     0.5) > 0.85: current.add("bb_upper")
        if snap.get("vol_ratio",    1.0) < 0.8:  current.add("low_volume")
        if snap.get("regime_state", 1.0) == 0.0: current.add("bear_regime")
        if snap.get("macd_diff",    0.0) < 0:    current.add("macd_negative")

        max_penalty = 0.0
        for pattern_key, count in self._loss_patterns.items():
            overlap = len(current & pattern_key)  # pattern_key는 frozenset → 직접 set 교집합
            if overlap >= 2:
                penalty = min(0.50, overlap * 0.10 * min(count, 4))
                max_penalty = max(max_penalty, penalty)
        return max_penalty

    def virtual_partial_sell(self, ticker: str, price: float, ratio: float = 0.40) -> AgentTrade | None:
        """부분 청산 (Scale-out): ratio 비율만큼 매도하고 나머지는 계속 보유.

        잔량 pos.qty * (1-ratio) 가 _positions에 남으므로 이후 full sell 가능.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return None
        ratio = min(max(ratio, 0.0), 1.0)  # 0~1 클램프
        sell_qty = pos.qty * ratio
        if sell_qty <= 0:
            return None
        actual_sell = price * 0.9995
        proceeds    = sell_qty * actual_sell
        profit_rate = (actual_sell - pos.entry_price) / pos.entry_price
        self._balance += proceeds
        pos.qty -= sell_qty             # 잔량 갱신 (포지션 유지)
        self._partial_tp_done.add(ticker)
        now   = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")
        trade = AgentTrade(self.agent_id, ticker, "SELL_PARTIAL", price, sell_qty, pos.entry_price, round(profit_rate, 4), self._balance, now)
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
        self._balance      = float(stats.get("current_balance") or INITIAL_CAPITAL)
        # SQLite INTEGER → Python int 보장 (일부 드라이버에서 str 반환 가능)
        self.total_trades  = int(stats.get("total_trades") or 0)
        self.win_trades    = int(stats.get("win_trades") or 0)
        self.is_champion   = bool(stats.get("is_champion", 0))
        self.is_active     = bool(stats.get("is_active", 1))
        # 자동 조정된 임계값 복구 (0이거나 없으면 AGENT_CONFIGS 기본값 유지, 상한 0.72 클램핑)
        thr = stats.get("buy_threshold")
        if thr is not None and float(thr) > 0:
            self.buy_threshold = min(float(thr), 0.72)  # DB 과거 높은 임계값 상한 강제 적용
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

        # 재시작 시 recent_trades에서 연속손실 복구 (BUY 레코드는 profit_rate=NULL → 스킵)
        consecutive = 0
        for t in self.recent_trades:
            if t.get("action", "SELL") not in ("SELL", "SELL_PARTIAL"):
                continue
            if (t.get("profit_rate") or 0) < 0:
                consecutive += 1
            else:
                break
        self._consecutive_losses = consecutive
        # 5연속 손실 상태 유지: needs_retrain은 인메모리라 재시작 시 소실
        # → consecutive 복원 후 재설정 (하루 3회 한도 체크는 생략, 재학습 기회 제공)
        if consecutive >= 5:
            self.needs_retrain = True

        # 재시작 시 최근 SL 거래 기준 쿨다운 복구 (30분 이내 -0.3% 이하 손절)
        from datetime import datetime as _dt, timedelta as _td
        now_kst = _dt.now(ZoneInfo("Asia/Seoul"))
        for t in self.recent_trades:
            if (t.get("profit_rate") or 0) >= -0.003:
                continue
            traded_at_str = t.get("traded_at", "")
            if not traded_at_str:
                continue
            try:
                traded_at = _dt.fromisoformat(traded_at_str).replace(tzinfo=ZoneInfo("Asia/Seoul"))
                if (now_kst - traded_at).total_seconds() < 1800:  # 30분 이내
                    ticker = t.get("ticker", "")
                    if ticker:
                        self._cooldown_tickers[ticker] = (traded_at + _td(minutes=30)).isoformat()
            except Exception:
                pass

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
        # 일일 P&L: day_start_balance가 0이면 미초기화 (첫 실행 시) → total_value 기준 0%
        _day_start = self.day_start_balance if self.day_start_balance > 0 else total_value
        _today_return_pct = round((total_value - _day_start) / _day_start * 100, 2) if _day_start > 0 else 0.0
        return {
            "agent_id":          self.agent_id,
            "market":            self.market,
            "interval_min":      self.interval_min,
            "label_threshold":   self.label_threshold,
            "buy_threshold":     self.buy_threshold,
            "feature_set":       self.feature_set,
            "balance":           round(self._balance, 0),
            "position_value":    round(pos_value, 0),
            "total_value":       round(total_value, 0),
            "total_return_pct":  round(self.total_return * 100, 2),
            "today_return_pct":  _today_return_pct,
            "day_start_balance": round(_day_start, 0),
            "win_rate":          round(self.win_rate * 100, 1),
            "total_trades":      self.total_trades,
            "win_trades":        self.win_trades,
            "today_buy_count":   self._today_buy_count,
            "blacklist_count":   len(self._ticker_blacklist),
            "is_champion":       self.is_champion,
            "is_active":         self.is_active,
            "trained_at":        self._trained_at,
            "positions":         positions_detail,
            "recent_trades":     self.recent_trades[:20],
        }


# ── 싱글톤 에이전트 풀 ────────────────────────────────────────────

def build_agents() -> dict[str, SimAgent]:
    return {cfg[0]: SimAgent(*cfg) for cfg in AGENT_CONFIGS}


AGENTS: dict[str, SimAgent] = build_agents()

# ── 앙상블 그림자 에이전트 — 실매매 앙상블 신호를 가상 포트폴리오로 추적 ──────
# predict_ensemble() 호출 결과를 동일 TP/SL 로직으로 가상 매매해 앙상블 기대수익 검증
ENSEMBLE_AGENTS: dict[str, SimAgent] = {
    "ENSEMBLE_COIN":  SimAgent("ENSEMBLE_COIN",  60, 0.010, 0.65, "all", "coin",  5, "lgbm"),
    "ENSEMBLE_STOCK": SimAgent("ENSEMBLE_STOCK", 15, 0.007, 0.60, "all", "stock", 5, "lgbm"),
}


def reset_all_agents() -> None:
    """모든 에이전트(가상+앙상블) 메모리 초기화 — 잔액 1M(100만원), 거래기록 0, 포지션 없음."""
    for agent in list(AGENTS.values()) + list(ENSEMBLE_AGENTS.values()):
        agent._balance = INITIAL_CAPITAL
        agent._positions.clear()
        agent._last_position_values.clear()
        agent.total_trades = 0
        agent.win_trades   = 0
        agent.recent_trades.clear()
        agent.is_champion  = False
        agent.is_active    = True


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
    btc_ohlcv: list[dict] | None = None,
    kospi_ohlcv: list[dict] | None = None,
    oi_hist: list[dict] | None = None,
    taker_hist: list[dict] | None = None,
    ls_hist: list[dict] | None = None,
    obi_snaps: list[float] | None = None,
    ohlcv_by_interval: dict | None = None,  # {interval_min: ohlcv_list} — 봉 타입별 OHLCV
) -> tuple[str, float]:
    """가중 앙상블 게이트 — 전 에이전트 동적 투표 + 실시간 보정.

    가중치 = 승률 × (1 + 총수익률).
    지금 잘하는 전략이 자동으로 발언권이 커져서 장세에 자동 적응.
    ohlcv_by_interval: {15: [...], 60: [...]} 제공 시 각 에이전트가 자기 봉 타입 OHLCV 사용.
    미제공 시 ohlcv_list로 폴백 (기존 동작).
    """
    if not ohlcv_list:
        return "hold", 0.5
    candidates = [
        a for a in AGENTS.values()
        if a.market == market and a._model is not None
        # 승률 40% 미만 (10거래 이상 검증된 경우) — 초기 30건 기준은 나쁜 에이전트를 너무 오래 포함시킴
        and not (a.total_trades >= 10 and a.win_rate < 0.40)
        # 수익률 -2% 이하 (20거래 이상 검증된 경우)
        and not (a.total_trades >= 20 and a.total_return < -0.02)
    ]
    if not candidates:
        # 폴백: 수익률 기준만 유지, 모델 있는 에이전트 사용
        candidates = [
            a for a in AGENTS.values()
            if a.market == market and a._model is not None
            and not (a.total_trades >= 20 and a.total_return < -0.03)
        ]
    if not candidates:
        candidates = [a for a in AGENTS.values() if a.market == market and a._model is not None]
    if not candidates:
        return "hold", 0.5

    # ── 현재 시장 레짐 감지 (앙상블 가중치 조정용) ───────────────
    current_regime = 1  # 기본: 횡보장
    try:
        _interval = candidates[0].interval_min if candidates else 5
        _tmp = compute_features(ohlcv_list, interval_min=_interval)
        if not _tmp.empty and "regime_state" in _tmp.columns:
            current_regime = int(_tmp["regime_state"].iloc[-1])
    except Exception:
        pass

    # ── 가중 예측 집계 (샤프비율 × 수익률 × 레짐 부스터) ─────────
    weighted_prob    = 0.0
    weighted_thr     = 0.0  # 임계값도 동일 가중치로 집계 (단순평균 불일치 방지)
    total_weight     = 0.0
    buy_votes        = 0    # 개별 에이전트 "buy" 투표 수
    buy_weighted_prob = 0.0  # buy 투표 에이전트만의 가중 확률 합
    buy_total_weight  = 0.0  # buy 투표 에이전트의 가중치 합
    _agent_results: list[tuple[str, str, float, str]] = []  # (aid, sig, prob, hold_reason)
    for agent in candidates:
        ret    = max(agent.total_return, -0.5)
        sharpe = agent._sharpe_weight()
        weight = max(sharpe * max(1.0 + ret, 0.1), 0.1)
        # 고승률 에이전트 발언권 강화 (30거래 이상 검증된 경우만)
        if agent.total_trades >= 30:
            if agent.win_rate >= 0.60:
                weight *= 2.0
            elif agent.win_rate >= 0.55:
                weight *= 1.5
            elif agent.win_rate >= 0.50:
                weight *= 1.2

        # 레짐별 전략 가중치 조정 (regime_state: 0=하락, 1=횡보, 2=상승, 3=고변동)
        if current_regime == 2:    # 상승추세: trend 우대, volume 축소
            if agent.feature_set == "trend":
                weight *= 1.5
            elif agent.feature_set == "volume":
                weight *= 0.7
        elif current_regime == 1:  # 횡보: momentum 우대, trend 축소
            if agent.feature_set == "momentum":
                weight *= 1.4
            elif agent.feature_set == "trend":
                weight *= 0.8
        elif current_regime == 3:  # 고변동: all 우대, 전체 보수적
            weight *= 0.7
            if agent.feature_set == "all":
                weight *= 1.3
        elif current_regime == 0:  # 하락추세: 전체 보수적 축소 (predict()도 이미 0.3 억제)
            weight *= 0.5

        # 에이전트 봉 타입에 맞는 OHLCV 선택 (15분봉 에이전트에 60분봉 줬을 때 hold 고착 방지)
        _agent_ohlcv = (ohlcv_by_interval or {}).get(agent.interval_min, ohlcv_list) or ohlcv_list
        agent._last_hold_reason = ""
        sig, prob = agent.predict(_agent_ohlcv, ticker=ticker, btc_ohlcv=btc_ohlcv, kospi_ohlcv=kospi_ohlcv, oi_hist=oi_hist, taker_hist=taker_hist, ls_hist=ls_hist, obi_snaps=obi_snaps)
        _agent_results.append((agent.agent_id, sig, round(prob, 3), agent._last_hold_reason))
        weighted_prob += prob * weight
        weighted_thr  += agent.buy_threshold * weight
        total_weight  += weight
        if sig == "buy":
            buy_votes        += 1
            buy_weighted_prob += prob * weight
            buy_total_weight  += weight

    final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
    # buy 투표 에이전트만의 평균 확률 — hold 에이전트가 끌어내리지 않는 진짜 매수 신호 강도
    buy_avg_prob = buy_weighted_prob / buy_total_weight if buy_total_weight > 0 else 0.0

    # ── 실시간 보정 (시장 데이터 1회씩만 조회) ──────────────────
    if market == "coin" and ticker:
        try:
            from backend.ml.model import _get_fear_greed
            fg = await _get_fear_greed()
            if fg > 80:    # 극단 탐욕: 과열 → 신규 롱 강한 억제
                final_prob = max(0.01, final_prob * 0.65)
            elif fg < 20:  # 극단 공포: 역추세 매수 기회 → 소폭 강화
                final_prob = min(0.99, final_prob * 1.20)
            else:           # 일반 구간: 선형 감소 (탐욕일수록 조심)
                final_prob = max(0.01, min(0.99, final_prob - (fg - 50) * 0.001))
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
            _inst = inv.get("institution_net_buy", 0)
            _frgn = inv.get("foreign_net_buy", 0)
            _net = _inst + _frgn
            _total = abs(_inst) + abs(_frgn) + 1
            _net_ratio = max(-1.0, min(1.0, _net / _total))
            final_prob = max(0.01, min(0.99, final_prob + _net_ratio * 0.04))
        except Exception:
            pass

    # ── 임계값: 개별 임계값 가중평균 × 0.98 (수익률 중심 — 진입 기준 상향)
    avg_thr = (weighted_thr / total_weight) * 0.98 if total_weight > 0 else 0.62
    vote_ratio = buy_votes / len(candidates) if candidates else 0.0

    # 투표 로그 (앙상블 진단용)
    logger.warning(
        "[앙상블-%s] ticker=%s final_prob=%.4f buy_avg=%.4f avg_thr=%.4f buy_votes=%d/%d",
        market, ticker or "-", final_prob, buy_avg_prob, avg_thr, buy_votes, len(candidates),
    )
    # buy_votes=0 시 hold 원인 분석 (원인 파악용 — 빈번 로그 방지: market/ticker당 1건)
    if buy_votes == 0 and _agent_results:
        _reason_summary = " | ".join(
            f"{aid}:{rsn or 'raw_low'}" for aid, sig, prob, rsn in _agent_results[:8]
        )
        logger.warning("[앙상블-%s][원인] ticker=%s 에이전트별 hold이유: %s", market, ticker or "-", _reason_summary)

    # ── 매수 판단: buy_avg_prob(buy 에이전트만의 평균) 기준 사용 ──────────────
    # final_prob(전체 에이전트 평균)은 hold 에이전트 8명이 끌어내려 항상 0.3~0.4 → 사용 불가
    # buy_avg_prob: 매수 신호 낸 에이전트들의 진짜 확신도
    if buy_votes >= 3 and buy_avg_prob >= 0.62:      # 강한 합의: 3명 이상, 각자 자신감 높음
        return "buy", round(buy_avg_prob, 4)
    if buy_votes >= 2 and buy_avg_prob >= 0.68:      # 소수 합의: 2명이지만 매우 자신 있을 때
        return "buy", round(buy_avg_prob, 4)
    if final_prob >= avg_thr:                        # 기존 조건 (전체 평균이 임계 초과시)
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
        active = [a for a in market_agents if a.total_trades >= 10 and a.win_trades > 0]
        if active:
            max(active, key=lambda a: a.total_return).is_champion = True
