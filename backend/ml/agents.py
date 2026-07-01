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

INITIAL_CAPITAL    = 10_000_000.0  # 에이전트당 초기 가상 자금 (1000만원)
POSITION_RATIO     = 0.5            # 잔액의 50%씩 사용
MAX_OPEN_POSITIONS = 3              # 에이전트당 동시 최대 포지션 수

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
    ],
    "trend": [
        "ma5_ratio", "ma20_ratio", "ma60_ratio",
        "ma5_cross_ma20", "ma20_cross_ma60",
        "adx_14", "adx_pos", "adx_neg",
        "trix_15", "dpo_20", "vortex_diff",
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
    ],
    "volume": [
        "vol_ratio", "obv_change", "cmf_20",
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
    ],
}

# ── 20개 에이전트 설정 (전체 5분봉, 중복 전략 없음) ─────────────
# (agent_id, interval_min, label_threshold, buy_threshold, feature_set, market, lookahead)
# lookahead: 3=단기(15분), 5=중기(25분), 8=장기(40분) — 앙상블 분산 극대화

AGENT_CONFIGS: list[tuple] = [
    # (agent_id, interval_min, label_threshold, buy_threshold, feature_set, market, lookahead, model_type)
    # label_threshold = 트리플배리어 TP/SL 기준 (ATR×1.5 실거래 손절 범위에 맞게 0.005~0.015 조정)
    # 코인 홀수 → LightGBM / 짝수 → XGBoost (앙상블 다양성 극대화)
    ("AI01",  5, 0.006, 0.58, "all",      "coin",  3, "lgbm"),  # 단기 공격형
    ("AI02",  5, 0.007, 0.63, "momentum", "coin",  5, "xgb"),
    ("AI03",  5, 0.008, 0.58, "trend",    "coin",  8, "lgbm"),  # 장기 추세형
    ("AI04",  5, 0.007, 0.62, "volume",   "coin",  3, "xgb"),
    ("AI05",  5, 0.010, 0.60, "all",      "coin",  5, "lgbm"),
    ("AI06",  5, 0.008, 0.65, "momentum", "coin",  8, "xgb"),
    ("AI07",  5, 0.010, 0.62, "trend",    "coin",  3, "lgbm"),
    ("AI08",  5, 0.007, 0.62, "volume",   "coin",  5, "xgb"),
    ("AI09",  5, 0.012, 0.65, "all",      "coin",  8, "lgbm"),
    ("AI10",  5, 0.008, 0.60, "trend",    "coin",  5, "xgb"),
    # 주식 전략 — 전부 LightGBM (histogram-based 분할이 불균형 5분봉 데이터에 더 안정적)
    ("AI11",  5, 0.006, 0.58, "all",      "stock", 3, "lgbm"),
    ("AI12",  5, 0.005, 0.60, "trend",    "stock", 5, "lgbm"),  # label 0.7→0.5% (양성레이블 확보)
    ("AI13",  5, 0.008, 0.58, "momentum", "stock", 8, "lgbm"),
    ("AI14",  5, 0.005, 0.60, "volume",   "stock", 3, "lgbm"),  # label 0.7→0.5%
    ("AI15",  5, 0.010, 0.60, "all",      "stock", 5, "lgbm"),
    ("AI16",  5, 0.005, 0.60, "trend",    "stock", 8, "lgbm"),  # label 0.8→0.5%
    ("AI17",  5, 0.010, 0.62, "momentum", "stock", 3, "lgbm"),
    ("AI18",  5, 0.006, 0.60, "volume",   "stock", 5, "lgbm"),  # label 1.0→0.6%
    ("AI19",  5, 0.012, 0.65, "all",      "stock", 8, "lgbm"),
    ("AI20",  5, 0.012, 0.60, "trend",    "stock", 3, "lgbm"),  # xgb→lgbm, 임계 0.70→0.60
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
        self._model_path = MODEL_DIR / f"{model_type}_agent_{agent_id}.pkl"
        self._train_lock = asyncio.Lock()   # 동시 재학습 race condition 방지

        self.total_trades = 0
        self.win_trades = 0
        self.is_champion = False
        self.is_active = True  # False면 매수 스킵 (자동 임계값 조정에서 관리)
        self.recent_trades: list[dict] = []
        self._last_position_values: dict[str, float] = {}  # ticker → 현재 평가액
        self._last_atr_pct: float = 0.0          # ATR 기반 동적 손익용 (predict()에서 업데이트)
        self._cached_funding_rates: list[dict] = []  # 재학습 시 업데이트, predict()에서 사용
        self._cached_oi_hist: list[dict] = []         # BTC OI 히스토리 캐시 (코인 전용)
        self._cached_taker_hist: list[dict] = []      # BTC Taker 비율 히스토리 캐시 (코인 전용)

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
            stored_feats = data.get("feature_names", [])
            if stored_feats and stored_feats != self.feature_names:
                return False  # 피처 불일치 → 재학습
            loaded = data["model"]
            # 모델 타입 불일치(예: xgb→lgbm 전환) 시 자동 폐기 → 다음 틱 재학습
            loaded_type = "lgbm" if isinstance(loaded, LGBMClassifier) else "xgb"
            if loaded_type != self.model_type:
                self._model_path.unlink(missing_ok=True)
                return False
            self._model = loaded
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

    def train(
        self,
        ohlcv_list: list[dict],
        funding_rates: list[dict] | None = None,
        btc_ohlcv: list[dict] | None = None,
        kospi_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
        trade_results: list[dict] | None = None,
    ) -> bool:
        """분봉 OHLCV로 전용 모델 학습."""
        try:
            feat_df = compute_features(
                ohlcv_list,
                funding_rates=funding_rates,
                btc_ohlcv=btc_ohlcv,
                kospi_ohlcv=kospi_ohlcv,
                oi_hist=oi_hist,
                taker_hist=taker_hist,
            )
            # ATR을 피처 필터링 전에 미리 추출 — 레이블 생성 시 실제 손익 기준과 정합하기 위해
            _atr_full = feat_df["atr_pct"].copy() if "atr_pct" in feat_df.columns else None
            # ADX 필터를 열 선택 전에 캡처 (feature_set 에 adx_14 없는 에이전트도 동일 필터 적용)
            _adx_mask = feat_df["adx_14"] >= 20 if "adx_14" in feat_df.columns else None
            feat_df = feat_df[[c for c in self.feature_names if c in feat_df.columns]].dropna()
            if _adx_mask is not None:
                feat_df = feat_df[_adx_mask.reindex(feat_df.index, fill_value=False)]
            # ATR도 feat_df 인덱스에 맞춰 정렬
            _atr_series = _atr_full.reindex(feat_df.index) if _atr_full is not None else None
            if len(feat_df) < 50:
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
                # 실제 _agent_execute 손익 기준 ATR×3.0 TP, ATR×1.5 SL 과 동일하게 맞춤
                if _atr_series is not None and pd.notna(_atr_series.iloc[i]) and float(_atr_series.iloc[i]) > 0:
                    _atr = float(_atr_series.iloc[i])
                    # ATR 기반 TP/SL — label_threshold 최소값 보장 (저변동성 구간 레이블 전멸 방지)
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

            # 수익률은 노이즈 필터용으로 유지
            next_close = close.shift(-LOOKAHEAD)
            ret = (next_close / close - 1).fillna(0)

            # 마지막 LOOKAHEAD 행 제거
            feat_df = feat_df.iloc[:-LOOKAHEAD]
            label   = label.iloc[:-LOOKAHEAD]
            ret     = ret.iloc[:-LOOKAHEAD]

            # 길이 보정
            min_len = min(len(feat_df), len(label))
            feat_df = feat_df.iloc[:min_len]
            label   = label.iloc[:min_len]
            ret     = ret.iloc[:min_len]

            # 노이즈 필터: label=0 중 LOOKAHEAD 후 수익률이 ±NOISE_FLOOR 미만인 구간 제외
            # label=1(TP 조기 발동 후 되돌림)은 유효한 매수 기회이므로 항상 보존
            clear_mask = (label == 1) | (ret.abs() >= NOISE_FLOOR)
            feat_df = feat_df[clear_mask.values]
            label   = label[clear_mask.values]

            if len(feat_df) < 30 or len(set(label.values)) < 2:
                return False

            X = feat_df.values
            y = label.values.astype(int)

            # ── 2-창 Purged Walk-Forward: 이전 레짐(창A) + 최신 레짐(창B) ──────
            # 데이터: [───훈련───][GAP][──창A──][GAP][──창B──]
            # 창A/B 모두 훈련셋 밖 → 데이터 리케이지 없음
            GAP      = self.lookahead
            VAL_SIZE = min(200, max(len(X) // 6, 30))
            _two_win = len(X) > 2 * VAL_SIZE + 2 * GAP + 50
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
                    ohlcv_dates = pd.to_datetime(
                        [d["date"] for d in reversed(ohlcv_list)]
                    ).round("5min")
                    ohlcv_dates_arr = ohlcv_dates.values
                    n_train = len(X_train)
                    for tr in trade_results:
                        try:
                            buy_dt = pd.Timestamp(tr["buy_at"]).round("5min").to_datetime64()
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
                clf = LGBMClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
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

            # Purged Walk-Forward 검증 — 정밀도(승률) 기반으로 평가
            if X_val is not None and len(X_val) > 0:
                _val_s    = scaler.transform(X_val)
                _val_prob = clf.predict_proba(_val_s)[:, 1]
                val_acc   = clf.score(_val_s, y_val)
                if val_acc < 0.50:
                    logger.warning("[%s] WF검증 %.1f%% < 50%% → 학습 실패", self.agent_id, val_acc * 100)
                    return False

                from sklearn.metrics import precision_score as _ps
                _init_thr = self.buy_threshold

                def _find_best_thr(prob_arr: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
                    best_thr, best_prec = _init_thr, 0.0
                    for _t in np.arange(0.50, 0.85, 0.01):
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
                        self.buy_threshold = round(min(max(_combined, 0.60), 0.75), 2)
                    logger.debug("[%s] 2창WF %.1f%% | 창A %.2f + 창B %.2f → %.2f",
                                 self.agent_id, val_acc * 100, thr_a, thr_b, self.buy_threshold)
                else:
                    if prec_b > 0:
                        self.buy_threshold = round(min(max(thr_b, 0.60), 0.75), 2)
                    logger.debug("[%s] WF검증 %.1f%% | 최적임계값 %.2f (정밀도 %.3f)",
                                 self.agent_id, val_acc * 100, self.buy_threshold, prec_b)

            # 검증 유무와 무관하게 최소 임계값 0.60 항상 보장 (초기값 0.58 에이전트 방어)
            self.buy_threshold = max(self.buy_threshold, 0.60)
            self._model = clf
            self._scaler = scaler
            self._trained_at = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            self.save_model()
            logger.info("[%s] 모델 학습 완료 (%d샘플)", self.agent_id, len(X))
            return True
        except Exception as e:
            logger.warning("[%s] 학습 실패: %s", self.agent_id, e)
            return False

    # ── 예측 ────────────────────────────────────────────────────

    def predict(
        self,
        ohlcv_list: list[dict],
        btc_ohlcv: list[dict] | None = None,
        kospi_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
    ) -> tuple[str, float]:
        """(signal, buy_prob) 반환. 모델 없으면 ('hold', 0.5)."""
        if self._model is None and not self.load_model():
            return "hold", 0.5
        try:
            full_df = compute_features(
                ohlcv_list,
                funding_rates=self._cached_funding_rates if self._cached_funding_rates else None,
                btc_ohlcv=btc_ohlcv,
                kospi_ohlcv=kospi_ohlcv,
                oi_hist=oi_hist or (self._cached_oi_hist or None),
                taker_hist=taker_hist or (self._cached_taker_hist or None),
            )
            if full_df.empty:
                return "hold", 0.5
            # ATR 캐싱 — _agent_execute에서 동적 손익 계산에 사용
            if "atr_pct" in full_df.columns:
                self._last_atr_pct = float(full_df["atr_pct"].iloc[-1])
            # ADX 레짐 필터 — 횡보장이면 예측 신뢰도 낮으므로 스킵
            # 주식은 코인보다 낮은 임계값(15) 적용 — 장 초반 ADX가 낮은 경우가 많음
            _adx_val = full_df["adx_14"].iloc[-1] if "adx_14" in full_df.columns else float("nan")
            _adx_thr = 15 if self.market == "stock" else 20
            if pd.isna(_adx_val) or _adx_val < _adx_thr:
                return "hold", 0.5
            feat_df = full_df[[c for c in self.feature_names if c in full_df.columns]].dropna()
            if feat_df.empty:
                return "hold", 0.5
            X_last = feat_df.iloc[[-1]].values
            expected = getattr(self._scaler, "n_features_in_", None)
            if expected is not None and X_last.shape[1] != expected:
                logger.warning(
                    "[%s] 피처 수 불일치: 현재 %d개, 스케일러 %d개 — hold 반환",
                    self.agent_id, X_last.shape[1], expected,
                )
                return "hold", 0.5
            X_scaled = self._scaler.transform(X_last)  # type: ignore[union-attr]
            prob = float(self._model.predict_proba(X_scaled)[0, 1])  # type: ignore[union-attr]
        except Exception:
            logger.debug("[%s] predict 예외 — hold 반환", self.agent_id, exc_info=True)
            return "hold", 0.5

        # ── MTF 정렬 필터: 3개 타임프레임 모두 불일치 시 신호 억제 ───
        # 펀딩률은 predict_live / predict_ensemble 에서 라이브 데이터로 처리 (이중 적용 방지)
        try:
            if "mtf_align" in full_df.columns:
                _align = float(full_df["mtf_align"].iloc[-1])
                if _align == 0.0:  # 15분·1시간·4시간 방향 전부 불일치
                    prob = max(0.01, min(0.99, prob * 0.85))
                elif _align == 3.0:  # 전부 일치 → 강화
                    prob = max(0.01, min(0.99, prob * 1.08))
        except Exception:
            pass

        if prob >= self.buy_threshold:
            return "buy", round(prob, 4)
        if prob <= (1.0 - self.buy_threshold):
            return "sell", round(prob, 4)
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
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")
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
        self._balance      = float(stats.get("current_balance") or INITIAL_CAPITAL)
        # SQLite INTEGER → Python int 보장 (일부 드라이버에서 str 반환 가능)
        self.total_trades  = int(stats.get("total_trades") or 0)
        self.win_trades    = int(stats.get("win_trades") or 0)
        self.is_champion   = bool(stats.get("is_champion", 0))
        self.is_active     = bool(stats.get("is_active", 1))
        # 자동 조정된 임계값 복구 (0이거나 없으면 AGENT_CONFIGS 기본값 유지)
        thr = stats.get("buy_threshold")
        if thr is not None and float(thr) > 0:
            self.buy_threshold = float(thr)
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

# ── 앙상블 그림자 에이전트 — 실매매 앙상블 신호를 가상 포트폴리오로 추적 ──────
# predict_ensemble() 호출 결과를 동일 TP/SL 로직으로 가상 매매해 앙상블 기대수익 검증
ENSEMBLE_AGENTS: dict[str, SimAgent] = {
    "ENSEMBLE_COIN":  SimAgent("ENSEMBLE_COIN",  5, 0.008, 0.65, "all", "coin",  5, "lgbm"),
    "ENSEMBLE_STOCK": SimAgent("ENSEMBLE_STOCK", 5, 0.006, 0.60, "all", "stock", 5, "lgbm"),
}


def reset_all_agents() -> None:
    """모든 에이전트(가상+앙상블) 메모리 초기화 — 잔액 10M, 거래기록 0, 포지션 없음."""
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
) -> tuple[str, float]:
    """가중 앙상블 게이트 — 전 에이전트 동적 투표 + 실시간 보정.

    가중치 = 승률 × (1 + 총수익률).
    지금 잘하는 전략이 자동으로 발언권이 커져서 장세에 자동 적응.
    """
    if not ohlcv_list:
        return "hold", 0.5
    candidates = [
        a for a in AGENTS.values()
        if a.market == market and a._model is not None
        # 30거래 이상 쌓인 에이전트 중 승률 40% 미만이면 앙상블에서 제외
        and not (a.total_trades >= 30 and a.win_rate < 0.40)
    ]
    if not candidates:
        # 폴백: 저승률이라도 모델 있는 에이전트 전원 사용 (완전 거래 중단 방지)
        candidates = [a for a in AGENTS.values() if a.market == market and a._model is not None]
    if not candidates:
        return "hold", 0.5

    # ── 현재 시장 레짐 감지 (앙상블 가중치 조정용) ───────────────
    current_regime = 1  # 기본: 추세장
    try:
        _tmp = compute_features(ohlcv_list)
        if not _tmp.empty and "regime_state" in _tmp.columns:
            current_regime = int(_tmp["regime_state"].iloc[-1])
    except Exception:
        pass

    # ── 가중 예측 집계 (샤프비율 × 수익률 × 레짐 부스터) ─────────
    weighted_prob = 0.0
    weighted_thr  = 0.0  # 임계값도 동일 가중치로 집계 (단순평균 불일치 방지)
    total_weight  = 0.0
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

        # 레짐별 전략 가중치 조정
        if current_regime == 1:    # 추세장: trend 우대, volume 축소
            if agent.feature_set == "trend":
                weight *= 1.5
            elif agent.feature_set == "volume":
                weight *= 0.7
        elif current_regime == 0:  # 횡보장: momentum 우대, trend 축소
            if agent.feature_set == "momentum":
                weight *= 1.4
            elif agent.feature_set == "trend":
                weight *= 0.8
        elif current_regime == 2:  # 고변동장: all 우대, 전체 보수적
            weight *= 0.7
            if agent.feature_set == "all":
                weight *= 1.3

        _, prob = agent.predict(ohlcv_list, btc_ohlcv=btc_ohlcv, kospi_ohlcv=kospi_ohlcv, oi_hist=oi_hist, taker_hist=taker_hist)
        weighted_prob += prob * weight
        weighted_thr  += agent.buy_threshold * weight
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

    # ── 임계값: 확률과 동일한 가중치로 집계 (고성능 에이전트 임계값 우선) ─────────
    avg_thr = weighted_thr / total_weight if total_weight > 0 else 0.60

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
