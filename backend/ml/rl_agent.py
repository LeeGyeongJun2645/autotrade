"""Fitted Q-Iteration 기반 RL 에이전트.

sklearn MLPRegressor로 Q-function을 근사. SimAgent를 상속해
포지션/통계/리스크 관리는 그대로 재사용하고 예측 모델만 교체.

학습:  OHLCV → 에피소드 시뮬레이션 → (s,a,r,s') 수집 → FQI로 Q-function 학습
예측:  현재 피처 + 포지션 상태 → Q(s,HOLD)/Q(s,BUY)/Q(s,SELL) → argmax 액션
"""

import logging
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from backend.ml.agents import FEATURE_SETS, MODEL_DIR, SimAgent
from backend.ml.features import compute_features

logger = logging.getLogger(__name__)

N_ACTIONS = 3      # 0=HOLD, 1=BUY, 2=SELL
GAMMA     = 0.95   # 할인율
FQI_ITERS = 5      # Fitted Q-Iteration 반복 횟수
COST      = 0.0015 # 거래비용 (슬리피지 포함)


class RLAgent(SimAgent):
    """Fitted Q-Iteration RL 에이전트 — SimAgent 상속으로 scheduler 완전 호환."""

    def __init__(
        self,
        agent_id: str,
        interval_min: int,
        market: str,
        feature_set: str = "all",
        lookahead: int = 5,
    ) -> None:
        # _rl_model 먼저 초기화 — super().__init__이 self._model=None 할당 시 setter가 호출되기 전에
        object.__setattr__(self, "_rl_model", None)
        object.__setattr__(self, "_rl_scaler", None)
        object.__setattr__(self, "_rl_feat_names", [])

        super().__init__(
            agent_id=agent_id,
            interval_min=interval_min,
            label_threshold=0.008,
            buy_threshold=0.5,
            feature_set=feature_set,
            market=market,
            lookahead=lookahead,
            model_type="lgbm",  # 경로 생성용, 실제 사용 안 함
        )

        self._rl_model_path = MODEL_DIR / f"rl_agent_{agent_id}_{interval_min}m.pkl"

    # ── _model을 _rl_model로 리다이렉트 ─────────────────────────────
    # scheduler: `if agent._model is None` → _rl_model이 None인지 확인하도록

    @property
    def _model(self):
        return self._rl_model

    @_model.setter
    def _model(self, v):
        # SimAgent.__init__이 self._model = None 설정하는 것만 무시
        # RLAgent에서는 _rl_model에 직접 저장
        pass

    # ── 모델 저장 / 로드 ──────────────────────────────────────────────

    def save_model(self) -> None:
        payload = {
            "rl_model":    self._rl_model,
            "rl_scaler":   self._rl_scaler,
            "feat_names":  self._rl_feat_names,
            "trained_at":  self._trained_at,
            "threshold":   self.buy_threshold,
        }
        with open(self._rl_model_path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self) -> bool:
        if not self._rl_model_path.exists():
            return False
        try:
            with open(self._rl_model_path, "rb") as f:
                p = pickle.load(f)
            self._rl_model    = p["rl_model"]
            self._rl_scaler   = p["rl_scaler"]
            self._rl_feat_names = p.get("feat_names", [])
            self._trained_at  = p.get("trained_at", "")
            self.buy_threshold = p.get("threshold", 0.5)
            return self._rl_model is not None
        except Exception as e:
            logger.warning("[%s] RL 모델 로드 실패: %s", self.agent_id, e)
            return False

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _build_features(self, ohlcv_list: list[dict], **kwargs) -> tuple | None:
        """OHLCV → (feat_arr, col_names, feat_df) 반환. 실패 시 None."""
        try:
            feat_df = compute_features(
                ohlcv_list,
                funding_rates=self._cached_funding_rates or None,
                btc_ohlcv=kwargs.get("btc_ohlcv"),
                oi_hist=self._cached_oi_hist or None,
                taker_hist=self._cached_taker_hist or None,
                ticker=kwargs.get("ticker", ""),
                interval_min=self.interval_min,
            )
            if feat_df.empty or len(feat_df) < 10:
                return None
            cols = [c for c in FEATURE_SETS[self.feature_set] if c in feat_df.columns]
            arr  = feat_df[cols].fillna(0).values
            return arr, cols, feat_df
        except Exception as e:
            logger.debug("[%s] 피처 빌드 실패: %s", self.agent_id, e)
            return None

    def _augment(self, state_vec: np.ndarray, in_pos: bool, pos_ret: float) -> np.ndarray:
        """피처 벡터에 포지션 상태를 추가한 augmented state."""
        return np.append(state_vec, [float(in_pos), pos_ret])

    def _q_values(self, aug_state: np.ndarray) -> np.ndarray:
        """augmented state → [Q(HOLD), Q(BUY), Q(SELL)]."""
        if self._rl_model is None or self._rl_scaler is None:
            return np.zeros(N_ACTIONS)
        try:
            scaled = self._rl_scaler.transform(aug_state.reshape(1, -1))
            q = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                onehot = np.eye(N_ACTIONS)[a]
                x = np.concatenate([scaled[0], onehot]).reshape(1, -1)
                q[a] = float(self._rl_model.predict(x)[0])
            return q
        except Exception as e:
            logger.debug("[%s] Q-value 계산 실패 (차원 불일치 등): %s", self.agent_id, e)
            return np.zeros(N_ACTIONS)

    # ── 예측 ─────────────────────────────────────────────────────────

    def predict(
        self,
        ohlcv_list: list[dict],
        btc_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
        ls_hist: list[dict] | None = None,
        ticker: str = "",
        **kwargs,
    ) -> tuple[str, float]:
        if self._rl_model is None:
            return "hold", 0.0

        result = self._build_features(ohlcv_list, btc_ohlcv=btc_ohlcv, ticker=ticker)
        if result is None:
            return "hold", 0.0

        feat_arr, cols, feat_df = result

        # ADX 업데이트 (ADX 차단 로직 호환)
        try:
            if "adx_14" in feat_df.columns:
                self._last_adx_14 = float(feat_df["adx_14"].iloc[-1])
        except Exception:
            pass

        # 최신 행 피처 → 학습 피처 순서에 맞게 정렬
        row_dict = dict(zip(cols, feat_arr[-1]))
        state_vec = np.array([row_dict.get(c, 0.0) for c in self._rl_feat_names[:-2]], dtype=float) \
            if len(self._rl_feat_names) > 2 else feat_arr[-1]

        # 포지션 상태
        in_pos  = ticker in self._positions
        pos_ret = 0.0
        if in_pos:
            try:
                entry = self._positions[ticker].entry_price
                pos_ret = float(feat_df["close"].iloc[-1]) / entry - 1.0
            except Exception:
                pass

        aug = self._augment(state_vec, in_pos, pos_ret)
        q   = self._q_values(aug)

        # 포지션 상태에 따른 유효 액션 마스킹
        masked_q = q.copy()
        if in_pos:
            masked_q[1] = -np.inf   # 이미 보유 → BUY 차단
        else:
            masked_q[2] = -np.inf   # 미보유 → SELL 차단

        best = int(np.argmax(masked_q))
        action = {0: "hold", 1: "buy", 2: "sell"}[best]

        # buy 확률: Q(BUY)와 Q(HOLD) 차이를 sigmoid로 변환
        # 차이 0.01(1%)마다 sigmoid가 움직이도록 스케일링
        q_diff = float(q[1] - q[0])
        prob   = float(1.0 / (1.0 + np.exp(-q_diff / 0.01)))
        prob   = round(min(max(prob, 0.01), 0.99), 4)

        return action, prob

    # ── 학습 (Fitted Q-Iteration) ────────────────────────────────────

    def train(
        self,
        ohlcv_list: list[dict],
        *args,
        **kwargs,
    ) -> bool:
        try:
            result = self._build_features(ohlcv_list, **{
                k: v for k, v in kwargs.items() if k in ("btc_ohlcv", "ticker")
            })
            if result is None:
                return False

            feat_arr, cols, feat_df = result
            n = len(feat_arr)
            if n < 60:
                logger.warning("[%s] 학습 데이터 부족 (%d행)", self.agent_id, n)
                return False

            closes = feat_df["close"].values.astype(float)
            return self._fit_fqi(feat_arr, cols, closes)

        except Exception as e:
            logger.warning("[%s] RL 학습 실패: %s", self.agent_id, e)
            return False

    def train_multi(
        self,
        ohlcv_by_symbol: dict[str, list[dict]],
        kospi_ohlcv: list[dict] | None = None,
        *args,
        **kwargs,
    ) -> bool:
        """주식 에이전트: 다중 종목 OHLCV 풀링 후 학습."""
        per_sym: list[tuple[np.ndarray, list[str], np.ndarray]] = []

        for ohlcv in ohlcv_by_symbol.values():
            if len(ohlcv) < 60:
                continue
            result = self._build_features(ohlcv)
            if result is None:
                continue
            feat_arr, cols, feat_df = result
            per_sym.append((feat_arr, cols, feat_df["close"].values.astype(float)))

        if not per_sym:
            return False

        # 교집합 컬럼 기준으로 각 종목 정렬 — 종목별 피처 수 불일치 방지
        ref_cols = per_sym[0][1]
        aligned_feats:  list[np.ndarray] = []
        aligned_closes: list[np.ndarray] = []

        for feat_arr, cols, close_arr in per_sym:
            col_pos = {c: i for i, c in enumerate(cols)}
            aligned = np.zeros((len(feat_arr), len(ref_cols)), dtype=np.float64)
            for j, c in enumerate(ref_cols):
                if c in col_pos:
                    aligned[:, j] = feat_arr[:, col_pos[c]]
            aligned_feats.append(aligned)
            aligned_closes.append(close_arr)

        all_feats  = np.concatenate(aligned_feats,  axis=0)
        all_closes = np.concatenate(aligned_closes, axis=0)
        return self._fit_fqi(all_feats, ref_cols, all_closes)

    def _fit_fqi(
        self,
        feat_arr: np.ndarray,
        cols: list[str],
        closes: np.ndarray,
    ) -> bool:
        """에피소드 시뮬레이션 → FQI 학습 핵심 로직."""
        n = len(feat_arr)

        # ── 에피소드 시뮬레이션 ───────────────────────────────────────
        aug_states:      list[np.ndarray] = []
        aug_next_states: list[np.ndarray] = []
        actions:  list[int]   = []
        rewards:  list[float] = []
        dones:    list[float] = []

        in_pos    = False
        entry_px  = 0.0
        entry_t   = 0

        for t in range(n - self.lookahead):
            pos_ret = (closes[t] / entry_px - 1.0) if in_pos else 0.0
            s  = self._augment(feat_arr[t], in_pos, pos_ret)
            aug_states.append(s)

            future_ret = closes[t + self.lookahead] / closes[t] - 1.0

            if not in_pos:
                if future_ret > COST * 2:        # 충분한 미래 상승 → BUY
                    a, r = 1, 0.0
                    in_pos   = True
                    entry_px = closes[t]
                    entry_t  = t
                else:
                    a, r = 0, -0.0001            # HOLD 기회비용
            else:
                held = t - entry_t
                if (pos_ret > self.lookahead * 0.002   # 목표 수익
                        or pos_ret < -0.015            # 손절
                        or held >= self.lookahead * 2): # 최대 보유
                    a, r  = 2, pos_ret - COST
                    in_pos = False
                else:
                    a, r = 0, pos_ret * 0.05     # HOLD 중간 보상

            actions.append(a)
            rewards.append(r)

            in_pos_next = in_pos
            pos_ret_next = (closes[t+1] / entry_px - 1.0) if in_pos_next else 0.0
            aug_next_states.append(self._augment(feat_arr[t+1], in_pos_next, pos_ret_next))
            dones.append(1.0 if t == n - self.lookahead - 1 else 0.0)

        if len(aug_states) < 30:
            return False

        S  = np.array(aug_states)
        S2 = np.array(aug_next_states)
        A  = np.array(actions)
        R  = np.array(rewards)
        D  = np.array(dones)

        scaler = StandardScaler()
        S_sc  = scaler.fit_transform(S)
        S2_sc = scaler.transform(S2)

        feat_dim = S_sc.shape[1]

        # (state, action_onehot) → Q-value
        X_sa = np.zeros((len(S), feat_dim + N_ACTIONS))
        for i, a in enumerate(A):
            X_sa[i] = np.concatenate([S_sc[i], np.eye(N_ACTIONS)[a]])

        q_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

        y = R.copy()

        # ── Fitted Q-Iteration ────────────────────────────────────────
        for _ in range(FQI_ITERS):
            q_model.fit(X_sa, y)

            # max_a' Q(s', a') 계산
            next_q = np.zeros((len(S2_sc), N_ACTIONS))
            for a_n in range(N_ACTIONS):
                X_next = np.column_stack([S2_sc, np.tile(np.eye(N_ACTIONS)[a_n], (len(S2_sc), 1))])
                next_q[:, a_n] = q_model.predict(X_next)

            y = R + GAMMA * next_q.max(axis=1) * (1 - D)

        # ── 검증 지표 ────────────────────────────────────────────────
        buy_n      = int((A == 1).sum())
        sell_rets  = R[A == 2]
        avg_ret    = float(sell_rets.mean()) if len(sell_rets) > 0 else 0.0
        win_rate   = float((sell_rets > 0).mean()) if len(sell_rets) > 0 else 0.0

        self._rl_model    = q_model
        self._rl_scaler   = scaler
        self._rl_feat_names = cols + ["__in_pos__", "__pos_ret__"]
        self._trained_at  = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        self.save_model()

        logger.warning(
            "[%s] RL학습 완료 (%d샘플, BUY=%d, 시뮬WR=%.1f%%, 평균수익=%.3f%%)",
            self.agent_id, len(S), buy_n, win_rate * 100, avg_ret * 100,
        )
        return True

    def train_meta(self, trade_results: list[dict]) -> bool:
        return False


# ── RL 에이전트 등록 ─────────────────────────────────────────────────
RL_AGENTS: dict[str, RLAgent] = {
    "RL01": RLAgent("RL01", interval_min=15, market="coin",  feature_set="all", lookahead=5),
    "RL02": RLAgent("RL02", interval_min=15, market="stock", feature_set="all", lookahead=5),
}
