"""XGBoost 기반 방향성 예측 모델.

레이블: 다음 봉 종가 / 현재 봉 종가 >= 1.005 (+0.5% 이상 상승) → 1
특징: features.py의 30개 기술지표 + 뉴스 감성 + 공포탐욕 지수
저장: data/models/xgb_{ticker}.pkl

신호 해석:
    buy_prob >= 0.70 → strong_buy
    buy_prob >= 0.58 → buy
    buy_prob <= 0.30 → strong_sell
    buy_prob <= 0.42 → sell
    나머지           → hold
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.features import FEATURE_NAMES, compute_features
from backend.ml.news import get_sentiment_score

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_THRESHOLD = 0.003  # +0.3%로 낮춤 — 거래비용(0.05%) 충분 커버, 양성 레이블 비율 확보

_FEAR_GREED_CACHE: tuple[float, float] | None = None  # (value_-1to1, timestamp)
_FEAR_GREED_TTL = 3600.0  # 1시간 캐시


@dataclass
class PredictResult:
    ticker: str
    buy_prob: float
    signal: str        # strong_buy | buy | hold | sell | strong_sell
    news_score: float
    fear_greed: float  # -1(극도공포) ~ +1(극도탐욕), 코인 전용 (주식은 0.0)
    confidence: str    # high | medium | low
    trained_at: str | None


@dataclass
class TrainResult:
    ticker: str
    accuracy: float
    roc_auc: float
    n_samples: int
    n_features: int
    trained_at: str


def _prob_to_signal(prob: float, buy_thr: float = 0.58) -> tuple[str, str]:
    strong_thr = min(buy_thr + 0.12, 0.90)
    sell_thr   = 1.0 - buy_thr
    if prob >= strong_thr:
        return "strong_buy", "high"
    if prob >= buy_thr:
        return "buy", "medium"
    if prob <= 1.0 - strong_thr:
        return "strong_sell", "high"
    if prob <= sell_thr:
        return "sell", "medium"
    return "hold", "low"


async def _get_fear_greed() -> float:
    """Crypto Fear & Greed Index 조회. -1(극도공포)~+1(극도탐욕) 반환."""
    global _FEAR_GREED_CACHE

    import time
    now = time.time()
    if _FEAR_GREED_CACHE and now - _FEAR_GREED_CACHE[1] < _FEAR_GREED_TTL:
        return _FEAR_GREED_CACHE[0]

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            resp.raise_for_status()
            data = resp.json()["data"][0]
            raw = float(data["value"])  # 0~100
            normalized = (raw - 50) / 50  # -1~+1
            _FEAR_GREED_CACHE = (normalized, now)
            logger.debug("[ML] 공포탐욕 지수: %.0f → %.3f", raw, normalized)
            return normalized
    except Exception as e:
        logger.warning("[ML] 공포탐욕 지수 조회 실패: %s", e)
        return 0.0


class XGBSignalModel:
    """티커별 XGBoost 신호 모델."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._model: XGBClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._trained_at: str | None = None
        self._buy_threshold: float = 0.58   # 정밀도 기반 최적 임계값 (학습 후 갱신)
        self._model_path = MODEL_DIR / f"xgb_{ticker.replace('/', '_').replace('-', '_')}.pkl"

    # ── 저장/로드 ────────────────────────────────────────────────

    def load(self) -> bool:
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            model = data["model"]
            # 피처 수 불일치 → 재학습 필요
            if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_NAMES):
                logger.warning(
                    "[ML] %s 피처 수 불일치 (%d → %d), 재학습 필요",
                    self.ticker, model.n_features_in_, len(FEATURE_NAMES),
                )
                return False
            self._model = model
            self._scaler = data["scaler"]
            self._trained_at = data.get("trained_at")
            self._buy_threshold = float(data.get("buy_threshold", 0.58))
            logger.info("[ML] %s 모델 로드 성공 (학습일: %s)", self.ticker, self._trained_at)
            return True
        except Exception as e:
            logger.warning("[ML] %s 모델 로드 실패: %s", self.ticker, e)
            return False

    def save(self) -> None:
        with open(self._model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "trained_at": self._trained_at,
                    "buy_threshold": self._buy_threshold,
                },
                f,
            )
        logger.info("[ML] %s 모델 저장: %s", self.ticker, self._model_path)

    # ── 학습 ────────────────────────────────────────────────────

    def train(
        self,
        ohlcv_list: list[dict],
        btc_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
    ) -> TrainResult:
        """동기 학습 함수 — FastAPI에서 asyncio.to_thread 로 호출."""
        feat_df = compute_features(ohlcv_list, btc_ohlcv=btc_ohlcv, oi_hist=oi_hist, taker_hist=taker_hist)
        if len(feat_df) < 60:
            raise ValueError(
                f"{self.ticker}: 학습 데이터 부족 ({len(feat_df)}봉, 최소 60봉 필요)"
            )

        # 위치 기반 정렬 — 일봉/분봉 모두 인덱스 불일치 없이 동작
        feat_df = feat_df.reset_index(drop=True)
        close_vals = [float(c["close"]) for c in reversed(ohlcv_list)]
        close = pd.Series(close_vals)
        n = len(feat_df)
        close = close.iloc[-n:].reset_index(drop=True)

        next_close = close.shift(-1)
        safe_close = close.replace(0, float("nan"))
        label = ((next_close / safe_close - 1) >= LABEL_THRESHOLD).fillna(False).astype(int)
        feat_df = feat_df.iloc[:-1]
        label = label.iloc[:-1]
        min_len = min(len(feat_df), len(label))
        feat_df = feat_df.iloc[:min_len]
        label = label.iloc[:min_len]

        X = feat_df[FEATURE_NAMES].values
        y = label.values.astype(int)

        # 클래스 불균형 보정
        _n_neg = int((y == 0).sum())
        _n_pos = int((y == 1).sum())
        _scale_pos = _n_neg / max(_n_pos, 1)

        # Purged TimeSeriesSplit — gap=12봉(1시간) 엠바고로 레이블 오염 방지
        tscv = TimeSeriesSplit(n_splits=5, gap=12)
        acc_scores, auc_scores = [], []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=_scale_pos,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

            pred = clf.predict(X_val_s)
            prob = clf.predict_proba(X_val_s)[:, 1]
            acc_scores.append(accuracy_score(y_val, pred))
            if len(set(y_val)) > 1:
                auc_scores.append(roc_auc_score(y_val, prob))

        # 전체 데이터로 최종 학습
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        final = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=_scale_pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        final.fit(X_s, y, verbose=False)

        # 정밀도(Precision) 기반 최적 임계값 탐색 — 승률 직결 지표
        _probs_all = final.predict_proba(X_s)[:, 1]
        _best_thr, _best_prec = 0.58, 0.0
        for _t in np.arange(0.45, 0.80, 0.01):
            _preds = (_probs_all >= _t).astype(int)
            _cnt = int(_preds.sum())
            if _cnt < max(10, len(y) // 20):
                continue
            _recall = float(_preds[y == 1].mean()) if (y == 1).sum() > 0 else 0.0
            if _recall < 0.20:
                continue
            from sklearn.metrics import precision_score as _ps
            _prec = _ps(y, _preds, zero_division=0)
            if _prec > _best_prec:
                _best_prec, _best_thr = _prec, float(_t)
        logger.info("[ML] %s 최적 임계값: %.2f (정밀도 %.3f)", self.ticker, _best_thr, _best_prec)

        self._model = final
        self._scaler = scaler
        self._trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._buy_threshold = _best_thr
        self.save()

        avg_acc = float(np.mean(acc_scores)) if acc_scores else 0.0
        avg_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

        logger.info(
            "[ML] %s 학습 완료 | 샘플: %d | 정확도: %.3f | ROC-AUC: %.3f",
            self.ticker, len(X), avg_acc, avg_auc,
        )
        return TrainResult(
            ticker=self.ticker,
            accuracy=round(avg_acc, 4),
            roc_auc=round(avg_auc, 4),
            n_samples=len(X),
            n_features=X.shape[1],
            trained_at=self._trained_at,
        )

    # ── 예측 ────────────────────────────────────────────────────

    async def predict(
        self,
        ohlcv_list: list[dict],
        token: str | None = None,
        market: str = "auto",
        btc_ohlcv: list[dict] | None = None,
        oi_hist: list[dict] | None = None,
        taker_hist: list[dict] | None = None,
    ) -> PredictResult:
        """최신 봉 기준 매수 확률 및 신호 반환.

        market: "coin" | "stock" | "auto" (KRW- 접두사로 자동 판별)
        """
        if self._model is None:
            if not self.load():
                raise RuntimeError(
                    f"{self.ticker}: 학습된 모델 없음. POST /ml/train 먼저 호출"
                )

        feat_df = compute_features(ohlcv_list, btc_ohlcv=btc_ohlcv, oi_hist=oi_hist, taker_hist=taker_hist)
        if feat_df.empty:
            raise ValueError(f"{self.ticker}: Feature 계산 실패")

        missing_cols = [c for c in FEATURE_NAMES if c not in feat_df.columns]
        if missing_cols:
            raise ValueError(f"{self.ticker}: Feature 컬럼 {len(missing_cols)}개 누락")
        X_last = feat_df.iloc[[-1]][FEATURE_NAMES].values
        X_scaled = self._scaler.transform(X_last)  # type: ignore[union-attr]
        buy_prob = float(self._model.predict_proba(X_scaled)[0, 1])  # type: ignore[union-attr]

        is_coin = self.ticker.startswith("KRW-") if market == "auto" else (market == "coin")
        fear_greed = 0.0

        if is_coin:
            # 공포&탐욕 지수 (역발상: 탐욕→과열 하향, 공포→저평가 상향)
            fear_greed = await _get_fear_greed()
            buy_prob = max(0.01, min(0.99, buy_prob - fear_greed * 0.04))

            # 김치프리미엄 + 펀딩비 — 최신 종가 기준
            try:
                current_price = float(ohlcv_list[0]["close"])
                coin_sym = self.ticker.replace("KRW-", "") + "USDT"
                from backend.api.binance import get_funding_rate, get_kimchi_premium
                kimchi, funding = await asyncio.gather(
                    get_kimchi_premium(current_price, coin_sym),
                    get_funding_rate(coin_sym),
                )
                # 김치프리미엄 >3% 과열, <-1% 저평가
                if kimchi > 3.0:
                    buy_prob = max(0.01, buy_prob - min((kimchi - 3.0) * 0.01, 0.05))
                elif kimchi < -1.0:
                    buy_prob = min(0.99, buy_prob + min((-kimchi - 1.0) * 0.01, 0.03))
                # 펀딩비 양수→롱과열, 음수→숏과열
                if abs(funding) > 0.001:
                    fund_adj = min(abs(funding) * 20, 0.04) * (1 if funding > 0 else -1)
                    buy_prob = max(0.01, min(0.99, buy_prob - fund_adj))
            except Exception as e:
                logger.debug("[ML] 코인 보조지표 실패 (무시): %s", e)

            # 업비트 호가창: buy_pressure 0.5 기준 ±4%
            try:
                from backend.api.upbit import get_orderbook
                ob = await get_orderbook(self.ticker)
                bp_adj = (ob.get("buy_pressure", 0.5) - 0.5) * 0.08
                buy_prob = max(0.01, min(0.99, buy_prob + bp_adj))
            except Exception as e:
                logger.debug("[ML] 업비트 호가창 실패 (무시): %s", e)

        else:  # stock
            # KIS 호가창 + 기관/외국인 순매수
            try:
                from backend.api.kis import get_investor_trend, get_orderbook_kis
                ob, inv = await asyncio.gather(
                    get_orderbook_kis(self.ticker),
                    get_investor_trend(self.ticker),
                )
                bp_adj = (ob.get("buy_pressure", 0.5) - 0.5) * 0.06
                buy_prob = max(0.01, min(0.99, buy_prob + bp_adj))

                net = inv.get("institution_net_buy", 0) + inv.get("foreign_net_buy", 0)
                if net > 0:
                    buy_prob = min(0.99, buy_prob + 0.02)
                elif net < 0:
                    buy_prob = max(0.01, buy_prob - 0.02)
            except Exception as e:
                logger.debug("[ML] KIS 보조지표 실패 (무시): %s", e)

        signal, confidence = _prob_to_signal(buy_prob, self._buy_threshold)

        # 뉴스 감성 조회
        news_score = await get_sentiment_score(self.ticker, token)

        # hold 신호를 뉴스+공포탐욕으로 보정
        if signal == "hold":
            combined = news_score * 0.6 + (-fear_greed) * 0.4
            if combined > 0.25:
                signal, confidence = "buy", "low"
            elif combined < -0.25:
                signal, confidence = "sell", "low"

        return PredictResult(
            ticker=self.ticker,
            buy_prob=round(buy_prob, 4),
            signal=signal,
            news_score=round(news_score, 4),
            fear_greed=round(fear_greed, 3),
            confidence=confidence,
            trained_at=self._trained_at,
        )
