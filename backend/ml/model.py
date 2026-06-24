"""XGBoost 기반 방향성 예측 모델.

레이블: 다음 봉 종가 / 현재 봉 종가 >= 1.005 (+0.5% 이상 상승) → 1
특징: features.py의 16개 기술지표 + 뉴스 감성 점수
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

LABEL_THRESHOLD = 0.005  # +0.5% 이상 상승 → 양성 레이블


@dataclass
class PredictResult:
    ticker: str
    buy_prob: float
    signal: str        # strong_buy | buy | hold | sell | strong_sell
    news_score: float
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


def _prob_to_signal(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "strong_buy", "high"
    if prob >= 0.58:
        return "buy", "medium"
    if prob <= 0.30:
        return "strong_sell", "high"
    if prob <= 0.42:
        return "sell", "medium"
    return "hold", "low"


class XGBSignalModel:
    """티커별 XGBoost 신호 모델."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._model: XGBClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._trained_at: str | None = None
        self._model_path = MODEL_DIR / f"xgb_{ticker.replace('/', '_').replace('-', '_')}.pkl"

    # ── 저장/로드 ────────────────────────────────────────────────

    def load(self) -> bool:
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            self._trained_at = data.get("trained_at")
            logger.info("[ML] %s 모델 로드 성공 (학습일: %s)", self.ticker, self._trained_at)
            return True
        except Exception as e:
            logger.warning("[ML] %s 모델 로드 실패: %s", self.ticker, e)
            return False

    def save(self) -> None:
        with open(self._model_path, "wb") as f:
            pickle.dump(
                {"model": self._model, "scaler": self._scaler, "trained_at": self._trained_at},
                f,
            )
        logger.info("[ML] %s 모델 저장: %s", self.ticker, self._model_path)

    # ── 학습 ────────────────────────────────────────────────────

    def train(self, ohlcv_list: list[dict]) -> TrainResult:
        """동기 학습 함수 — FastAPI에서 asyncio.to_thread 로 호출."""
        feat_df = compute_features(ohlcv_list)
        if len(feat_df) < 60:
            raise ValueError(
                f"{self.ticker}: 학습 데이터 부족 ({len(feat_df)}봉, 최소 60봉 필요)"
            )

        # 원본 close 시리즈 (feat_df 인덱스에 align)
        df_raw = pd.DataFrame(list(reversed(ohlcv_list)))
        df_raw["date"] = pd.to_datetime(df_raw["date"].astype(str).str[:10])
        df_raw = df_raw.set_index("date").sort_index()
        close = df_raw["close"].astype(float)

        next_close = close.shift(-1)
        label = ((next_close / close - 1) >= LABEL_THRESHOLD).astype(int)
        label = label.reindex(feat_df.index).dropna()
        feat_df = feat_df.loc[label.index].iloc[:-1]
        label = label.iloc[:-1]

        X = feat_df[FEATURE_NAMES].values
        y = label.values.astype(int)

        # TimeSeriesSplit 5-fold
        tscv = TimeSeriesSplit(n_splits=5)
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
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        final.fit(X_s, y, verbose=False)

        self._model = final
        self._scaler = scaler
        self._trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    async def predict(self, ohlcv_list: list[dict], token: str | None = None) -> PredictResult:
        """최신 봉 기준 매수 확률 및 신호 반환."""
        if self._model is None:
            if not self.load():
                raise RuntimeError(
                    f"{self.ticker}: 학습된 모델 없음. POST /ml/train 먼저 호출"
                )

        feat_df = compute_features(ohlcv_list)
        if feat_df.empty:
            raise ValueError(f"{self.ticker}: Feature 계산 실패")

        X_last = feat_df.iloc[[-1]][FEATURE_NAMES].values
        X_scaled = self._scaler.transform(X_last)  # type: ignore[union-attr]
        buy_prob = float(self._model.predict_proba(X_scaled)[0, 1])  # type: ignore[union-attr]
        signal, confidence = _prob_to_signal(buy_prob)

        news_score = await get_sentiment_score(self.ticker, token)

        # 뉴스 감성으로 hold 신호 보정
        if signal == "hold":
            if news_score > 0.3:
                signal, confidence = "buy", "low"
            elif news_score < -0.3:
                signal, confidence = "sell", "low"

        return PredictResult(
            ticker=self.ticker,
            buy_prob=round(buy_prob, 4),
            signal=signal,
            news_score=round(news_score, 4),
            confidence=confidence,
            trained_at=self._trained_at,
        )
