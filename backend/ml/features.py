"""기술지표 Feature 계산 모듈.

OHLCV 리스트 → 16개 Feature DataFrame 반환.
XGBoost 학습 및 예측에서 공통 사용.
"""

import pandas as pd
from ta import momentum, trend, volatility

FEATURE_NAMES = [
    "rsi_14",
    "macd_diff",
    "bb_pband",
    "atr_pct",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma5_cross_ma20",
    "ma20_cross_ma60",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_ratio",
    "cci_20",
    "stoch_rsi",
]


def compute_features(ohlcv_list: list[dict]) -> pd.DataFrame:
    """OHLCV 리스트 → Feature DataFrame 변환.

    Args:
        ohlcv_list: API 반환 형식 (최신 날짜 앞, 내림차순)

    Returns:
        FEATURE_NAMES 컬럼만 가진 DataFrame (NaN 행 제거됨)
        인덱스: 날짜 (DatetimeIndex)
    """
    df = pd.DataFrame(list(reversed(ohlcv_list)))
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # RSI(14)
    df["rsi_14"] = momentum.RSIIndicator(close, window=14).rsi()

    # MACD difference (MACD line - Signal line)
    macd = trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands %B
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()

    # ATR / 종가 (정규화된 변동성)
    atr = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = atr / close

    # 이동평균 대비 비율
    ma5  = trend.SMAIndicator(close, window=5).sma_indicator()
    ma20 = trend.SMAIndicator(close, window=20).sma_indicator()
    ma60 = trend.SMAIndicator(close, window=60).sma_indicator()
    df["ma5_ratio"]  = close / ma5 - 1
    df["ma20_ratio"] = close / ma20 - 1
    df["ma60_ratio"] = close / ma60 - 1

    # 골든/데드 크로스 상태
    df["ma5_cross_ma20"]  = (ma5 > ma20).astype(float)
    df["ma20_cross_ma60"] = (ma20 > ma60).astype(float)

    # 단기 모멘텀 (수익률)
    df["ret_1d"]  = close.pct_change(1)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # 거래량 비율 (현재 / 20일 평균)
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ratio"] = vol / vol_ma20

    # CCI(20)
    df["cci_20"] = trend.CCIIndicator(high, low, close, window=20).cci()

    # Stochastic RSI
    df["stoch_rsi"] = momentum.StochRSIIndicator(close, window=14).stochrsi()

    return df[FEATURE_NAMES].dropna()
