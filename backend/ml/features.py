"""기술지표 Feature 계산 모듈.

OHLCV 리스트 → 30개 Feature DataFrame 반환.
XGBoost 학습 및 예측에서 공통 사용.
"""

import pandas as pd
from ta import momentum, trend, volatility, volume

FEATURE_NAMES = [
    # ── 모멘텀 ────────────────────────────
    "rsi_14",
    "macd_diff",
    "stoch_rsi",
    "williams_r",
    "mfi_14",
    "roc_10",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    # ── 추세 ──────────────────────────────
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma5_cross_ma20",
    "ma20_cross_ma60",
    "adx_14",
    "adx_pos",
    "adx_neg",
    "trix_15",
    "dpo_20",
    "vortex_diff",
    # ── 변동성 ────────────────────────────
    "bb_pband",
    "atr_pct",
    "mass_index",
    "hl_range_20",
    # ── 거래량 ────────────────────────────
    "vol_ratio",
    "obv_change",
    "cmf_20",
    # ── 기타 ──────────────────────────────
    "cci_20",
]


def compute_features(ohlcv_list: list[dict]) -> pd.DataFrame:
    """OHLCV 리스트 → Feature DataFrame 변환.

    Args:
        ohlcv_list: API 반환 형식 (최신 날짜 앞, 내림차순)

    Returns:
        FEATURE_NAMES 컬럼만 가진 DataFrame (NaN 행 제거됨)
    """
    df = pd.DataFrame(list(reversed(ohlcv_list)))
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── 모멘텀 ────────────────────────────────────────────────────
    df["rsi_14"] = momentum.RSIIndicator(close, window=14).rsi()

    macd = trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_diff"] = macd.macd_diff()

    df["stoch_rsi"] = momentum.StochRSIIndicator(close, window=14).stochrsi()

    df["williams_r"] = momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    df["mfi_14"] = volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()

    df["roc_10"] = momentum.ROCIndicator(close, window=10).roc()

    df["ret_1d"]  = close.pct_change(1)
    df["ret_3d"]  = close.pct_change(3)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # ── 추세 ──────────────────────────────────────────────────────
    ma5  = trend.SMAIndicator(close, window=5).sma_indicator()
    ma20 = trend.SMAIndicator(close, window=20).sma_indicator()
    ma60 = trend.SMAIndicator(close, window=60).sma_indicator()
    df["ma5_ratio"]  = close / ma5 - 1
    df["ma20_ratio"] = close / ma20 - 1
    df["ma60_ratio"] = close / ma60 - 1

    df["ma5_cross_ma20"]  = (ma5 > ma20).astype(float)
    df["ma20_cross_ma60"] = (ma20 > ma60).astype(float)

    adx_ind = trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"]  = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()
    df["adx_neg"] = adx_ind.adx_neg()

    df["trix_15"] = trend.TRIXIndicator(close, window=15).trix()

    df["dpo_20"] = trend.DPOIndicator(close, window=20).dpo()

    vortex = trend.VortexIndicator(high, low, close, window=14)
    df["vortex_diff"] = vortex.vortex_indicator_pos() - vortex.vortex_indicator_neg()

    # ── 변동성 ────────────────────────────────────────────────────
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()

    atr = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = atr / close

    df["mass_index"] = trend.MassIndex(high, low, window_fast=9, window_slow=25).mass_index()

    # 20일 고저폭 / 종가 — 상대적 변동 범위
    df["hl_range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close

    # ── 거래량 ────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ratio"] = vol / vol_ma20

    obv = volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    obv_ma5 = obv.rolling(5).mean()
    df["obv_change"] = (obv - obv_ma5) / (obv_ma5.abs() + 1e-9)

    df["cmf_20"] = volume.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20).chaikin_money_flow()

    # ── 기타 ──────────────────────────────────────────────────────
    df["cci_20"] = trend.CCIIndicator(high, low, close, window=20).cci()

    return df[FEATURE_NAMES].dropna()
