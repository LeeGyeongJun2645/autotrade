"""기술지표 Feature 계산 모듈.

OHLCV 리스트 → Feature DataFrame 반환.
XGBoost 학습 및 예측에서 공통 사용.
"""

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility, volume

FEATURE_NAMES = [
    # ── 모멘텀 ────────────────────────────
    "rsi_9",          # 14→9 (5분봉 단타 최적화)
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
    "ema9_cross_ema21",  # EMA 9/21 골든·데드 크로스
    # ── VWAP ──────────────────────────────
    "vwap_ratio",        # (close/VWAP20) - 1 : 현재가 vs 거래량가중평균 괴리율
    "vwap_cross",        # close > VWAP20 이면 1 (추세 방향)
    # ── 변동성 ────────────────────────────
    "bb_pband",
    "atr_pct",
    "mass_index",
    "hl_range_20",
    # ── 거래량 ────────────────────────────
    "vol_ratio",
    "vol_surge",
    "vol_surge_flag",
    "obv_change",
    "cmf_20",
    # ── 기타 ──────────────────────────────
    "cci_20",
    # ── 캔들 패턴 ─────────────────────────
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
    "is_doji",
    "is_hammer",
    "is_shooting_star",
    "is_bullish_engulf",
    "is_bearish_engulf",
    "is_marubozu_bull",
    "is_marubozu_bear",
    "consecutive_up",
    "consecutive_down",
    "candle_color",
]


def compute_features(ohlcv_list: list[dict]) -> pd.DataFrame:
    """OHLCV 리스트 → Feature DataFrame 변환.

    Args:
        ohlcv_list: API 반환 형식 (최신 날짜 앞, 내림차순)

    Returns:
        FEATURE_NAMES 컬럼만 가진 DataFrame (NaN 행 제거됨)
    """
    df = pd.DataFrame(list(reversed(ohlcv_list)))
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:19])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    vol    = df["volume"]

    # ── 모멘텀 ────────────────────────────────────────────────────
    df["rsi_9"]     = momentum.RSIIndicator(close, window=9).rsi()
    macd            = trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_diff"] = macd.macd_diff()
    df["stoch_rsi"] = momentum.StochRSIIndicator(close, window=14).stochrsi()
    df["williams_r"]= momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df["mfi_14"]    = volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    df["roc_10"]    = momentum.ROCIndicator(close, window=10).roc()
    df["ret_1d"]    = close.pct_change(1)
    df["ret_3d"]    = close.pct_change(3)
    df["ret_5d"]    = close.pct_change(5)
    df["ret_10d"]   = close.pct_change(10)
    df["ret_20d"]   = close.pct_change(20)

    # ── 추세 ──────────────────────────────────────────────────────
    ma5  = trend.SMAIndicator(close, window=5).sma_indicator()
    ma20 = trend.SMAIndicator(close, window=20).sma_indicator()
    ma60 = trend.SMAIndicator(close, window=60).sma_indicator()
    df["ma5_ratio"]       = close / ma5 - 1
    df["ma20_ratio"]      = close / ma20 - 1
    df["ma60_ratio"]      = close / ma60 - 1
    df["ma5_cross_ma20"]  = (ma5 > ma20).astype(float)
    df["ma20_cross_ma60"] = (ma20 > ma60).astype(float)

    adx_ind       = trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"]  = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()
    df["adx_neg"] = adx_ind.adx_neg()
    df["trix_15"] = trend.TRIXIndicator(close, window=15).trix()
    df["dpo_20"]  = trend.DPOIndicator(close, window=20).dpo()
    vortex        = trend.VortexIndicator(high, low, close, window=14)
    df["vortex_diff"] = vortex.vortex_indicator_pos() - vortex.vortex_indicator_neg()

    ema9   = trend.EMAIndicator(close, window=9).ema_indicator()
    ema21  = trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema9_cross_ema21"] = (ema9 > ema21).astype(float)

    # ── VWAP (20봉 롤링) ─────────────────────────────────────────
    typical_price = (high + low + close) / 3
    vwap_20       = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum()
    df["vwap_ratio"] = close / vwap_20 - 1
    df["vwap_cross"] = (close > vwap_20).astype(float)

    # ── 변동성 ────────────────────────────────────────────────────
    bb             = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()
    atr            = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"]  = atr / close
    df["mass_index"]  = trend.MassIndex(high, low, window_fast=9, window_slow=25).mass_index()
    df["hl_range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close

    # ── 거래량 ────────────────────────────────────────────────────
    vol_ma20          = vol.rolling(20).mean()
    df["vol_ratio"]   = vol / vol_ma20
    df["vol_surge"]   = vol / vol.rolling(5).mean()   # 5봉 평균 대비 급증 비율
    df["vol_surge_flag"] = (df["vol_surge"] >= 3).astype(float)  # 3배 이상 = 급증

    obv               = volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    obv_ma5           = obv.rolling(5).mean()
    df["obv_change"]  = (obv - obv_ma5) / (obv_ma5.abs() + 1e-9)
    df["cmf_20"]      = volume.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20).chaikin_money_flow()

    # ── 기타 ──────────────────────────────────────────────────────
    df["cci_20"] = trend.CCIIndicator(high, low, close, window=20).cci()

    # ── 캔들 패턴 ─────────────────────────────────────────────────
    candle_range = (high - low).replace(0, 1e-9)
    body         = (close - open_).abs()
    body_top     = close.where(close > open_, open_)
    body_bot     = open_.where(close > open_, close)

    df["body_ratio"]    = body / candle_range
    df["upper_shadow"]  = (high - body_top) / candle_range
    df["lower_shadow"]  = (body_bot - low) / candle_range
    df["candle_color"]  = (close > open_).astype(float)  # 1=양봉, 0=음봉

    # 도지: 몸통이 전체 범위의 10% 미만
    df["is_doji"] = (df["body_ratio"] < 0.1).astype(float)

    # 망치형(Hammer): 아래꼬리 60%+, 몸통 30% 미만, 위꼬리 10% 미만
    df["is_hammer"] = (
        (df["lower_shadow"] >= 0.6) &
        (df["body_ratio"] < 0.3) &
        (df["upper_shadow"] < 0.1)
    ).astype(float)

    # 유성형(Shooting Star): 위꼬리 60%+, 몸통 30% 미만, 아래꼬리 10% 미만
    df["is_shooting_star"] = (
        (df["upper_shadow"] >= 0.6) &
        (df["body_ratio"] < 0.3) &
        (df["lower_shadow"] < 0.1)
    ).astype(float)

    # 상승장악형(Bullish Engulfing): 이전 음봉을 현재 양봉이 완전히 덮음
    prev_open  = open_.shift(1)
    prev_close = close.shift(1)
    df["is_bullish_engulf"] = (
        (close > open_) &          # 현재 양봉
        (prev_close < prev_open) & # 이전 음봉
        (open_ < prev_close) &     # 현재 시가 < 이전 종가
        (close > prev_open)        # 현재 종가 > 이전 시가
    ).astype(float)

    # 하락장악형(Bearish Engulfing): 이전 양봉을 현재 음봉이 완전히 덮음
    df["is_bearish_engulf"] = (
        (close < open_) &          # 현재 음봉
        (prev_close > prev_open) & # 이전 양봉
        (open_ > prev_close) &     # 현재 시가 > 이전 종가
        (close < prev_open)        # 현재 종가 < 이전 시가
    ).astype(float)

    # 마루보쥬(Marubozu): 꼬리 거의 없는 긴 몸통 (몸통 90%+)
    df["is_marubozu_bull"] = ((df["body_ratio"] >= 0.9) & (close > open_)).astype(float)
    df["is_marubozu_bear"] = ((df["body_ratio"] >= 0.9) & (close < open_)).astype(float)

    # 연속 상승/하락 봉 수 (최대 5봉까지)
    is_up   = (close > close.shift(1)).astype(int)
    is_down = (close < close.shift(1)).astype(int)

    def _count_consecutive(series: pd.Series) -> pd.Series:
        result = pd.Series(0, index=series.index)
        count  = pd.Series(0, index=series.index)
        for i in range(1, min(6, len(series))):
            count += series.shift(i).fillna(0)
            # 끊기면 중단 (누적이 아닌 연속 체크)
        # 간단 버전: rolling sum (연속 여부 근사)
        return series.rolling(5, min_periods=1).sum()

    df["consecutive_up"]   = is_up.rolling(5, min_periods=1).sum()
    df["consecutive_down"] = is_down.rolling(5, min_periods=1).sum()

    return df[FEATURE_NAMES].dropna()
