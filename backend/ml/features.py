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
    # ── 래그 피처 (XGBoost에 시계열 맥락 제공) ────────────────────
    "rsi_lag_1",        # 1봉 전 RSI
    "rsi_lag_2",        # 2봉 전 RSI
    "macd_diff_lag_1",  # 1봉 전 MACD 다이버전스
    "vol_ratio_lag_1",  # 1봉 전 거래량 비율
    "bb_pband_lag_1",   # 1봉 전 볼린저밴드 위치
    "ret_lag_1",        # 1봉 전 수익률
    "ret_lag_2",        # 2봉 전 수익률
    "ret_lag_3",        # 3봉 전 수익률
    # ── 펀딩비 (코인: 8시간 주기 실제값, 주식: 항상 0) ─────────────
    "funding_rate",
    # ── 시장 레짐 & 구조 피처 ────────────────────────────────────────
    "regime_state",         # 0=횡보, 1=추세, 2=고변동 (ADX+BB폭 복합)
    "anchored_vwap_ratio",  # close / 당일시가VWAP - 1 (기관 기준선)
    "anchored_vwap_cross",  # close > 당일 앵커드VWAP → 1
    "bb_squeeze",           # BB폭 수축 중 → 1 (돌파 직전 포착)
    "mtf_ema_bull",         # 15분봉 EMA9 > EMA21 → 1 (상위 추세 강세)
]


def compute_features(
    ohlcv_list: list[dict],
    funding_rates: list[dict] | None = None,
) -> pd.DataFrame:
    """OHLCV 리스트 → Feature DataFrame 변환.

    Args:
        ohlcv_list:    API 반환 형식 (최신 날짜 앞, 내림차순)
        funding_rates: 바이낸스 펀딩비 히스토리 (코인만, 주식 None → 0으로 채움)
                       형식: [{"fundingTime": ms타임스탬프, "fundingRate": "0.0001"}, ...]

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

    # ── 앵커드 VWAP (당일 시가 기준, 기관 매매 참조선) ──────────
    _tp_av     = (df["high"] + df["low"] + df["close"]) / 3
    _date_grp  = df.index.date
    _avwap     = (
        (_tp_av * df["volume"]).groupby(_date_grp).cumsum()
        / df["volume"].groupby(_date_grp).cumsum().replace(0, np.nan)
    )
    df["anchored_vwap_ratio"] = (close / _avwap.fillna(close) - 1)
    df["anchored_vwap_cross"] = (close > _avwap).astype(float)

    # ── 변동성 ────────────────────────────────────────────────────
    bb             = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()
    atr            = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"]  = atr / close
    df["mass_index"]  = trend.MassIndex(high, low, window_fast=9, window_slow=25).mass_index()
    df["hl_range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close

    # BB폭 지표 (regime & squeeze에 공유)
    _bb_wband    = bb.bollinger_wband()          # (상단-하단)/중간선
    _bb_wband_ma = _bb_wband.rolling(20).mean()
    df["bb_squeeze"] = (_bb_wband <= _bb_wband_ma * 0.85).astype(float)  # 폭 수축 → 돌파 대기

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
    df["consecutive_up"]   = is_up.rolling(5, min_periods=1).sum()
    df["consecutive_down"] = is_down.rolling(5, min_periods=1).sum()

    # ── 래그 피처 (XGBoost에 시계열 맥락 제공) ───────────────────────
    df["rsi_lag_1"]       = df["rsi_9"].shift(1)
    df["rsi_lag_2"]       = df["rsi_9"].shift(2)
    df["macd_diff_lag_1"] = df["macd_diff"].shift(1)
    df["vol_ratio_lag_1"] = df["vol_ratio"].shift(1)
    df["bb_pband_lag_1"]  = df["bb_pband"].shift(1)
    df["ret_lag_1"]       = close.pct_change(1).shift(1)
    df["ret_lag_2"]       = close.pct_change(1).shift(2)
    df["ret_lag_3"]       = close.pct_change(1).shift(3)

    # ── 시장 레짐 (0=횡보, 1=추세, 2=고변동) ────────────────────
    _regime = pd.Series(0.0, index=df.index)
    _regime[df["adx_14"] > 25] = 1.0                     # ADX>25: 추세장
    _regime[_bb_wband > _bb_wband_ma * 1.5] = 2.0        # BB폭 급팽창: 고변동장 (추세장 override)
    df["regime_state"] = _regime

    # ── MTF 15분봉 상위 추세 (5분봉 리샘플 → EMA 크로스) ────────
    try:
        _df15 = df[["open", "high", "low", "close", "volume"]].resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        if len(_df15) >= 22:
            _e9  = trend.EMAIndicator(_df15["close"], window=9).ema_indicator()
            _e21 = trend.EMAIndicator(_df15["close"], window=21).ema_indicator()
            _mtf = (_e9 > _e21).astype(float)
            df["mtf_ema_bull"] = _mtf.reindex(df.index, method="ffill").fillna(0.5)
        else:
            df["mtf_ema_bull"] = 0.5
    except Exception:
        df["mtf_ema_bull"] = 0.5

    # ── 펀딩비 (코인: 8시간 주기 forward-fill, 주식: 항상 0) ─────────
    if funding_rates:
        try:
            fr_df = pd.DataFrame(funding_rates)
            # UTC ms → KST naive (df.index와 타임존 일치시켜야 reindex 매칭됨)
            fr_df["ts"] = pd.to_datetime(
                fr_df["fundingTime"].astype(float), unit="ms", utc=True
            ).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
            fr_df = fr_df.set_index("ts").sort_index()
            fr_series = fr_df["fundingRate"].astype(float)
            df["funding_rate"] = fr_series.reindex(df.index, method="ffill").fillna(0.0)
        except Exception:
            df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = 0.0

    return df[FEATURE_NAMES].dropna()
