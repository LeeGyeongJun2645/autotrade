"""기술지표 Feature 계산 모듈.

OHLCV 리스트 → Feature DataFrame 반환.
XGBoost 학습 및 예측에서 공통 사용.
"""

import warnings
import numpy as np
import pandas as pd
from ta import momentum, trend, volatility, volume

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
    "vortex_diff",
    "ema9_cross_ema21",  # EMA 9/21 골든·데드 크로스
    # ── VWAP ──────────────────────────────
    "vwap_ratio",        # (close/VWAP20) - 1 : 현재가 vs 거래량가중평균 괴리율
    "vwap_cross",        # close > VWAP20 이면 1 (추세 방향)
    "vwap_upper2_dist",  # (close/VWAP+2σ) - 1: 상단 밴드까지 거리 (음수=이미 돌파)
    "vwap_lower2_dist",  # (close/VWAP-2σ) - 1: 하단 밴드 대비 위치 (양수=지지 위)
    # ── 변동성 ────────────────────────────
    "bb_pband",
    "bb_width_pct",      # BB폭의 252봉 롤링 백분위 (0=최저변동성, 1=최고변동성) — 레짐 강도
    "atr_pct",
    "rv_ratio",          # 단기(5봉) / 중기(20봉) 실현변동성 비율 — 변동성 레짐 탐지
    "mass_index",
    "hl_range_20",
    # ── 거래량 ────────────────────────────
    "vol_ratio",
    "vol_surge",
    "vol_surge_flag",
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
    "is_inverted_hammer",  # 역망치형: 저점 반전 시도
    "is_morning_star",     # 아침별형: 3봉 반전 패턴
    "is_piercing",         # 관통형: 이전 음봉 50% 이상 회복
    "is_pullback_to_ma20",      # 눌림목: 정배열+MA20 근처 반등 시작
    "close_reclaimed_ma20",     # MA20 회복봉: 아래서 위로 돌파 (지지선 회복 확인)
    "close_reclaimed_ma60",     # MA60 회복봉: 중기 지지선 회복
    "vol_reversal_signal",      # 거래량 반전: 2봉 감소→현재 급증 (반등 임박)
    "bullish_count_3",          # 최근 3봉 중 양봉 개수 (3틱룰 기반 다중 확인)
    "bearish_count_3",          # 최근 3봉 중 음봉 개수
    "swing_near_support",       # 최근 20봉 저점 대비 현재가 거리 (동적 지지선)
    "swing_near_resistance",    # 최근 20봉 고점 대비 현재가 거리 (동적 저항선)
    "near_support_3pct",        # 동적 지지선 3% 이내 → 매수 유리 구간
    "near_resistance_2pct",     # 동적 저항선 2% 이내 → 익절 준비 구간
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
    "regime_state",         # 0=하락, 1=횡보, 2=상승, 3=고변동 (ADX+DI 기반)
    "anchored_vwap_ratio",  # close / 당일시가VWAP - 1 (기관 기준선)
    "anchored_vwap_cross",  # close > 당일 앵커드VWAP → 1
    "bb_squeeze",           # BB폭 수축 중 → 1 (돌파 직전 포착)
    "mtf_ema_bull",         # 15분봉 EMA9 > EMA21 → 1 (상위 추세 강세)
    "mtf_1h_ema_bull",      # 1시간봉 EMA9 > EMA21 → 1 (더 높은 상위 추세 필터)
    # ── 시간대 패턴 (KST 기준) ───────────────────────────────────────
    "hour_sin",             # 시간 sin 인코딩 (0~24h 주기 연속성)
    "hour_cos",             # 시간 cos 인코딩
    "is_opening_hour",      # 09:00~10:00 (장 시작 급변동)
    "is_lunch_hour",        # 11:30~13:00 (유동성 저하)
    "is_closing_hour",      # 14:30~15:30 (마감 정리)
    # ── BTC 상관관계 (코인 에이전트 전용, 주식·BTC자신은 0) ──────────
    "btc_corr_20",          # 최근 20봉 BTC 상관계수 (동조화 강도)
    "btc_ret_5",            # BTC 5봉 수익률 (알트 선행 지표)
    # ── KOSPI 연동 (주식 에이전트 전용, 코인은 0) ────────────────────
    "kospi_ret_5",          # KOSPI 5봉 수익률 (시장 방향)
    "kospi_rel_str",        # 개별주 상대강도 (종목ret5 - KOSPI ret5)
    # ── OI & Taker (코인 에이전트 전용, BTC 선물 기준 / 주식은 0) ──
    "oi_change_pct",        # OI 5봉 변화율 (포지션 쌓임 = 추세 확신)
    "oi_price_diverge",     # OI↑+가격↓=숏증가(약세) / OI↑+가격↑=롱증가(강세) 불일치 여부
    "taker_buy_ratio",      # Taker 매수 비율 (>0.5 = 공격적 매수 우세)
    "taker_ma20",           # Taker 비율 20봉 MA (원시값 노이즈 제거)
    # ── 헤이킨 아시 (노이즈 제거 캔들, 추세 지속성 포착) ───────────
    "ha_color",             # HA 봉 색깔 (1=양봉, 0=음봉)
    "ha_no_lower_shadow",   # HA 아래꼬리 없음 → 강한 상승 지속 신호
    "ha_no_upper_shadow",   # HA 위꼬리 없음 → 강한 하락 지속 신호
    "ha_bull_streak",       # 연속 HA 양봉 수 (최대 5봉, 모멘텀 강도)
    # ── 이치모쿠 클라우드 ────────────────────────────────────────────
    "ichi_above_cloud",        # close > 구름 상단 (강세 구조)
    "ichi_cloud_green",        # 선행스팬A > B → 상승 구름
    "ichi_tenkan_kijun_bull",  # 전환선(9봉) > 기준선(26봉) → 단기 골든크로스
    # ── 일별 피봇 포인트 (전일 H/L/C 기준) ──────────────────────────
    "dist_to_pp",   # (close/PP) - 1: 피봇 대비 현재가 위치
    "dist_to_r1",   # (R1/close) - 1: 익절 목표까지 남은 거리
    "dist_to_s1",   # (close/S1) - 1: 지지선 대비 여유
    "above_pp",     # close > PP → 1 (추세 방향)
    # ── Order Flow Imbalance Proxy ──────────────────────────────────
    "ofi_5",        # 5봉 방향성 볼륨 비율 (-1~1, 양수=매수우세)
    "ofi_20",       # 20봉 OFI
    "cvd_ratio",    # 60봉 누적 볼륨 델타 비율
    "obi_mean",     # 오더북 잔량 불균형 평균 (bid-ask)/(bid+ask), 코인전용 · 주식=0
    # ── Yang-Zhang 변동성 (overnight gap 처리, GK 대비 14배 효율) ──────
    "gk_vol",       # Yang-Zhang vol: open-gap + intraday + Rogers-Satchell 합성
    # ── 멀티타임프레임 수익률 ────────────────────────────────────────
    "mtf_ret_15m",  # 15분 수익률 (3봉)
    "mtf_ret_1h",   # 1시간 수익률 (12봉)
    "mtf_ret_4h",   # 4시간 수익률 (48봉)
    "mtf_align",    # 현재봉 방향과 상위 TF 3개 일치 수 (0~3)
    # ── ORB (Opening Range Breakout) — 주식 전용, 코인=0 ──────────
    "orb_position",    # (close - orb_low) / (orb_high - orb_low): 09:00~09:30 범위 내 위치 (0~1)
    "orb_breakout",    # 1=상단 돌파 / -1=하단 돌파 / 0=범위 내
    "orb_high_pct",    # close / orb_high - 1: 상단 대비 거리
    "orb_vol_confirm", # ORB 상단 돌파 AND 거래량 1.5배 이상 동반 (고확률 ORB 필터)
    # ── 장중 세션 구조 — 주식 전용, 코인=0 ──────────────────────────
    "session_phase",      # 0=개장(~09:30) / 1=오전 / 2=점심 / 3=오후마감 (기존 3개 이진 대비 압축)
    "ret_since_open",     # 당일 시가 대비 현재 수익률 (장중 누적 모멘텀)
    "intraday_reversal",  # 오전 강세(+1%↑) AND 마감 구간 → 반전 패턴 (1=반전 가능성)
    # ── 뉴스 감성 ────────────────────────────────────────────────────
    "news_score",  # 종목 뉴스 감성 점수 -1.0~+1.0 (캐시 없으면 0.0)
    # ── SuperTrend (동적 ATR 방향 지표) ─────────────────────────────
    "supertrend_bull",  # 1=상승 추세(SuperTrend 아래), 0=하락 추세(SuperTrend 위)
    # ── 미국 시장 세션 ────────────────────────────────────────────────
    "is_us_session",  # UTC 13~21시(KST 22~06시) = 1: 미국 시장 최고 유동성 구간
    # ── CVD 다이버전스 ────────────────────────────────────────────────
    "cvd_div_5",         # 5봉 가격 방향 vs CVD 방향 불일치 (단기 다이버전스)
    "cvd_div_20",        # 20봉 가격 방향 vs CVD 방향 불일치 (중기 다이버전스)
    # ── BB + Keltner Channel 스퀴즈 ──────────────────────────────────
    "bb_inside_kc",      # BB 상하단이 KC 안쪽 → 스퀴즈 (강한 움직임 대기)
    "kc_squeeze_bars",   # 스퀴즈 지속 봉 수 (많을수록 에너지 축적)
    # ── StochRSI K/D 교차 ────────────────────────────────────────────
    "stoch_rsi_k",       # StochRSI K선 (3봉 SMA of raw stochrsi)
    "stoch_rsi_d",       # StochRSI D선 (3봉 SMA of K)
    "stoch_rsi_cross",   # K가 D를 상향 돌파 → 1 (단기 모멘텀 전환)
    # ── 코인 ORB (UTC 00:00 기준, 첫 30분 범위) ──────────────────────
    "coin_orb_pos",      # (close - 00:00 저가) / (고가 - 저가): 0~1 위치
    "coin_orb_bull",     # 코인 ORB 상단 돌파 → 1
    # ── EMA 5-8-13 Ribbon ────────────────────────────────────────────
    "ema_align_score",   # EMA5>EMA8>EMA13 순 정렬 점수 (0~3, 3=완전 정배열)
    "ema_spread_ratio",  # (EMA5 - EMA13) / EMA13: ribbon 펼침 강도
    # ── RSI 히든 다이버전스 ──────────────────────────────────────────
    "hidden_bull_div",   # RSI 저점↑ + 가격 저점↓ → 숨은 상승 다이버전스 (추세 지속)
    "hidden_bear_div",   # RSI 고점↓ + 가격 고점↑ → 숨은 하락 다이버전스 (추세 지속)
    # ── Volume Profile POC 거리 ──────────────────────────────────────
    "dist_to_poc_pct",   # (close / POC가격 - 1): 60봉 거래량 가중 POC 대비 거리
    "above_poc",         # close > POC → 1 (지지 vs 저항 판단)
    # ── 롱/숏 비율 (코인 전용, BTC 선물 기준) ───────────────────────
    "ls_ratio",          # 롱 계좌 비율 (0.5=균등, >0.65=과도한 롱)
    "ls_extreme_long",   # ls_ratio > 0.65 → 1 (과열 롱, 역추세 신호)
    "ls_extreme_short",  # ls_ratio < 0.35 → 1 (과열 숏, 반등 신호)
    # ── ICT Premium / Discount Zone ──────────────────────────────────────
    "premium_discount",  # (close-60봉저점)/(60봉고점-저점): 0=저가구간, 1=고가구간
    # ── 이중바닥 (W 패턴) ────────────────────────────────────────────────
    "double_bottom_signal",  # 40봉 내 두 유사저점+RSI 다이버전스+넥라인 돌파 → 반전 신호
    # ── 피보나치 되돌림 레벨 근접 ────────────────────────────────────────
    "fib_382_support",  # 60봉 고저 기준 38.2% 되돌림 ±1.5% 이내
    "fib_50_support",   # 50% 되돌림 ±1.5% 이내
    "fib_618_support",  # 61.8% 되돌림 ±1.5% 이내 (가장 강력한 지지)
    # ── 이평선 정배열/역배열 ─────────────────────────────────────────────
    "ma_bull_align",  # MA5 > MA20 > MA60 완전 정배열 (매수 우위 구조)
    "ma_bear_align",  # MA5 < MA20 < MA60 완전 역배열 (매도 우위 구조)
    # ── 매도소진 고신뢰 반등 ─────────────────────────────────────────────
    "exhaustion_bounce",  # vol_reversal_signal AND RSI<35 → 반등 고신뢰 구간
    # ── 가짜 돌파 경고 ───────────────────────────────────────────────────
    "false_breakout_warn",  # 전봉 저항 돌파 후 현재봉 복귀 → 휩쏘 경고
    # ── 지지선 기울기 ────────────────────────────────────────────────────
    "support_slope",  # 최근 10봉 저가 선형회귀 기울기 (양수=상승지지선, 음수=하락)
    # ── RSI(2) 역추세 — Larry Connors 검증 전략 ──────────────────────────
    "rsi_2",             # 2기간 RSI: 극단 과매도(<10)/과매수(>90) 포착용
    "rsi_2_extreme_low", # rsi_2 < 10 → 극단 과매도, 역추세 반등 신호
    "rsi_2_extreme_hi",  # rsi_2 > 90 → 극단 과매수, 역추세 하락 신호
    # ── Connors RSI (CRSI) — 3요소 복합 역추세 지표 ─────────────────────
    "crsi",              # [RSI(3) + streak_RSI(2) + ROC백분위(100)] / 3, <10=강한매수
    # ── Donchian Channel — 거북이 트레이딩 돌파 신호 ─────────────────────
    "dc_position",       # (close - 20봉저가)/(20봉고가-20봉저가): 채널 내 위치 0~1
    "dc_upper_break",    # close >= 20봉 최고가 → 1 (거북이 상향돌파)
    "dc_lower_break",    # close <= 20봉 최저가 → 1 (거북이 하향돌파)
    # ── Z-Score 편차 — 통계적 평균회귀 신호 ─────────────────────────────
    "zscore_20",         # (close - MA20) / std20: -3~+3, 극단값=역추세 진입
    # ── Keltner Channel 이탈 — 역추세 과매도/과매수 신호 ────────────────
    "kc_lower_breach",   # close < KC 하단 → 1 (역추세 매수 후보, 77% 백테스트)
    "kc_upper_breach",   # close > KC 상단 → 1 (역추세 매도 후보)
    # ── 갭(Gap) 피처 — 갭업/갭다운 방향성 및 갭필 신호 ──────────────────
    "gap_pct",           # (open - prev_close) / prev_close: 갭 크기
    "gap_up",            # gap_pct > 0.005 → 1 (0.5% 이상 갭업)
    "gap_down",          # gap_pct < -0.005 → 1 (0.5% 이상 갭다운)
    # ── 선형회귀 채널 — 통계적 추세 채널 편차 ───────────────────────────
    "lrc_deviation",     # (close - 20봉 선형회귀 예측값) / close: 회귀선 대비 편차
    "lrc_slope",         # 20봉 선형회귀 정규화 기울기 (양수=상승추세)
    # ── 이중천장(Double Top / M 패턴) — 이중바닥의 반대 ─────────────────
    "double_top_signal", # 40봉 내 두 유사고점+RSI 부정 다이버전스+넥라인 하향돌파
    # ── 변동성 돌파 k값 신호 — Larry Williams ML 피처화 ──────────────────
    "vb_signal",         # open + (prev_high - prev_low) * 0.5 초과 시 1
    # ── 추세선 피처 — 스윙포인트 기반 동적 추세선 ──────────────────────
    "tl_support_dist",   # 상승 추세선(저점 연결) 대비 현재가 거리 % (음수=이탈)
    "tl_resist_dist",    # 하락 추세선(고점 연결) 대비 현재가 거리 % (음수=저항 아래)
    "tl_near_bounce",    # 추세선 ±1.5% 이내 접근 → 1 (반발 매매 구간)
    # ── Volume X-Ray (캔들별 매수/매도 압력 분석) ───────────────────────
    "vol_pressure_bull",  # 최근 5봉 양봉 볼륨 비율 (0~1, >0.6=매수 지배)
    "taker_delta",        # (taker_buy_ratio - 0.5) × 2: -1~+1 정규화 매수/매도 불균형
    "vol_xray_diverge",   # 가격 방향 vs 볼륨 압력 불일치 (1=다이버전스 경고)
]


def compute_features(
    ohlcv_list: list[dict],
    funding_rates: list[dict] | None = None,
    btc_ohlcv: list[dict] | None = None,
    kospi_ohlcv: list[dict] | None = None,
    oi_hist: list[dict] | None = None,
    taker_hist: list[dict] | None = None,
    ticker: str = "",                          # 뉴스 감성 조회용 종목 코드
    ls_hist: list[dict] | None = None,         # 바이낸스 글로벌 L/S 비율 히스토리
    obi_snaps: list[float] | None = None,      # 60분봉 내 오더북 OBI 스냅샷 리스트 (코인전용)
    interval_min: int = 5,                     # 봉 단위 (5분봉=5, 60분봉=60) — MTF 봉수 계산용
) -> pd.DataFrame:
    """OHLCV 리스트 → Feature DataFrame 변환.

    Args:
        ohlcv_list:    API 반환 형식 (최신 날짜 앞, 내림차순)
        funding_rates: 바이낸스 펀딩비 히스토리 (코인만, 주식 None → 0으로 채움)
                       형식: [{"fundingTime": ms타임스탬프, "fundingRate": "0.0001"}, ...]

    Returns:
        FEATURE_NAMES 컬럼만 가진 DataFrame (NaN 행 제거됨)
    """
    if not ohlcv_list:
        return pd.DataFrame(columns=FEATURE_NAMES)
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
    # ── StochRSI K/D (3봉 SMA 평활, 교차 신호) ─────────────────────
    _srsi_k = df["stoch_rsi"].rolling(3, min_periods=1).mean()
    _srsi_d = _srsi_k.rolling(3, min_periods=1).mean()
    df["stoch_rsi_k"]    = _srsi_k.fillna(0.5)
    df["stoch_rsi_d"]    = _srsi_d.fillna(0.5)
    df["stoch_rsi_cross"] = ((_srsi_k > _srsi_d) & (_srsi_k.shift(1) <= _srsi_d.shift(1))).astype(float)
    df["williams_r"]= momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df["mfi_14"]    = volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    df["roc_10"]    = momentum.ROCIndicator(close, window=10).roc()
    df["ret_1d"]    = close.pct_change(1)
    df["ret_3d"]    = close.pct_change(3)
    df["ret_5d"]    = close.pct_change(5)
    df["ret_10d"]   = close.pct_change(10)
    df["ret_20d"]   = close.pct_change(20)

    # ── RSI 히든 다이버전스 (20봉 lookback) ─────────────────────────
    try:
        _rsi_hd = df["rsi_9"]
        _lb     = 20
        _price_low_prev  = low.rolling(_lb).min().shift(1)
        _rsi_low_prev    = _rsi_hd.rolling(_lb).min().shift(1)
        _price_high_prev = high.rolling(_lb).max().shift(1)
        _rsi_high_prev   = _rsi_hd.rolling(_lb).max().shift(1)
        # 숨은 상승: 가격 저점↓ + RSI 저점↑ + RSI < 50 (하락 후 추세 지속 예상)
        df["hidden_bull_div"] = (
            (low < _price_low_prev) & (_rsi_hd > _rsi_low_prev) & (_rsi_hd < 50)
        ).astype(float)
        # 숨은 하락: 가격 고점↑ + RSI 고점↓ + RSI > 50 (상승 후 추세 지속 예상)
        df["hidden_bear_div"] = (
            (high > _price_high_prev) & (_rsi_hd < _rsi_high_prev) & (_rsi_hd > 50)
        ).astype(float)
    except Exception:
        df["hidden_bull_div"] = 0.0
        df["hidden_bear_div"] = 0.0

    # ── 추세 ──────────────────────────────────────────────────────
    ma5  = trend.SMAIndicator(close, window=5).sma_indicator()
    ma20 = trend.SMAIndicator(close, window=20).sma_indicator()
    ma60 = trend.SMAIndicator(close, window=60).sma_indicator()
    df["ma5_ratio"]       = close / ma5.replace(0, np.nan)  - 1
    df["ma20_ratio"]      = close / ma20.replace(0, np.nan) - 1
    df["ma60_ratio"]      = close / ma60.replace(0, np.nan) - 1
    df["ma5_cross_ma20"]  = (ma5 > ma20).astype(float)
    df["ma20_cross_ma60"] = (ma20 > ma60).astype(float)

    # MA 회복봉: MA20/MA60 아래서 위로 돌파한 봉 (지지선 회복 = 반등 확인 시그널)
    # 실전: 이평선 아래에 있다가 종가 기준으로 다시 선 위에서 마감하는 봉이 진입 신호
    df["close_reclaimed_ma20"] = (
        (close > ma20) & (close.shift(1) <= ma20.shift(1))
    ).astype(float)
    df["close_reclaimed_ma60"] = (
        (close > ma60) & (close.shift(1) <= ma60.shift(1))
    ).astype(float)

    # 눌림목(Pullback to MA20): 정배열 구조에서 MA20 근처까지 내려온 후 반등 시작
    # 실전 트레이더들이 가장 많이 쓰는 진입 자리 — MA20 눌림 후 양봉 전환 시 매수
    df["is_pullback_to_ma20"] = (
        (df["ma5_cross_ma20"] == 1) &         # MA5 > MA20 (정배열 — 하락 구조 아님)
        (df["ma20_ratio"].abs() <= 0.015) &    # 현재가가 MA20 ±1.5% 이내 (눌린 자리)
        (close > close.shift(1))               # 직전봉 대비 상승 (반등 시작 확인)
    ).astype(float)

    adx_ind       = trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"]  = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()
    df["adx_neg"] = adx_ind.adx_neg()
    df["trix_15"] = trend.TRIXIndicator(close, window=15).trix()
    vortex        = trend.VortexIndicator(high, low, close, window=14)
    df["vortex_diff"] = vortex.vortex_indicator_pos() - vortex.vortex_indicator_neg()

    ema9   = trend.EMAIndicator(close, window=9).ema_indicator()
    ema21  = trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema9_cross_ema21"] = (ema9 > ema21).astype(float)

    # ── EMA 5-8-13 Ribbon (단기 추세 정렬 강도) ─────────────────────
    _ema5  = trend.EMAIndicator(close, window=5).ema_indicator()
    _ema8  = trend.EMAIndicator(close, window=8).ema_indicator()
    _ema13 = trend.EMAIndicator(close, window=13).ema_indicator()
    df["ema_align_score"]  = (
        (_ema5 > _ema8).astype(int) + (_ema8 > _ema13).astype(int) + (_ema5 > _ema13).astype(int)
    ).astype(float)
    df["ema_spread_ratio"] = ((_ema5 - _ema13) / _ema13.replace(0, np.nan)).fillna(0.0)

    # ── VWAP (20봉 롤링) + ±2σ 밴드 ─────────────────────────────
    typical_price = (high + low + close) / 3
    vwap_20       = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    df["vwap_ratio"] = close / vwap_20 - 1
    df["vwap_cross"] = (close > vwap_20).astype(float)
    _vwap_std        = typical_price.rolling(20).std().fillna(0)
    _vwap_upper2     = vwap_20 + 2 * _vwap_std
    _vwap_lower2     = vwap_20 - 2 * _vwap_std
    df["vwap_upper2_dist"] = (close / _vwap_upper2.replace(0, np.nan) - 1).fillna(0.0)
    df["vwap_lower2_dist"] = (close / _vwap_lower2.where(_vwap_lower2 > 0, np.nan) - 1).fillna(0.0)

    # ── 앵커드 VWAP (당일 시가 기준, 기관 매매 참조선) ──────────
    _tp_av     = (df["high"] + df["low"] + df["close"]) / 3
    _date_grp  = df.index.date
    _avwap     = (
        (_tp_av * df["volume"]).groupby(_date_grp).cumsum()
        / df["volume"].groupby(_date_grp).cumsum().replace(0, np.nan)
    )
    _avwap_filled = _avwap.fillna(close)
    df["anchored_vwap_ratio"] = (close / _avwap_filled - 1)
    df["anchored_vwap_cross"] = (close > _avwap_filled).astype(float)

    # ── 변동성 ────────────────────────────────────────────────────
    bb             = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()
    atr            = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"]  = atr / close.replace(0, np.nan)
    _rv5  = close.pct_change().rolling(5).std()
    _rv20 = close.pct_change().rolling(20).std()
    df["rv_ratio"] = (_rv5 / _rv20.replace(0, np.nan)).fillna(1.0)   # >1=변동성 확장, <1=수축
    df["mass_index"]  = trend.MassIndex(high, low, window_fast=9, window_slow=25).mass_index()
    df["hl_range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close.replace(0, np.nan)

    # BB폭 지표 (regime & squeeze에 공유)
    _bb_wband    = bb.bollinger_wband()          # (상단-하단)/중간선
    _bb_wband_ma = _bb_wband.rolling(20).mean()
    df["bb_squeeze"] = (_bb_wband <= _bb_wband_ma * 0.85).astype(float)  # 폭 수축 → 돌파 대기
    # BB폭 252봉 롤링 백분위: 0=최저변동성(레인징), 1=최고변동성(추세/돌파)
    df["bb_width_pct"] = _bb_wband.rolling(252, min_periods=20).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False
    ).fillna(0.5)

    # ── BB + Keltner Channel 이중 스퀴즈 (LazyBear 방식) ────────────
    # KC: EMA20 ± ATR×1.5 — BB가 KC 안에 들어올수록 강한 돌파 임박 신호
    _kc_mid    = trend.EMAIndicator(close, window=20).ema_indicator()
    _kc_upper  = _kc_mid + atr * 1.5
    _kc_lower  = _kc_mid - atr * 1.5
    _bb_upper  = bb.bollinger_hband()
    _bb_lower  = bb.bollinger_lband()
    _squeeze   = (_bb_upper <= _kc_upper) & (_bb_lower >= _kc_lower)
    df["bb_inside_kc"]    = _squeeze.astype(float)
    df["kc_squeeze_bars"] = _squeeze.astype(int).rolling(20, min_periods=1).sum().clip(0, 20).astype(float)

    # ── 거래량 ────────────────────────────────────────────────────
    vol_ma20          = vol.rolling(20).mean()
    df["vol_ratio"]   = vol / vol_ma20.replace(0, np.nan)
    _vol_ma5          = vol.rolling(5).mean()
    df["vol_surge"]   = (vol / _vol_ma5.replace(0, np.nan)).fillna(1.0)
    df["vol_surge_flag"] = (df["vol_surge"] >= 3).astype(float)  # 3배 이상 = 급증

    # 거래량 반전 패턴: 직전 2봉 연속 감소 후 현재 5봉 평균 대비 1.3배 급증
    # 실전: 하락하면서 거래량이 줄다가 → 급증하면 매도 소진 + 매수세 유입 = 반등 임박
    _vol_declining = (vol.shift(1) < vol.shift(2)) & (vol.shift(2) < vol.shift(3))
    _vol_surge_now = vol > _vol_ma5.shift(1) * 1.3
    df["vol_reversal_signal"] = (_vol_declining & _vol_surge_now).astype(float)

    df["cmf_20"]      = volume.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20).chaikin_money_flow()

    # ── Volume Profile POC (60봉 거래량 최다 가격대 대비 현재가 거리) ──
    try:
        _poc_win  = 60
        _n_bins   = 20
        _poc_arr  = np.full(len(df), np.nan)
        _c_arr    = close.values
        _v_arr    = vol.values
        for _pi in range(_poc_win - 1, len(df)):
            _sc = _c_arr[_pi - _poc_win + 1 : _pi + 1]
            _sv = _v_arr[_pi - _poc_win + 1 : _pi + 1]
            _lo_p, _hi_p = _sc.min(), _sc.max()
            if _hi_p <= _lo_p:
                _poc_arr[_pi] = _sc[-1]
                continue
            _bins     = np.linspace(_lo_p, _hi_p, _n_bins + 1)
            _bidx     = np.clip(np.digitize(_sc, _bins) - 1, 0, _n_bins - 1)
            _bin_vol  = np.bincount(_bidx, weights=_sv, minlength=_n_bins)
            _best_bin = int(_bin_vol.argmax())
            _poc_arr[_pi] = (_bins[_best_bin] + _bins[_best_bin + 1]) / 2
        _poc_s = pd.Series(_poc_arr, index=df.index)
        df["dist_to_poc_pct"] = (close / _poc_s.replace(0, np.nan) - 1).fillna(0.0)
        df["above_poc"]       = (close > _poc_s).astype(float)
    except Exception:
        df["dist_to_poc_pct"] = 0.0
        df["above_poc"]       = 0.5

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

    # 역망치형(Inverted Hammer): 저점 부근 위꼬리 긴 캔들 → 상승 반전 시도
    df["is_inverted_hammer"] = (
        (df["upper_shadow"] >= 0.5) &
        (df["body_ratio"] < 0.3) &
        (df["lower_shadow"] < 0.15)
    ).astype(float)

    # 아침별형(Morning Star): 음봉 → 소몸통별 → 양봉이 음봉의 절반 이상 회복
    _prev2_body = (open_.shift(2) - close.shift(2)).clip(lower=0)  # 2봉전 음봉 크기
    df["is_morning_star"] = (
        (close.shift(2) < open_.shift(2)) &                              # 2봉전: 음봉
        (df["body_ratio"].shift(1) < 0.30) &                             # 1봉전: 소몸통(별)
        (close > open_) &                                                  # 현재: 양봉
        (close > (close.shift(2) + open_.shift(2)) / 2)                  # 음봉 중간 이상 회복
    ).astype(float)

    # 관통형(Piercing Pattern): 이전 음봉의 50% 이상 회복하는 양봉
    _prev_mid = (prev_open + prev_close) / 2  # 이전봉 중간값
    df["is_piercing"] = (
        (prev_close < prev_open) &   # 이전: 음봉
        (close > open_) &            # 현재: 양봉
        (open_ < prev_close) &       # 갭다운 시작
        (close > _prev_mid) &        # 이전 음봉의 50% 이상 회복
        (close < prev_open)          # 완전 장악은 아님 (장악형과 구분)
    ).astype(float)

    # 연속 상승/하락 봉 수 (최대 5봉까지)
    is_up   = (close > close.shift(1)).astype(int)
    is_down = (close < close.shift(1)).astype(int)
    df["consecutive_up"]   = is_up.rolling(5, min_periods=1).sum()
    df["consecutive_down"] = is_down.rolling(5, min_periods=1).sum()

    # 3봉 방향 확인 (3틱룰): 최근 3봉 중 양봉/음봉 개수 — 다중 캔들 컨텍스트
    # 실전: 단봉 하나만 보지 않고 최소 3봉의 흐름을 봐야 방향성 신뢰도 높아짐
    _is_bull = (close > open_).astype(int)
    _is_bear = (close < open_).astype(int)
    df["bullish_count_3"]  = _is_bull.rolling(3, min_periods=1).sum().astype(float)
    df["bearish_count_3"]  = _is_bear.rolling(3, min_periods=1).sum().astype(float)

    # 동적 지지/저항선: 최근 20봉 스윙 저점/고점 기반 (실전 수평선 자동화)
    # 실전: 고점은 고점끼리, 저점은 저점끼리 이어서 지지/저항선 — 3번 이상 터치 시 신뢰도 높음
    _roll20_low  = low.rolling(20).min()
    _roll20_high = high.rolling(20).max()
    # 현재가 vs 20봉 저점 거리: 음수(이탈)는 NaN으로 처리해 near_support_3pct가 False되도록 함
    # clip(0, 0.20) 만 쓰면 이탈 시 0이 돼서 ≤0.03 조건이 True가 되는 버그 있음
    _raw_sup = close / _roll20_low.replace(0, np.nan) - 1
    df["swing_near_support"]    = _raw_sup.where(_raw_sup >= 0, other=np.nan).clip(upper=0.20).fillna(0.20)
    # 현재가가 최근 고점에서 얼마나 아래에 있는지 (0=저항선, 값 작을수록 저항선 근처)
    df["swing_near_resistance"] = (_roll20_high / close.replace(0, np.nan) - 1).clip(0, 0.20)
    # 지지선 3% 이내이면서 지지선 위 → 매수 유리 구간 (지지선 이탈 시 False)
    df["near_support_3pct"]     = (df["swing_near_support"] <= 0.03).astype(float)
    # 저항선 2% 이내 → 익절 준비 구간
    df["near_resistance_2pct"]  = (df["swing_near_resistance"] <= 0.02).astype(float)

    # ── 래그 피처 (XGBoost에 시계열 맥락 제공) ───────────────────────
    df["rsi_lag_1"]       = df["rsi_9"].shift(1)
    df["rsi_lag_2"]       = df["rsi_9"].shift(2)
    df["macd_diff_lag_1"] = df["macd_diff"].shift(1)
    df["vol_ratio_lag_1"] = df["vol_ratio"].shift(1)
    df["bb_pband_lag_1"]  = df["bb_pband"].shift(1)
    df["ret_lag_1"]       = close.pct_change(1).shift(1)
    df["ret_lag_2"]       = close.pct_change(1).shift(2)
    df["ret_lag_3"]       = close.pct_change(1).shift(3)

    # ── 시장 레짐 (0=하락추세, 1=횡보, 2=상승추세, 3=고변동) ──────
    # ADX + DI방향 + MA20 기울기로 방향 있는 레짐 감지
    # 기존: 추세 강도만 (ADX>25=추세). 문제: 하락 추세도 1로 분류 → 하락장 BUY 허용
    # 수정: +DI/-DI 비교 + MA20 기울기로 상승/하락 방향 분리
    _adx_trend = df["adx_14"] > 25
    try:
        _uptrend   = _adx_trend & (df["adx_pos"] > df["adx_neg"])
        _downtrend = _adx_trend & (df["adx_neg"] >= df["adx_pos"])
    except Exception:
        _uptrend   = _adx_trend
        _downtrend = pd.Series(False, index=df.index)
    _regime = pd.Series(1.0, index=df.index)   # 기본: 횡보
    _regime[_uptrend]   = 2.0   # 상승 추세
    _regime[_downtrend] = 0.0   # 하락 추세
    # 고변동은 상승/횡보에만 덮어씀 — 하락추세(0) 구간은 유지해 BUY 억제 보존
    _high_vol = _bb_wband > _bb_wband_ma * 1.5
    _regime[_high_vol & ~_downtrend] = 3.0
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
            df["mtf_ema_bull"] = _mtf.reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["mtf_ema_bull"] = 0.0
    except Exception:
        df["mtf_ema_bull"] = 0.0

    # ── MTF 1시간봉 상위 추세 (5분봉 리샘플 → EMA 크로스, 더 높은 타임프레임 필터) ──
    try:
        _df1h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        if len(_df1h) >= 22:
            _e9h  = trend.EMAIndicator(_df1h["close"], window=9).ema_indicator()
            _e21h = trend.EMAIndicator(_df1h["close"], window=21).ema_indicator()
            _mtf1h = (_e9h > _e21h).astype(float)
            df["mtf_1h_ema_bull"] = _mtf1h.reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["mtf_1h_ema_bull"] = 0.0
    except Exception:
        df["mtf_1h_ema_bull"] = 0.0

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

    # ── 시간대 패턴 (KST 기준) ───────────────────────────────────────
    _h = df.index.hour
    _m = df.index.minute
    _hour_frac = _h + _m / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * _hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * _hour_frac / 24)
    df["is_opening_hour"] = ((_h == 9) | ((_h == 10) & (_m == 0))).astype(float)
    df["is_lunch_hour"]   = (((_h == 11) & (_m >= 30)) | (_h == 12)).astype(float)
    df["is_closing_hour"] = (((_h == 14) & (_m >= 30)) | ((_h == 15) & (_m <= 30))).astype(float)

    # ── BTC 상관관계 (코인 전용: btc_ohlcv 전달 시 계산, else 0) ──────
    if btc_ohlcv:
        try:
            _btc = pd.DataFrame(list(reversed(btc_ohlcv)))
            _btc["date"] = pd.to_datetime(_btc["date"].astype(str).str[:19])
            _btc = _btc.set_index("date").sort_index()
            _btc_close = _btc["close"].astype(float).reindex(df.index, method="ffill")
            _btc_ret   = _btc_close.pct_change(1)
            _coin_ret  = close.pct_change(1)
            df["btc_corr_20"] = _btc_ret.rolling(20).corr(_coin_ret)
            df["btc_ret_5"]   = _btc_close.pct_change(5)
        except Exception:
            df["btc_corr_20"] = 0.0
            df["btc_ret_5"]   = 0.0
    else:
        df["btc_corr_20"] = 0.0
        df["btc_ret_5"]   = 0.0

    # ── KOSPI 연동 (주식 전용: kospi_ohlcv 전달 시 계산, else 0) ───────
    if kospi_ohlcv:
        try:
            _ki = pd.DataFrame(list(reversed(kospi_ohlcv)))
            _ki["date"] = pd.to_datetime(_ki["date"].astype(str).str[:19])
            _ki = _ki.set_index("date").sort_index()
            _ki_close   = _ki["close"].astype(float).reindex(df.index, method="ffill")
            # KOSPI 시간대와 주식 시간대 불일치 시 (전행 NaN) → 0으로 대체
            if _ki_close.isna().all():
                df["kospi_ret_5"]  = 0.0
                df["kospi_rel_str"] = 0.0
            else:
                _ki_ret5    = _ki_close.pct_change(5).fillna(0.0)
                _stock_ret5 = close.pct_change(5).fillna(0.0)
                df["kospi_ret_5"]  = _ki_ret5
                df["kospi_rel_str"] = _stock_ret5 - _ki_ret5
        except Exception:
            df["kospi_ret_5"]  = 0.0
            df["kospi_rel_str"] = 0.0
    else:
        df["kospi_ret_5"]  = 0.0
        df["kospi_rel_str"] = 0.0

    # ── OI 히스토리 (코인 전용: BTC 선물 기준, else 0) ──────────────
    if oi_hist:
        try:
            _oi = pd.DataFrame(oi_hist)
            _oi["ts"] = pd.to_datetime(
                _oi["timestamp"].astype(float), unit="ms", utc=True
            ).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
            _oi = _oi.set_index("ts").sort_index()
            _oi_val        = _oi["sumOpenInterest"].astype(float).reindex(df.index, method="ffill")
            df["oi_change_pct"]  = _oi_val.pct_change(5).fillna(0.0)
            _oi_pct        = _oi_val.pct_change(5)
            _price_pct     = close.pct_change(5)
            _oi_rising     = _oi_pct > 0
            _price_rising  = _price_pct > 0
            _both_valid    = _oi_pct.notna() & _price_pct.notna()
            df["oi_price_diverge"] = ((_oi_rising != _price_rising) & _both_valid).astype(float)
        except Exception:
            df["oi_change_pct"]    = 0.0
            df["oi_price_diverge"] = 0.0
    else:
        df["oi_change_pct"]    = 0.0
        df["oi_price_diverge"] = 0.0

    # ── Taker 비율 (코인 전용: BTC 선물 기준, else 0.5) ──────────────
    if taker_hist:
        try:
            _tk = pd.DataFrame(taker_hist)
            _tk["ts"] = pd.to_datetime(
                _tk["timestamp"].astype(float), unit="ms", utc=True
            ).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
            _tk = _tk.set_index("ts").sort_index()
            _buy   = _tk["buyVol"].astype(float)
            _sell  = _tk["sellVol"].astype(float)
            _ratio = _buy / (_buy + _sell).replace(0, np.nan)
            df["taker_buy_ratio"] = _ratio.reindex(df.index, method="ffill").fillna(0.5)
        except Exception:
            df["taker_buy_ratio"] = 0.5
    else:
        df["taker_buy_ratio"] = 0.5
    # 20봉 MA로 평활화 (원시 Taker 비율 노이즈 제거)
    df["taker_ma20"] = df["taker_buy_ratio"].rolling(20, min_periods=5).mean().fillna(0.5)

    # ── Volume X-Ray: 캔들별 매수/매도 압력 가시화 (TradingView Volume X-Ray 개념) ──
    # vol_pressure_bull: 최근 5봉 중 양봉(close>open) 캔들의 볼륨 비율
    _is_bull5 = (close > open_).astype(float)
    _bull_vol5 = (vol * _is_bull5).rolling(5, min_periods=1).sum()
    _total_vol5 = vol.rolling(5, min_periods=1).sum().replace(0, np.nan)
    df["vol_pressure_bull"] = (_bull_vol5 / _total_vol5).fillna(0.5)
    # taker_delta: taker_buy_ratio를 -1~+1로 정규화 (0=균형, +1=완전 매수지배)
    df["taker_delta"] = (df["taker_buy_ratio"] - 0.5) * 2.0
    # vol_xray_diverge: 가격 방향 vs 볼륨 압력 방향 불일치 탐지
    _price_dir5 = np.sign(close.pct_change(5))
    _vol_dir5   = np.sign(df["vol_pressure_bull"] - 0.5)
    df["vol_xray_diverge"] = (
        (_price_dir5 != _vol_dir5) & (_price_dir5 != 0) & (_vol_dir5 != 0)
    ).astype(float).fillna(0.0)

    # ── 롱/숏 비율 (코인 전용: BTC 선물 기준, else 0.5) ──────────────
    if ls_hist:
        try:
            _ls = pd.DataFrame(ls_hist)
            _ls["ts"] = pd.to_datetime(
                _ls["timestamp"].astype(float), unit="ms", utc=True
            ).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
            _ls = _ls.set_index("ts").sort_index()
            _lr = _ls["longAccount"].astype(float).reindex(df.index, method="ffill").fillna(0.5)
            df["ls_ratio"]         = _lr
            df["ls_extreme_long"]  = (_lr > 0.65).astype(float)
            df["ls_extreme_short"] = (_lr < 0.35).astype(float)
        except Exception:
            df["ls_ratio"]         = 0.5
            df["ls_extreme_long"]  = 0.0
            df["ls_extreme_short"] = 0.0
    else:
        df["ls_ratio"]         = 0.5
        df["ls_extreme_long"]  = 0.0
        df["ls_extreme_short"] = 0.0

    # ── 헤이킨 아시 (재귀 계산: ha_open = (prev_ha_open + prev_ha_close) / 2) ──
    _ha_close  = (open_ + high + low + close) / 4
    _hc_arr    = _ha_close.values
    _ho_arr    = np.empty(len(df))
    _ho_arr[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for _i in range(1, len(df)):
        _ho_arr[_i] = (_ho_arr[_i - 1] + _hc_arr[_i - 1]) / 2
    _ha_open = pd.Series(_ho_arr, index=df.index)
    _ha_high = pd.concat([high, _ha_open, _ha_close], axis=1).max(axis=1)
    _ha_low  = pd.concat([low,  _ha_open, _ha_close], axis=1).min(axis=1)
    df["ha_color"]           = (_ha_close > _ha_open).astype(float)
    df["ha_no_lower_shadow"] = ((_ha_open - _ha_low).abs()   < 1e-9).astype(float)
    df["ha_no_upper_shadow"] = ((_ha_high - _ha_close).abs() < 1e-9).astype(float)
    df["ha_bull_streak"]     = (_ha_close > _ha_open).astype(int).rolling(5, min_periods=1).sum().astype(float)

    # ── 이치모쿠 클라우드 ─────────────────────────────────────────────
    try:
        _ichi   = trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        _span_a = _ichi.ichimoku_a().ffill().fillna(close)
        _span_b = _ichi.ichimoku_b().ffill().fillna(close)
        _tenkan = _ichi.ichimoku_conversion_line().ffill().fillna(close)
        _kijun  = _ichi.ichimoku_base_line().ffill().fillna(close)
        _cloud_top = pd.concat([_span_a, _span_b], axis=1).max(axis=1)
        df["ichi_above_cloud"]       = (close > _cloud_top).astype(float)
        df["ichi_cloud_green"]       = (_span_a > _span_b).astype(float)
        df["ichi_tenkan_kijun_bull"] = (_tenkan > _kijun).astype(float)
    except Exception:
        df["ichi_above_cloud"]       = 0.0
        df["ichi_cloud_green"]       = 0.0
        df["ichi_tenkan_kijun_bull"] = 0.0

    # ── Order Flow Imbalance Proxy (논문: Sharpe 1.4→3.6 개선) ────────
    # OHLCV에서 방향성 볼륨 추정: 양봉=매수압력, 음봉=매도압력
    _vol_delta     = vol * np.sign(close - open_)
    _vr5           = vol.rolling(5).sum().replace(0, np.nan)
    _vr20          = vol.rolling(20).sum().replace(0, np.nan)
    _vr60          = vol.rolling(60).sum().replace(0, np.nan)
    df["ofi_5"]    = (_vol_delta.rolling(5).sum()  / _vr5).fillna(0.0)
    df["ofi_20"]   = (_vol_delta.rolling(20).sum() / _vr20).fillna(0.0)
    df["cvd_ratio"] = (_vol_delta.rolling(60).sum() / _vr60).fillna(0.0)
    # OBI: 오더북 잔량 불균형 (bid-ask)/(bid+ask), 코인전용 — 없으면 0.0
    df["obi_mean"] = float(np.mean(obi_snaps)) if obi_snaps else 0.0

    # ── CVD 다이버전스 (가격 방향 vs 누적 볼륨 델타 방향 불일치) ────
    _cvd_dir_5  = np.sign(df["cvd_ratio"] - df["cvd_ratio"].shift(5))
    _cvd_dir_20 = np.sign(df["cvd_ratio"] - df["cvd_ratio"].shift(20))
    _price_dir_5  = np.sign(close.pct_change(5))
    _price_dir_20 = np.sign(close.pct_change(20))
    df["cvd_div_5"]  = (
        (_price_dir_5 != _cvd_dir_5) & (_price_dir_5 != 0) & (_cvd_dir_5 != 0)
    ).astype(float)
    df["cvd_div_20"] = (
        (_price_dir_20 != _cvd_dir_20) & (_price_dir_20 != 0) & (_cvd_dir_20 != 0)
    ).astype(float)

    # ── Yang-Zhang 변동성 (overnight gap 포함, GK 대비 14배 효율) ──
    # 주식 overnight 갭 + crypto micro-gap 모두 포착 (GK는 연속 시장 가정)
    _yz_n       = 5
    _yz_open_r  = np.log((open_ / close.shift(1)).clip(lower=1e-9))
    _yz_close_r = np.log((close  / open_).clip(lower=1e-9))
    _rs_yz      = (np.log(high / close.clip(lower=1e-9)) * np.log(high / open_.clip(lower=1e-9)) +
                   np.log(low  / close.clip(lower=1e-9)) * np.log(low  / open_.clip(lower=1e-9)))
    _k_yz       = 0.34 / (1.34 + (_yz_n + 1) / (_yz_n - 1))
    df["gk_vol"] = np.sqrt(
        (_yz_open_r.rolling(_yz_n).var() + _k_yz * _yz_close_r.rolling(_yz_n).var() +
         (1 - _k_yz) * _rs_yz.rolling(_yz_n).mean()).clip(lower=0)
    ).fillna(0.0)

    # df.copy()로 100+ 컬럼 누적 단편화 해소 (PerformanceWarning 방지 — 이전 위치 L727에서 앞으로 이동)
    df = df.copy()

    # ── 멀티타임프레임 수익률 (interval_min 기반 봉수 동적 계산)
    # 60분봉: 15분→1봉, 1h→max(2,1)=2봉, 4h→max(3,4)=4봉으로 자동 스케일
    _bars_s = max(1, 15 // interval_min)
    _bars_m = max(_bars_s + 1, 60 // interval_min)
    _bars_l = max(_bars_m + 1, 240 // interval_min)
    df["mtf_ret_15m"] = close.pct_change(_bars_s).fillna(0.0)
    df["mtf_ret_1h"]  = close.pct_change(_bars_m).fillna(0.0)
    df["mtf_ret_4h"]  = close.pct_change(_bars_l).fillna(0.0)
    # mtf_align: 단기/중기/장기 방향이 모두 상승(bullish)인 개수 (0=전부하락, 3=전부상승)
    # 주의: _bars_s==1이면 _dir_1==_dir_15m 동어반복 → 현재 봉 방향 대신 각 TF 방향 직접 계산
    _dir_s = np.sign(df["mtf_ret_15m"])
    _dir_m = np.sign(df["mtf_ret_1h"])
    _dir_l = np.sign(df["mtf_ret_4h"])
    df["mtf_align"] = (
        (_dir_s > 0).astype(float) +
        (_dir_m > 0).astype(float) +
        (_dir_l > 0).astype(float)
    ).fillna(0.0)

    # ── 일별 피봇 포인트 (전일 H/L/C 기준) ──────────────────────────
    try:
        _ph = high.resample("1D", offset="9h").max().shift(1).reindex(df.index, method="ffill")
        _pl = low.resample("1D", offset="9h").min().shift(1).reindex(df.index, method="ffill")
        _pc = close.resample("1D", offset="9h").last().shift(1).reindex(df.index, method="ffill")
        _pp = (_ph + _pl + _pc) / 3
        _r1 = 2 * _pp - _pl
        _s1 = 2 * _pp - _ph
        df["dist_to_pp"] = (close / _pp.replace(0, np.nan) - 1).fillna(0.0)
        df["dist_to_r1"] = (_r1 / close.replace(0, np.nan) - 1).fillna(0.0)
        df["dist_to_s1"] = (close / _s1.where(_s1 > 0, np.nan) - 1).fillna(0.0)
        df["above_pp"]   = (close > _pp).astype(float)
    except Exception:
        df["dist_to_pp"] = 0.0
        df["dist_to_r1"] = 0.0
        df["dist_to_s1"] = 0.0
        df["above_pp"]   = 0.5

    # ── ORB + 장중 세션 피처 (주식 전용: kospi_ohlcv 있을 때만, 코인=0) ──
    # kospi_ohlcv 가 None 이면 코인 에이전트 → 전부 중립값
    if kospi_ohlcv is not None:
        try:
            _h_idx = df.index.hour
            _m_idx = df.index.minute
            _date_key = df.index.date

            # 당일 09:00~09:30 ORB 고가/저가
            _orb_mask = (_h_idx == 9) & (_m_idx < 30)
            _orb_h = df["high"].where(_orb_mask).groupby(_date_key).transform("max")
            _orb_l = df["low"].where(_orb_mask).groupby(_date_key).transform("min")
            # 09:30 이후 봉은 당일 ORB 값 유지 (forward fill 효과)
            _orb_h = _orb_h.groupby(_date_key).ffill()
            _orb_l = _orb_l.groupby(_date_key).ffill()
            _orb_range = (_orb_h - _orb_l).replace(0, np.nan)

            df["orb_position"] = ((close - _orb_l) / _orb_range).clip(0, 2).fillna(0.5)
            df["orb_breakout"] = 0.0
            df.loc[_orb_h.notna() & (close > _orb_h), "orb_breakout"] = 1.0
            df.loc[_orb_l.notna() & (close < _orb_l), "orb_breakout"] = -1.0
            df["orb_high_pct"] = (close / _orb_h.replace(0, np.nan) - 1).fillna(0.0)
            # ORB 거래량 확인: 상단 돌파 + 거래량 1.5배 이상 (연구 기준 150%+, 기존 200%는 과도)
            df["orb_vol_confirm"] = (
                (df["orb_breakout"] == 1.0) & (vol >= vol_ma20 * 1.5)
            ).astype(float)

            # session_phase: 0=개장(~09:30) / 1=오전(09:30~11:30) / 2=점심(11:30~13:00) / 3=오후(13:00~)
            _sp = pd.Series(0.0, index=df.index)
            _sp[((_h_idx == 9) & (_m_idx >= 30)) | (_h_idx == 10) | ((_h_idx == 11) & (_m_idx < 30))] = 1.0
            _sp[((_h_idx == 11) & (_m_idx >= 30)) | (_h_idx == 12)] = 2.0
            _sp[_h_idx >= 13] = 3.0
            df["session_phase"] = _sp

            # ret_since_open: 당일 시가(첫 봉 open) 대비 수익률
            _day_open = open_.groupby(_date_key).transform("first")
            df["ret_since_open"] = (close / _day_open.replace(0, np.nan) - 1).fillna(0.0)

            # intraday_reversal: 오전 강세(+1%↑) AND 오후 마감 구간 → 반전 신호
            _closing_zone = _h_idx >= 14
            df["intraday_reversal"] = ((df["ret_since_open"] > 0.01) & _closing_zone).astype(float)
            # 주식: 코인 ORB 피처 → 중립값 (try 성공 시)
            df["coin_orb_pos"]  = 0.5
            df["coin_orb_bull"] = 0.0
        except Exception:
            df = pd.concat([df, pd.DataFrame({
                "orb_position": 0.5, "orb_breakout": 0.0, "orb_high_pct": 0.0,
                "orb_vol_confirm": 0.0,
                "session_phase": 1.0, "ret_since_open": 0.0, "intraday_reversal": 0.0,
            }, index=df.index)], axis=1)
        # 주식: 코인 ORB 피처 → 중립값 (try/except 공통)
        df["coin_orb_pos"]  = 0.5
        df["coin_orb_bull"] = 0.0
    else:
        # 코인: 주식 ORB 피처 → 중립값, 코인 ORB (UTC 00:00 기준) 별도 계산
        df = pd.concat([df, pd.DataFrame({
            "orb_position": 0.5, "orb_breakout": 0.0, "orb_high_pct": 0.0,
            "orb_vol_confirm": 0.0,
            "session_phase": 1.0, "ret_since_open": 0.0, "intraday_reversal": 0.0,
        }, index=df.index)], axis=1)

        # ── 코인 ORB (UTC 00:00 기준 첫 30분 범위) ──────────────────
        try:
            _utc_h_c  = (df.index.hour - 9) % 24    # KST naive → UTC 근사
            _utc_m_c  = df.index.minute
            _date_utc = (df.index - pd.Timedelta(hours=9)).date
            _c_orb_mask = (_utc_h_c == 0) & (_utc_m_c < 30)
            _corb_h = df["high"].where(_c_orb_mask).groupby(_date_utc).transform("max")
            _corb_l = df["low"].where(_c_orb_mask).groupby(_date_utc).transform("min")
            _corb_h = _corb_h.groupby(_date_utc).ffill()
            _corb_l = _corb_l.groupby(_date_utc).ffill()
            _corb_range = (_corb_h - _corb_l).replace(0, np.nan)
            df["coin_orb_pos"]  = ((close - _corb_l) / _corb_range).clip(0, 2).fillna(0.5)
            df["coin_orb_bull"] = (close > _corb_h.fillna(0)).astype(float)
        except Exception:
            df["coin_orb_pos"]  = 0.5
            df["coin_orb_bull"] = 0.0

    # 뉴스 감성 점수 (캐시된 값, 없으면 0.0)
    try:
        from backend.ml.news import get_cached_score
        _ns = get_cached_score(ticker) if ticker else None
        df["news_score"] = float(_ns) if _ns is not None else 0.0
    except Exception:
        df["news_score"] = 0.0

    # ── SuperTrend (ATR 기반 동적 추세 방향 지표, period=10, mult=3.0) ─
    try:
        _st_atr = volatility.AverageTrueRange(high, low, close, window=10).average_true_range().values
        _hl2    = ((high + low) / 2).values
        _close_arr = close.values
        n = len(_close_arr)
        _basic_up = _hl2 + 3.0 * _st_atr
        _basic_dn = _hl2 - 3.0 * _st_atr
        _final_up = _basic_up.copy()
        _final_dn = _basic_dn.copy()
        _st_dir   = np.ones(n)
        for _i in range(1, n):
            if np.isnan(_st_atr[_i]):
                _final_up[_i] = _basic_up[_i]
                _final_dn[_i] = _basic_dn[_i]
                _st_dir[_i]   = _st_dir[_i - 1]
                continue
            _final_up[_i] = _basic_up[_i] if (_basic_up[_i] < _final_up[_i-1] or _close_arr[_i-1] > _final_up[_i-1]) else _final_up[_i-1]
            _final_dn[_i] = _basic_dn[_i] if (_basic_dn[_i] > _final_dn[_i-1] or _close_arr[_i-1] < _final_dn[_i-1]) else _final_dn[_i-1]
            if _st_dir[_i-1] == -1.0:
                _st_dir[_i] = 1.0 if _close_arr[_i] > _final_up[_i] else -1.0
            else:
                _st_dir[_i] = -1.0 if _close_arr[_i] < _final_dn[_i] else 1.0
        df["supertrend_bull"] = pd.Series((_st_dir == 1.0).astype(float), index=df.index)
    except Exception:
        df["supertrend_bull"] = 0.5

    # ── ICT Premium / Discount Zone (60봉 스윙 내 현재가 위치) ───────────
    # 0 = 저가구간(Discount, 매수 유리), 1 = 고가구간(Premium, 매도 유리)
    # 50% 이하에서 매수, 50% 이상에서 매도가 ICT 기본 원칙
    _sw_high = high.rolling(60, min_periods=10).max()
    _sw_low  = low.rolling(60, min_periods=10).min()
    _sw_rng  = (_sw_high - _sw_low).replace(0, np.nan)
    df["premium_discount"] = ((close - _sw_low) / _sw_rng).clip(0.0, 1.0).fillna(0.5)

    # ── 미국 시장 세션 피처 (UTC 13~21시 = KST 22~06시, 최고 유동성) ──
    # df.index는 KST naive → UTC 변환 후 시간 비교
    _utc_hour = (df.index.hour - 9) % 24   # KST → UTC 근사 (naive index 기준)
    df["is_us_session"] = ((_utc_hour >= 13) & (_utc_hour < 21)).astype(float)

    # ── 완전 정배열/역배열 ────────────────────────────────────────────
    df["ma_bull_align"] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)
    df["ma_bear_align"] = ((ma5 < ma20) & (ma20 < ma60)).astype(float)

    # ── 피보나치 되돌림 레벨 근접 신호 ──────────────────────────────────
    # 60봉 스윙 고점-저점 기준 38.2%/50%/61.8% 레벨 ±1.5% 이내
    try:
        _fib_high = high.rolling(60, min_periods=10).max()
        _fib_low  = low.rolling(60, min_periods=10).min()
        _fib_rng  = (_fib_high - _fib_low).replace(0, np.nan)
        _fib_382  = _fib_high - 0.382 * _fib_rng
        _fib_50   = _fib_high - 0.500 * _fib_rng
        _fib_618  = _fib_high - 0.618 * _fib_rng
        _tol = 0.015
        df["fib_382_support"] = (((close - _fib_382).abs() / _fib_382.replace(0, np.nan)) <= _tol).astype(float)
        df["fib_50_support"]  = (((close - _fib_50).abs()  / _fib_50.replace(0, np.nan))  <= _tol).astype(float)
        df["fib_618_support"] = (((close - _fib_618).abs() / _fib_618.replace(0, np.nan)) <= _tol).astype(float)
    except Exception:
        df["fib_382_support"] = 0.0
        df["fib_50_support"]  = 0.0
        df["fib_618_support"] = 0.0

    # ── 매도소진 고신뢰 반등 (거래량 반전 + RSI 과매도 동시) ────────────
    df["exhaustion_bounce"] = (
        (df["vol_reversal_signal"] == 1.0) & (df["rsi_9"] < 35)
    ).astype(float)

    # ── 가짜 돌파 경고 (이전봉 저항 돌파 후 현재봉 복귀 = 휩쏘) ─────────
    # 전봉 고가가 20봉 저항선 돌파했는데 현재 종가가 저항선 아래로 복귀
    try:
        _res_level = high.rolling(20).max().shift(2)   # 2봉 전까지의 20봉 최고점
        _prev_broke = high.shift(1) > _res_level       # 전봉 고가가 저항 돌파
        _curr_fail  = close < _res_level               # 현재 종가가 저항 아래 복귀
        df["false_breakout_warn"] = (_prev_broke & _curr_fail).astype(float)
    except Exception:
        df["false_breakout_warn"] = 0.0

    # ── 지지선 기울기 (최근 10봉 저가 선형회귀, 정규화된 기울기) ──────────
    # 양수 = 상승 지지선(고점 높아지는 구조), 음수 = 하락 지지선
    _sl_win = 10
    _x_sl   = np.arange(_sl_win, dtype=float)
    _sl_arr = np.full(len(df), np.nan)
    _low_v  = low.values
    for _si in range(_sl_win - 1, len(df)):
        try:
            _y_sl = _low_v[_si - _sl_win + 1: _si + 1]
            _mean_y = float(np.mean(_y_sl))
            if np.any(np.isnan(_y_sl)) or _mean_y == 0:
                continue
            _sl_arr[_si] = float(np.polyfit(_x_sl, _y_sl, 1)[0]) / _mean_y
        except Exception:
            pass  # 이 봉만 건너뜀, 이전 결과 유지
    df["support_slope"] = pd.Series(_sl_arr, index=df.index).fillna(0.0)

    # ── 이중바닥(Double Bottom / W 패턴) 감지 ───────────────────────────
    # 40봉 내 두 개의 유사저점(±3%) + RSI 긍정 다이버전스 + 넥라인 돌파 → 반전
    try:
        _lb_db   = 40
        _half_db = _lb_db // 2
        _db_sig  = np.zeros(len(df))
        _lv      = low.values
        _hv      = high.values
        _cv      = close.values
        _rv      = df["rsi_9"].values
        for _di in range(_lb_db, len(df)):
            _sl = _lv[_di - _lb_db: _di]
            _sh = _hv[_di - _lb_db: _di]
            _sr = _rv[_di - _lb_db: _di]
            # 1차 저점: 앞쪽 절반의 최소
            _i1  = int(np.argmin(_sl[:_half_db]))
            _v1  = _sl[_i1]
            # 2차 저점: 뒤쪽 절반의 최소
            _i2  = int(np.argmin(_sl[_half_db:])) + _half_db
            _v2  = _sl[_i2]
            # 두 저점 차이 3% 이내
            _avg = (_v1 + _v2) / 2 + 1e-9
            if abs(_v1 - _v2) / _avg > 0.03:
                continue
            # 두 저점 사이 넥라인(최고점) — _i1+1 부터 슬라이스해 바닥봉 wick 제외
            if _i2 <= _i1:
                continue
            _between = _sh[_i1 + 1: _i2]   # 바닥봉 자체 고가는 제외 (wick이 넥라인 과대평가 방지)
            if len(_between) == 0:
                continue
            _neckline = float(_between.max())
            if _neckline <= 0:
                continue
            # 현재 종가가 넥라인 위 (W 패턴 완성)
            if _cv[_di] <= _neckline:
                continue
            # RSI 긍정 다이버전스: 두 저점 모두 유효한 RSI 값일 때만 확인
            _rsi1, _rsi2 = _sr[_i1], _sr[_i2]
            if np.isfinite(_rsi1) and np.isfinite(_rsi2) and _rsi2 > _rsi1:
                _db_sig[_di] = 1.0
        df["double_bottom_signal"] = _db_sig
    except Exception:
        df["double_bottom_signal"] = 0.0

    # ── RSI(2) 역추세 — Larry Connors, 승률 70~80% (200MA 위 조건 결합 시) ──
    try:
        _rsi2 = momentum.RSIIndicator(close, window=2).rsi()
        df["rsi_2"]             = _rsi2
        df["rsi_2_extreme_low"] = (_rsi2 < 10).astype(float)
        df["rsi_2_extreme_hi"]  = (_rsi2 > 90).astype(float)
    except Exception:
        df["rsi_2"] = df["rsi_9"]
        df["rsi_2_extreme_low"] = 0.0
        df["rsi_2_extreme_hi"]  = 0.0

    # ── Connors RSI (CRSI) — RSI(3) + 연속방향 RSI(2) + ROC 100봉 백분위 ──
    try:
        _rsi3 = momentum.RSIIndicator(close, window=3).rsi()
        # 연속 상승/하락 기간 계산
        _up   = (close > close.shift(1)).astype(int)
        _dn   = (close < close.shift(1)).astype(int)
        _streak = pd.Series(0.0, index=close.index)
        for _si in range(1, len(close)):
            if _up.iloc[_si]:
                _streak.iloc[_si] = max(_streak.iloc[_si - 1] + 1, 1)
            elif _dn.iloc[_si]:
                _streak.iloc[_si] = min(_streak.iloc[_si - 1] - 1, -1)
        _streak_rsi = momentum.RSIIndicator(_streak, window=2).rsi()
        # ROC 100봉 백분위 (rank)
        _roc3 = momentum.ROCIndicator(close, window=3).roc()
        _roc_pct = _roc3.rolling(100, min_periods=20).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100, raw=False
        )
        df["crsi"] = (_rsi3 + _streak_rsi + _roc_pct) / 3
    except Exception:
        df["crsi"] = 50.0

    # ── Donchian Channel (20봉) — 거북이 트레이딩 돌파 신호 ───────────────
    try:
        _dc_high = high.rolling(20, min_periods=5).max()
        _dc_low  = low.rolling(20, min_periods=5).min()
        _dc_rng  = (_dc_high - _dc_low).replace(0, np.nan)
        df["dc_position"]   = ((close - _dc_low) / _dc_rng).fillna(0.5)
        df["dc_upper_break"] = (close >= _dc_high).astype(float)
        df["dc_lower_break"] = (close <= _dc_low).astype(float)
    except Exception:
        df["dc_position"]    = 0.5
        df["dc_upper_break"] = 0.0
        df["dc_lower_break"] = 0.0

    # ── Z-Score 편차 (20봉) — 통계적 평균회귀 이탈 강도 ─────────────────
    try:
        _ma20_z  = close.rolling(20, min_periods=10).mean()
        _std20_z = close.rolling(20, min_periods=10).std()
        df["zscore_20"] = ((close - _ma20_z) / _std20_z.replace(0, np.nan)).fillna(0.0).clip(-4, 4)
    except Exception:
        df["zscore_20"] = 0.0

    # ── Keltner Channel 이탈 신호 (역추세) — 백테스트 승률 77% ─────────────
    try:
        _kc_ema  = trend.EMAIndicator(close, window=20).ema_indicator()
        _kc_atr  = volatility.AverageTrueRange(high, low, close, window=10).average_true_range()
        _kc_mult = 1.5
        _kc_up   = _kc_ema + _kc_mult * _kc_atr
        _kc_dn   = _kc_ema - _kc_mult * _kc_atr
        df["kc_lower_breach"] = (close < _kc_dn).astype(float)
        df["kc_upper_breach"] = (close > _kc_up).astype(float)
    except Exception:
        df["kc_lower_breach"] = 0.0
        df["kc_upper_breach"] = 0.0

    # ── 갭(Gap) 피처 — 갭업/갭다운 방향 및 크기 ─────────────────────────
    try:
        _prev_close   = close.shift(1)
        _gap          = (open_ - _prev_close) / _prev_close.replace(0, np.nan)
        df["gap_pct"]  = _gap.fillna(0.0).clip(-0.15, 0.15)
        df["gap_up"]   = (_gap > 0.005).astype(float)
        df["gap_down"] = (_gap < -0.005).astype(float)
    except Exception:
        df["gap_pct"]  = 0.0
        df["gap_up"]   = 0.0
        df["gap_down"] = 0.0

    # ── 선형회귀 채널 편차 (20봉) — 통계적 추세 채널 ───────────────────
    try:
        _lrc_win  = 20
        _lrc_x    = np.arange(_lrc_win, dtype=float)
        _lrc_dev  = np.full(len(df), np.nan)
        _lrc_slp  = np.full(len(df), np.nan)
        _cv_lrc   = close.values
        for _li in range(_lrc_win - 1, len(df)):
            _y = _cv_lrc[_li - _lrc_win + 1: _li + 1]
            if np.any(np.isnan(_y)) or _y[-1] == 0:
                continue
            _coef   = np.polyfit(_lrc_x, _y, 1)
            _pred   = _coef[0] * (_lrc_win - 1) + _coef[1]
            _lrc_dev[_li] = (_cv_lrc[_li] - _pred) / (_cv_lrc[_li] + 1e-9)
            _lrc_slp[_li] = _coef[0] / (np.mean(_y) + 1e-9)
        df["lrc_deviation"] = pd.Series(_lrc_dev, index=df.index).fillna(0.0).clip(-0.1, 0.1)
        df["lrc_slope"]     = pd.Series(_lrc_slp, index=df.index).fillna(0.0).clip(-0.05, 0.05)
    except Exception:
        df["lrc_deviation"] = 0.0
        df["lrc_slope"]     = 0.0

    # ── 이중천장(Double Top / M 패턴) — double_bottom_signal의 반대 ────────
    try:
        _lb_dt   = 40
        _half_dt = _lb_dt // 2
        _dt_sig  = np.zeros(len(df))
        _hv_dt   = high.values
        _lv_dt   = low.values
        _cv_dt   = close.values
        _rv_dt   = df["rsi_9"].values
        for _di in range(_lb_dt, len(df)):
            _sh = _hv_dt[_di - _lb_dt: _di]
            _sl = _lv_dt[_di - _lb_dt: _di]
            _sr = _rv_dt[_di - _lb_dt: _di]
            # 1차 고점: 앞쪽 절반 최대
            _i1  = int(np.argmax(_sh[:_half_dt]))
            _v1  = _sh[_i1]
            # 2차 고점: 뒤쪽 절반 최대
            _i2  = int(np.argmax(_sh[_half_dt:])) + _half_dt
            _v2  = _sh[_i2]
            _avg = (_v1 + _v2) / 2 + 1e-9
            if abs(_v1 - _v2) / _avg > 0.03:
                continue
            if _i2 <= _i1:
                continue
            # 두 고점 사이 넥라인(최저점)
            _between = _sl[_i1 + 1: _i2]
            if len(_between) == 0:
                continue
            _neckline = float(_between.min())
            if _neckline <= 0:
                continue
            # 현재 종가가 넥라인 아래 (M 패턴 완성)
            if _cv_dt[_di] >= _neckline:
                continue
            # RSI 부정 다이버전스: 두 고점 RSI 중 2차가 더 낮아야 함
            _rsi1, _rsi2 = _sr[_i1], _sr[_i2]
            if np.isfinite(_rsi1) and np.isfinite(_rsi2) and _rsi2 < _rsi1:
                _dt_sig[_di] = 1.0
        df["double_top_signal"] = _dt_sig
    except Exception:
        df["double_top_signal"] = 0.0

    # ── 변동성 돌파 k=0.5 신호 (Larry Williams) — ML 피처화 ──────────────
    try:
        _prev_h  = high.shift(1)
        _prev_l  = low.shift(1)
        _vb_thr  = open_ + (_prev_h - _prev_l) * 0.5
        df["vb_signal"] = (close > _vb_thr).astype(float)
    except Exception:
        df["vb_signal"] = 0.0

    # ── 추세선 피처 — 스윙포인트 기반 동적 지지/저항 추세선 ─────────────
    # 최적화: 전체 피벗 포인트 1회 계산 후 슬라이딩 윈도우로 추세선 투영 (O(N × order))
    try:
        _tl_order = 5    # 좌우 5봉 기준 피벗
        _tl_win   = 60   # 최근 60봉 피벗만 사용
        _lv_tl    = low.values
        _hv_tl    = high.values
        _cv_tl    = close.values
        _n_tl     = len(df)

        # 전체 배열에서 피벗 저점/고점 한 번에 탐지
        _piv_low_mask  = np.zeros(_n_tl, dtype=bool)
        _piv_high_mask = np.zeros(_n_tl, dtype=bool)
        for _pi in range(_tl_order, _n_tl - _tl_order):
            _win_l = _lv_tl[_pi - _tl_order: _pi + _tl_order + 1]
            _win_h = _hv_tl[_pi - _tl_order: _pi + _tl_order + 1]
            if _lv_tl[_pi] == _win_l.min():
                _piv_low_mask[_pi] = True
            if _hv_tl[_pi] == _win_h.max():
                _piv_high_mask[_pi] = True

        _piv_low_idx  = np.where(_piv_low_mask)[0]
        _piv_high_idx = np.where(_piv_high_mask)[0]

        _tl_sd = np.zeros(_n_tl)
        _tl_rd = np.zeros(_n_tl)

        for _ti in range(_tl_win, _n_tl):
            _cutoff = _ti - _tl_win
            # 현재봉 기준 최근 60봉 내 피벗만 사용
            _sl_idx = _piv_low_idx[(_piv_low_idx >= _cutoff) & (_piv_low_idx < _ti)]
            _sh_idx = _piv_high_idx[(_piv_high_idx >= _cutoff) & (_piv_high_idx < _ti)]
            if len(_sl_idx) >= 2:
                # 피벗 저점들의 절대 인덱스로 선형 회귀 → 현재봉(_ti) 투영
                _coef_s = np.polyfit(_sl_idx.astype(float), _lv_tl[_sl_idx], 1)
                _tl_val = _coef_s[0] * _ti + _coef_s[1]
                if _tl_val > 0:
                    _tl_sd[_ti] = (_cv_tl[_ti] - _tl_val) / (_tl_val + 1e-9)
            if len(_sh_idx) >= 2:
                _coef_r = np.polyfit(_sh_idx.astype(float), _hv_tl[_sh_idx], 1)
                _tl_val = _coef_r[0] * _ti + _coef_r[1]
                if _tl_val > 0:
                    _tl_rd[_ti] = (_cv_tl[_ti] - _tl_val) / (_tl_val + 1e-9)

        _tl_sd_s = pd.Series(_tl_sd, index=df.index).clip(-0.1, 0.1)
        _tl_rd_s = pd.Series(_tl_rd, index=df.index).clip(-0.1, 0.1)
        df["tl_support_dist"] = _tl_sd_s
        df["tl_resist_dist"]  = _tl_rd_s
        df["tl_near_bounce"]  = (
            ((_tl_sd_s.abs() < 0.015) & (_tl_sd_s.abs() > 0.0)) |
            ((_tl_rd_s.abs() < 0.015) & (_tl_rd_s.abs() > 0.0))
        ).astype(float)
    except Exception:
        df["tl_support_dist"] = 0.0
        df["tl_resist_dist"]  = 0.0
        df["tl_near_bounce"]  = 0.0

    _pre_drop = df[FEATURE_NAMES].copy()
    # 전체가 NaN인 컬럼(가격 변동 없는 종목의 bb_pband 등) → 0으로 대체 후 dropna
    for _c in FEATURE_NAMES:
        if _pre_drop[_c].isna().all():
            _pre_drop[_c] = 0.0
    _result = _pre_drop.dropna()
    # close 컬럼 보존 — RL 에이전트 시뮬레이션 및 predict()에서 참조
    _result = _result.copy()
    _result["close"] = df.loc[_result.index, "close"]
    return _result
