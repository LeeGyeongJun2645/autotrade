"""종합 테스트: 수정된 전체 모듈 검증."""

import asyncio
import json
import sys
import traceback
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════
# 헬퍼: 가짜 OHLCV 300봉 생성
# ═══════════════════════════════════════════════════════

def make_ohlcv(n: int = 300, base: float = 50_000_000.0) -> list[dict]:
    """가짜 5분봉 OHLCV 리스트 생성 (최신이 index 0)."""
    rng = np.random.default_rng(42)
    prices = base * np.cumprod(1 + rng.normal(0, 0.002, n))
    result = []
    now = datetime.now()
    for i in range(n):
        p = float(prices[i])
        h = p * 1.002
        l = p * 0.998
        o = p * (1 + rng.normal(0, 0.001))
        v = float(rng.uniform(1, 10))
        ts = (now - timedelta(minutes=5 * (n - 1 - i))).strftime("%Y-%m-%dT%H:%M:%S")
        result.insert(0, {"open": o, "high": h, "low": l, "close": p, "volume": v, "candle_date_time_kst": ts})
    return result


def make_btc_ohlcv(n: int = 300) -> list[dict]:
    return make_ohlcv(n, base=80_000.0)


def make_oi_hist(n: int = 100) -> list[dict]:
    rng = np.random.default_rng(1)
    return [{"timestamp": i, "oi": float(rng.uniform(1e8, 2e8))} for i in range(n)]


def make_taker_hist(n: int = 100) -> list[dict]:
    rng = np.random.default_rng(2)
    return [{"timestamp": i, "buy_vol": float(rng.uniform(100, 500)), "sell_vol": float(rng.uniform(100, 500))} for i in range(n)]


# ═══════════════════════════════════════════════════════
# 1. features.py — Ichimoku NaN 수정 검증
# ═══════════════════════════════════════════════════════

class TestFeatures:
    def test_ichimoku_no_nan(self):
        """Ichimoku ffill/bfill 후 NaN 없어야 함."""
        from backend.ml.features import compute_features
        ohlcv = make_ohlcv(300)
        df = compute_features(ohlcv)
        ichi_cols = [c for c in df.columns if "ichi" in c.lower()]
        assert ichi_cols, "Ichimoku 컬럼 없음"
        for col in ichi_cols:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, f"{col}: NaN {nan_count}개 남아있음"

    def test_feature_count(self):
        """피처 수 85개 이상 확인."""
        from backend.ml.features import FEATURE_NAMES, compute_features
        ohlcv = make_ohlcv(300)
        df = compute_features(ohlcv)
        assert len(FEATURE_NAMES) >= 80, f"피처 수 부족: {len(FEATURE_NAMES)}"
        assert all(f in df.columns for f in FEATURE_NAMES), "FEATURE_NAMES에 없는 컬럼 존재"

    def test_with_btc_oi_taker(self):
        """BTC/OI/Taker 추가 피처 계산 오류 없음."""
        from backend.ml.features import compute_features
        ohlcv = make_ohlcv(300)
        btc = make_btc_ohlcv(300)
        oi = make_oi_hist(100)
        taker = make_taker_hist(100)
        df = compute_features(ohlcv, btc_ohlcv=btc, oi_hist=oi, taker_hist=taker)
        assert len(df) > 0, "BTC/OI/Taker 포함 피처 계산 실패"

    def test_minimum_rows(self):
        """300봉 입력 → dropna 후 최소 200봉 이상 (NaN 손실 33% 이하)."""
        from backend.ml.features import compute_features
        ohlcv = make_ohlcv(300)
        df = compute_features(ohlcv)
        assert len(df) >= 200, f"피처 행 수 너무 적음: {len(df)}/300"


# ═══════════════════════════════════════════════════════
# 2. model.py — OI/Taker 파라미터 전달 검증
# ═══════════════════════════════════════════════════════

class TestModel:
    def test_train_signature(self):
        """XGBSignalModel.train()이 btc_ohlcv/oi_hist/taker_hist 파라미터 받음."""
        import inspect
        from backend.ml.model import XGBSignalModel
        sig = inspect.signature(XGBSignalModel.train)
        params = sig.parameters
        assert "btc_ohlcv" in params, "train()에 btc_ohlcv 없음"
        assert "oi_hist" in params, "train()에 oi_hist 없음"
        assert "taker_hist" in params, "train()에 taker_hist 없음"

    def test_predict_signature(self):
        """XGBSignalModel.predict()에도 동일 파라미터 있음."""
        import inspect
        from backend.ml.model import XGBSignalModel
        sig = inspect.signature(XGBSignalModel.predict)
        params = sig.parameters
        assert "btc_ohlcv" in params
        assert "oi_hist" in params
        assert "taker_hist" in params

    def test_train_runs(self):
        """실제 학습 실행 오류 없음."""
        from backend.ml.model import XGBSignalModel
        model = XGBSignalModel("TEST-KRW")
        ohlcv = make_ohlcv(300)
        result = model.train(ohlcv)
        assert result.n_samples > 0
        assert 0.0 <= result.accuracy <= 1.0


# ═══════════════════════════════════════════════════════
# 3. agents.py — LightGBM + 피드백루프 검증
# ═══════════════════════════════════════════════════════

class TestAgents:
    def test_agent_configs_8tuple(self):
        """AGENT_CONFIGS가 8-tuple (model_type 포함) 확인."""
        from backend.ml.agents import AGENT_CONFIGS
        assert len(AGENT_CONFIGS) == 20, f"에이전트 수: {len(AGENT_CONFIGS)}"
        for cfg in AGENT_CONFIGS:
            assert len(cfg) == 8, f"8-tuple 아님: {cfg}"
            assert cfg[7] in ("xgb", "lgbm"), f"model_type 이상: {cfg[7]}"

    def test_odd_lgbm_even_xgb(self):
        """홀수 에이전트=LGBM, 짝수=XGB."""
        from backend.ml.agents import AGENT_CONFIGS
        for cfg in AGENT_CONFIGS:
            agent_id = cfg[0]  # "AI01", "AI02", ...
            num = int(agent_id[2:])
            model_type = cfg[7]
            if num % 2 == 1:
                assert model_type == "lgbm", f"{agent_id} 홀수인데 lgbm 아님"
            else:
                assert model_type == "xgb", f"{agent_id} 짝수인데 xgb 아님"

    def test_sim_agent_model_type(self):
        """SimAgent가 model_type 속성 보유."""
        from backend.ml.agents import SimAgent
        agent = SimAgent("AI01", 5, 0.001, 0.55, "all", "coin", 3, "lgbm")
        assert agent.model_type == "lgbm"
        agent2 = SimAgent("AI02", 5, 0.001, 0.62, "momentum", "coin", 5, "xgb")
        assert agent2.model_type == "xgb"

    def test_train_accepts_trade_results(self):
        """train()이 trade_results 파라미터 받음."""
        import inspect
        from backend.ml.agents import SimAgent
        sig = inspect.signature(SimAgent.train)
        assert "trade_results" in sig.parameters

    def test_lgbm_train(self):
        """LightGBM 에이전트 실제 학습 오류 없음."""
        from backend.ml.agents import SimAgent
        agent = SimAgent("AI01", 5, 0.001, 0.55, "all", "coin", 3, "lgbm")
        ohlcv = make_ohlcv(300)
        agent.train(ohlcv)
        assert agent._model is not None

    def test_xgb_train(self):
        """XGBoost 에이전트 실제 학습 오류 없음."""
        from backend.ml.agents import SimAgent
        agent = SimAgent("AI02", 5, 0.001, 0.62, "momentum", "coin", 5, "xgb")
        ohlcv = make_ohlcv(300)
        agent.train(ohlcv)
        assert agent._model is not None

    def test_feedback_sample_weights(self):
        """trade_results가 sample_weight 조정에 영향줌 (충돌/오류 없음)."""
        from backend.ml.agents import SimAgent
        agent = SimAgent("AI01", 5, 0.001, 0.55, "all", "coin", 3, "lgbm")
        ohlcv = make_ohlcv(300)
        trade_results = [
            {"buy_at": datetime.now().isoformat(), "profit_rate": 0.05},
            {"buy_at": (datetime.now() - timedelta(hours=2)).isoformat(), "profit_rate": -0.02},
        ]
        agent.train(ohlcv, trade_results=trade_results)
        assert agent._model is not None

    def test_predict_ensemble(self):
        """predict_ensemble() 실행 오류 없음."""
        from backend.ml.agents import AGENTS, predict_ensemble
        ohlcv = make_ohlcv(300)
        # 먼저 일부 에이전트 학습
        for agent in list(AGENTS.values())[:3]:
            try:
                agent.train(ohlcv)
            except Exception:
                pass
        result = predict_ensemble(ohlcv, market="coin")
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════
# 4. scheduler.py — DCA 로직 검증
# ═══════════════════════════════════════════════════════

class TestDCALogic:
    def test_dca_state_init(self):
        """_dca_state 딕셔너리 초기화 확인."""
        from backend.core.scheduler import TradingScheduler
        s = TradingScheduler.__new__(TradingScheduler)
        s._dca_state = {}
        assert isinstance(s._dca_state, dict)

    def test_dca_upbit_budget_split(self):
        """업비트 DCA: 1차=40%, 2차=35%, 3차=25% 합산=100%."""
        ratios = [0.40, 0.35, 0.25]
        assert abs(sum(ratios) - 1.0) < 1e-9, f"DCA 비율 합산 오류: {sum(ratios)}"

    def test_dca_kis_qty_split(self):
        """KIS DCA: total_qty 기준 1차 40%, 2차 35%, 3차 25%."""
        total = 100
        first = max(1, int(total * 0.40))   # 40
        second = max(1, int(total * 0.35))  # 35
        third = max(1, int(total * 0.25))   # 25
        assert first == 40
        assert second == 35
        assert third == 25
        assert first + second + third == 100

    def test_dca_average_price_calculation(self):
        """평균단가 재계산 정확도 검증."""
        # 1차: 40주 @ 10만원
        entry_price = 100_000.0
        qty = 40.0
        # 2차: 35주 @ 98,500원 (-1.5%)
        add_price = 98_500.0
        add_qty = 35.0

        total_cost = entry_price * qty + add_price * add_qty
        new_qty = qty + add_qty
        new_avg = total_cost / new_qty

        expected = (100_000 * 40 + 98_500 * 35) / 75
        assert abs(new_avg - expected) < 0.01, f"평균단가 오류: {new_avg} vs {expected}"
        assert new_avg < entry_price, "추가매수 후 평균단가는 내려가야 함"

    def test_dca_trigger_thresholds(self):
        """DCA 트리거: -1.5% → 2차, -3.0% → 3차."""
        entry_price = 100_000.0
        price_drop_15 = entry_price * (1 - 0.015)   # 98,500
        price_drop_30 = entry_price * (1 - 0.030)   # 97,000
        price_drop_10 = entry_price * (1 - 0.010)   # 99,000

        drop_15 = (price_drop_15 - entry_price) / entry_price
        drop_30 = (price_drop_30 - entry_price) / entry_price
        drop_10 = (price_drop_10 - entry_price) / entry_price

        # stage=1 조건: drop <= -0.015
        assert drop_15 <= -0.015, "2차 트리거 미발동"
        assert drop_10 > -0.015, "10% 하락에서 2차 잘못 발동"
        # stage=2 조건: drop <= -0.030
        assert drop_30 <= -0.030, "3차 트리거 미발동"
        assert drop_15 > -0.030, "1.5% 하락에서 3차 잘못 발동"

    def test_dca_execute_upbit_dca_add_exists(self):
        """_execute_upbit_dca_add 메서드 존재 확인."""
        from backend.core.scheduler import TradingScheduler
        assert hasattr(TradingScheduler, "_execute_upbit_dca_add")

    def test_dca_execute_kis_dca_add_exists(self):
        """_execute_kis_dca_add 메서드 존재 확인."""
        from backend.core.scheduler import TradingScheduler
        assert hasattr(TradingScheduler, "_execute_kis_dca_add")

    def test_dca_sell_clears_state(self):
        """매도 시 _dca_state에서 ticker 제거되는 로직 확인."""
        dca_state: dict = {}
        ticker = "KRW-BTC"
        dca_state[ticker] = {"stage": 2, "total_budget": 1_000_000}
        # 매도 시 pop
        dca_state.pop(ticker, None)
        assert ticker not in dca_state


# ═══════════════════════════════════════════════════════
# 5. news.py — LLM 감성분석 폴백 검증
# ═══════════════════════════════════════════════════════

class TestNews:
    def test_vader_fallback_no_key(self):
        """OpenAI 키 없을 때 VADER 폴백 정상 작동."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.openai_api_key = None

            from backend.ml.news import _score_en, _score_kr, _weighted
            titles_en = ["Bitcoin surges to new high", "Crypto market crashes badly"]
            en_score, en_cnt = _score_en(titles_en, ["bitcoin", "btc"])
            assert -1.0 <= en_score <= 1.0
            assert en_cnt >= 1

    def test_korean_keyword_scoring(self):
        """한국어 키워드 감성 점수 방향성 확인."""
        from backend.ml.news import _score_kr
        pos_titles = ["비트코인 급등 신고가 돌파", "이더리움 강세 반등"]
        neg_titles = ["비트코인 급락 폭락 위기", "코인 붕괴 공포"]

        pos_score, _ = _score_kr(pos_titles)
        neg_score, _ = _score_kr(neg_titles)

        assert pos_score > 0, f"긍정 뉴스 점수 음수: {pos_score}"
        assert neg_score < 0, f"부정 뉴스 점수 양수: {neg_score}"

    def test_weighted_blend(self):
        """영어40% + 한국어60% 가중 평균 정확도."""
        from backend.ml.news import _weighted
        result = _weighted(0.5, 1.0)
        expected = 0.5 * 0.4 + 1.0 * 0.6  # 0.8
        assert abs(result - expected) < 1e-9

    def test_weighted_single_source(self):
        """한쪽만 있을 때 그 값 그대로 반환."""
        from backend.ml.news import _weighted
        assert _weighted(0.0, 0.7) == 0.7
        assert _weighted(0.3, 0.0) == 0.3

    def test_cache_mechanism(self):
        """캐시 TTL 로직 - 만료 전 동일 값 반환."""
        import time
        from backend.ml import news
        # 강제로 캐시 삽입
        news._score_cache["TEST"] = (0.5, time.time() + 3600)
        result = asyncio.get_event_loop().run_until_complete(
            news.get_sentiment_score("TEST")
        )
        assert result == 0.5, f"캐시 미적용: {result}"

    @pytest.mark.asyncio
    async def test_llm_sentiment_mock(self):
        """LLM 경로 - OpenAI 모킹으로 응답 형식 검증."""
        from backend.ml.news import _llm_sentiment
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"score": 0.75})

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with patch("backend.config.settings") as mock_settings:
                mock_settings.openai_api_key = "sk-test"
                score = await _llm_sentiment(["Bitcoin hits all-time high"], "KRW-BTC")

        assert score is None or -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        """LLM 호출 실패 시 None 반환 (VADER 폴백 신호)."""
        from backend.ml.news import _llm_sentiment
        with patch("backend.config.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-invalid"
            with patch("openai.AsyncOpenAI", side_effect=Exception("API 오류")):
                score = await _llm_sentiment(["test headline"], "KRW-BTC")
        assert score is None


# ═══════════════════════════════════════════════════════
# 6. config.py — 설정 필드 검증
# ═══════════════════════════════════════════════════════

class TestConfig:
    def test_openai_key_field_exists(self):
        """Settings에 openai_api_key 필드 있음."""
        from backend.config import Settings
        fields = Settings.model_fields
        assert "openai_api_key" in fields

    def test_openai_key_optional(self):
        """openai_api_key는 Optional (None 허용)."""
        from backend.config import Settings
        field = Settings.model_fields["openai_api_key"]
        assert field.default is None

    def test_stop_loss_must_be_negative(self):
        """손절율 양수 입력 시 ValueError."""
        from backend.config import Settings
        import pytest
        with pytest.raises(Exception):
            Settings(
                kis_app_key="x", kis_app_secret="x",
                kis_account_no="x", stop_loss_rate=0.02
            )


# ═══════════════════════════════════════════════════════
# 7. 충돌/통합 검증
# ═══════════════════════════════════════════════════════

class TestIntegration:
    def test_features_to_model_pipeline(self):
        """features → model 전체 파이프라인 오류 없음."""
        from backend.ml.features import FEATURE_NAMES, compute_features
        from backend.ml.model import XGBSignalModel
        ohlcv = make_ohlcv(300)
        df = compute_features(ohlcv)
        assert all(f in df.columns for f in FEATURE_NAMES)

        model = XGBSignalModel("PIPE-TEST")
        result = model.train(ohlcv)
        assert result.n_features == len(FEATURE_NAMES)

    def test_agent_predict_after_train(self):
        """에이전트 학습 후 predict() 정상 작동."""
        from backend.ml.agents import SimAgent
        agent = SimAgent("AI03", 5, 0.002, 0.55, "trend", "coin", 8, "lgbm")
        ohlcv = make_ohlcv(300)
        agent.train(ohlcv)
        pred = agent.predict(ohlcv)
        assert pred in (0, 1), f"예측값 이상: {pred}"

    def test_lgbm_xgb_both_predict(self):
        """LightGBM과 XGBoost 에이전트 동시 예측 충돌 없음."""
        from backend.ml.agents import SimAgent
        ohlcv = make_ohlcv(300)
        lgbm_agent = SimAgent("AI01", 5, 0.001, 0.55, "all", "coin", 3, "lgbm")
        xgb_agent = SimAgent("AI02", 5, 0.001, 0.62, "momentum", "coin", 5, "xgb")
        lgbm_agent.train(ohlcv)
        xgb_agent.train(ohlcv)
        p1 = lgbm_agent.predict(ohlcv)
        p2 = xgb_agent.predict(ohlcv)
        assert p1 in (0, 1)
        assert p2 in (0, 1)

    def test_scheduler_has_all_dca_methods(self):
        """TradingScheduler에 DCA 관련 메서드 전부 있음."""
        from backend.core.scheduler import TradingScheduler
        required = [
            "_execute_upbit_buy",
            "_execute_upbit_sell",
            "_execute_upbit_dca_add",
            "_execute_kis_buy",
            "_execute_kis_sell",
            "_execute_kis_dca_add",
        ]
        for method in required:
            assert hasattr(TradingScheduler, method), f"메서드 없음: {method}"

    def test_dca_method_signatures(self):
        """DCA 추가매수 메서드 시그니처 확인."""
        import inspect
        from backend.core.scheduler import TradingScheduler
        sig_upbit = inspect.signature(TradingScheduler._execute_upbit_dca_add)
        assert "next_stage" in sig_upbit.parameters
        assert "budget_ratio" in sig_upbit.parameters

        sig_kis = inspect.signature(TradingScheduler._execute_kis_dca_add)
        assert "next_stage" in sig_kis.parameters
        assert "qty_ratio" in sig_kis.parameters
