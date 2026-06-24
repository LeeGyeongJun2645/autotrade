"""환경변수 로드 및 전역 설정 관리 (pydantic-settings 기반)."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── 실행 모드 ─────────────────────────────────────────────────
    trade_mode: Literal["PAPER", "LIVE"] = "PAPER"

    # ── KIS API ──────────────────────────────────────────────────
    kis_app_key: str
    kis_app_secret: str
    kis_account_no: str
    kis_account_prod_code: str = "01"
    kis_base_url_paper: str = "https://openapivts.koreainvestment.com:29443"
    kis_base_url_live: str = "https://openapi.koreainvestment.com:9443"

    # ── 업비트 API (미설정 시 업비트 기능 비활성화) ──────────────────
    upbit_access_key: Optional[str] = None
    upbit_secret_key: Optional[str] = None

    # ── 텔레그램 (미설정 시 알림 비활성화) ───────────────────────────
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # ── ML / 뉴스 감성 (미설정 시 뉴스 기능 비활성화) ────────────────
    cryptopanic_token: Optional[str] = None

    # ── 데이터베이스 ─────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./autotrade.db"

    # ── 리스크 관리 ──────────────────────────────────────────────
    stop_loss_rate: float = -0.02
    take_profit_rate: float = 0.04
    trailing_stop_rate: float = 0.01
    max_position_ratio: float = 0.20

    # ── 전략 파라미터 ────────────────────────────────────────────
    volatility_k: float = 0.5
    ma_short: int = 20
    ma_long: int = 60
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # ── Rate Limit ───────────────────────────────────────────────
    kis_rate_limit_per_sec: int = 18

    # ── FastAPI 서버 ─────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        """콤마 구분 문자열을 리스트로 변환."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @field_validator("stop_loss_rate")
    @classmethod
    def validate_stop_loss(cls, v: float) -> float:
        """손절은 반드시 음수여야 함."""
        if v >= 0:
            raise ValueError("stop_loss_rate 는 음수여야 합니다 (예: -0.02)")
        return v

    @property
    def kis_base_url(self) -> str:
        """TRADE_MODE에 따라 KIS 기본 URL 자동 반환."""
        return self.kis_base_url_live if self.trade_mode == "LIVE" else self.kis_base_url_paper

    @property
    def is_paper(self) -> bool:
        """모의투자 모드 여부."""
        return self.trade_mode == "PAPER"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤 반환 (최초 1회만 .env 파싱)."""
    return Settings()


settings = get_settings()
