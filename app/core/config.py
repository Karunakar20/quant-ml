from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "Quant ML"
    APP_ENV: str = "dev"
    DEBUG: bool = True

    DATABASE_URL: str

    DATA_RESOLUTION: str = "D"
    HISTORY_START_DATE: str = "2021-01-01"
    HISTORY_END_DATE: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()