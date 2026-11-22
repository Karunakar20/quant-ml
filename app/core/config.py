
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Quant ML"
    APP_ENV: str = "dev"
    DEBUG: bool = True

    DATABASE_URL: str

    DATA_RESOLUTION: str = "D"
    HISTORY_START_DATE: str = "2021-01-01"
    HISTORY_END_DATE: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
