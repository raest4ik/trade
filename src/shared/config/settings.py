from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_APP_NAME = "trade-ai-news-mvp"
DEFAULT_ENVIRONMENT = "local"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DATABASE_URL = "postgresql+asyncpg://trade_ai:trade_ai@localhost:5432/trade_ai"


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = DEFAULT_APP_NAME
    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = DEFAULT_LOG_LEVEL
    database_url: str = DEFAULT_DATABASE_URL

    @property
    def sync_database_url(self) -> str:
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
        if self.database_url.startswith("sqlite+aiosqlite://"):
            return self.database_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", DEFAULT_APP_NAME),
        environment=os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
