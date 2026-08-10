from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_APP_NAME = "trade-ai-news-mvp"
DEFAULT_ENVIRONMENT = "local"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DATABASE_URL = "postgresql+asyncpg://trade_ai:trade_ai@localhost:5432/trade_ai"
DEFAULT_MOEX_ISS_BASE_URL = "https://iss.moex.com/iss"
DEFAULT_MOEX_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_MOEX_HTTP_MAX_RETRIES = 3
DEFAULT_MOEX_HTTP_MAX_PAGES = 1000
DEFAULT_MOEX_HTTP_USER_AGENT = "trade-ai-news-mvp/0.1"
DEFAULT_AI_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_THINK = False
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_AI_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_AI_MAX_RETRIES = 3
DEFAULT_AI_MAX_CONCURRENCY = 2
DEFAULT_AI_MAX_OUTPUT_TOKENS = 4096
DEFAULT_AI_REASONING_EFFORT = "low"


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = DEFAULT_APP_NAME
    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = DEFAULT_LOG_LEVEL
    database_url: str = DEFAULT_DATABASE_URL
    moex_iss_base_url: str = DEFAULT_MOEX_ISS_BASE_URL
    moex_http_timeout_seconds: float = DEFAULT_MOEX_HTTP_TIMEOUT_SECONDS
    moex_http_max_retries: int = DEFAULT_MOEX_HTTP_MAX_RETRIES
    moex_http_max_pages: int = DEFAULT_MOEX_HTTP_MAX_PAGES
    moex_http_user_agent: str = DEFAULT_MOEX_HTTP_USER_AGENT
    ai_provider: str = DEFAULT_AI_PROVIDER
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_think: bool = DEFAULT_OLLAMA_THINK
    openai_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    ai_request_timeout_seconds: float = DEFAULT_AI_REQUEST_TIMEOUT_SECONDS
    ai_max_retries: int = DEFAULT_AI_MAX_RETRIES
    ai_max_concurrency: int = DEFAULT_AI_MAX_CONCURRENCY
    ai_max_output_tokens: int = DEFAULT_AI_MAX_OUTPUT_TOKENS
    ai_reasoning_effort: str | None = DEFAULT_AI_REASONING_EFFORT

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
        moex_iss_base_url=os.getenv("MOEX_ISS_BASE_URL", DEFAULT_MOEX_ISS_BASE_URL),
        moex_http_timeout_seconds=float(
            os.getenv("MOEX_HTTP_TIMEOUT_SECONDS", str(DEFAULT_MOEX_HTTP_TIMEOUT_SECONDS))
        ),
        moex_http_max_retries=int(
            os.getenv("MOEX_HTTP_MAX_RETRIES", str(DEFAULT_MOEX_HTTP_MAX_RETRIES))
        ),
        moex_http_max_pages=int(os.getenv("MOEX_HTTP_MAX_PAGES", str(DEFAULT_MOEX_HTTP_MAX_PAGES))),
        moex_http_user_agent=os.getenv("MOEX_HTTP_USER_AGENT", DEFAULT_MOEX_HTTP_USER_AGENT),
        ai_provider=os.getenv("AI_PROVIDER", DEFAULT_AI_PROVIDER),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_think=_environment_bool("OLLAMA_THINK", DEFAULT_OLLAMA_THINK),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        ai_request_timeout_seconds=float(
            os.getenv(
                "AI_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_AI_REQUEST_TIMEOUT_SECONDS),
            )
        ),
        ai_max_retries=int(os.getenv("AI_MAX_RETRIES", str(DEFAULT_AI_MAX_RETRIES))),
        ai_max_concurrency=int(os.getenv("AI_MAX_CONCURRENCY", str(DEFAULT_AI_MAX_CONCURRENCY))),
        ai_max_output_tokens=int(
            os.getenv("AI_MAX_OUTPUT_TOKENS", str(DEFAULT_AI_MAX_OUTPUT_TOKENS))
        ),
        ai_reasoning_effort=os.getenv("AI_REASONING_EFFORT", DEFAULT_AI_REASONING_EFFORT) or None,
    )


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
