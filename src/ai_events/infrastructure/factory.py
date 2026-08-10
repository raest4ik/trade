from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.ai_events.application.reliability import ReliableAIEventModelClient
from src.ai_events.application.use_cases import AnalyzeAIEvent
from src.ai_events.domain.enums import AIProvider
from src.ai_events.domain.exceptions import AIConfigurationError
from src.ai_events.infrastructure.cache import JsonFileAIEventCache
from src.ai_events.infrastructure.ollama_client import OllamaEventModelClient
from src.shared.config.settings import Settings

DEFAULT_AI_CACHE_DIRECTORY = Path("artifacts/ai-event-v0/cache")


@dataclass(frozen=True, slots=True)
class AIProviderConfig:
    provider: AIProvider
    requested_model: str
    reasoning_effort: str | None
    think: bool
    artifact_slug: str


def resolve_ai_provider_config(
    settings: Settings,
    provider_override: str | None = None,
) -> AIProviderConfig:
    value = provider_override or settings.ai_provider
    try:
        provider = AIProvider(value.lower())
    except ValueError as exc:
        raise AIConfigurationError(f"unsupported AI provider: {value}") from exc
    if provider == AIProvider.OLLAMA:
        model = settings.ollama_model
        return AIProviderConfig(
            provider=provider,
            requested_model=model,
            reasoning_effort=None,
            think=settings.ollama_think,
            artifact_slug=_artifact_slug(provider, model),
        )
    if settings.openai_api_key is None:
        raise AIConfigurationError("OPENAI_API_KEY is not configured")
    return AIProviderConfig(
        provider=provider,
        requested_model=settings.openai_model,
        reasoning_effort=settings.ai_reasoning_effort,
        think=False,
        artifact_slug=_artifact_slug(provider, settings.openai_model),
    )


def create_ai_event_analyzer(
    settings: Settings,
    *,
    cache_directory: Path = DEFAULT_AI_CACHE_DIRECTORY,
    provider_override: str | None = None,
) -> AnalyzeAIEvent:
    provider_config = resolve_ai_provider_config(settings, provider_override)
    if provider_config.provider == AIProvider.OLLAMA:
        client = OllamaEventModelClient(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ai_request_timeout_seconds,
        )
    else:
        from src.ai_events.infrastructure.openai_client import OpenAIEventModelClient

        assert settings.openai_api_key is not None
        client = OpenAIEventModelClient(api_key=settings.openai_api_key)
    reliable = ReliableAIEventModelClient(
        client,
        timeout_seconds=settings.ai_request_timeout_seconds,
        max_retries=settings.ai_max_retries,
        max_concurrency=settings.ai_max_concurrency,
    )
    return AnalyzeAIEvent(reliable, JsonFileAIEventCache(cache_directory))


def _artifact_slug(provider: AIProvider, model: str) -> str:
    normalized_model = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"{provider.value}-{normalized_model}"
