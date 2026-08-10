from __future__ import annotations

from pathlib import Path

from src.ai_events.application.reliability import ReliableAIEventModelClient
from src.ai_events.application.use_cases import AnalyzeAIEvent
from src.ai_events.domain.exceptions import AIConfigurationError
from src.ai_events.infrastructure.cache import JsonFileAIEventCache
from src.ai_events.infrastructure.openai_client import OpenAIResponsesEventModelClient
from src.shared.config.settings import Settings

DEFAULT_AI_CACHE_DIRECTORY = Path("artifacts/ai-event-v0/cache")


def create_ai_event_analyzer(
    settings: Settings,
    *,
    cache_directory: Path = DEFAULT_AI_CACHE_DIRECTORY,
) -> AnalyzeAIEvent:
    if settings.openai_api_key is None:
        raise AIConfigurationError("OPENAI_API_KEY is not configured")
    client = OpenAIResponsesEventModelClient(api_key=settings.openai_api_key)
    reliable = ReliableAIEventModelClient(
        client,
        timeout_seconds=settings.ai_request_timeout_seconds,
        max_retries=settings.ai_max_retries,
        max_concurrency=settings.ai_max_concurrency,
    )
    return AnalyzeAIEvent(reliable, JsonFileAIEventCache(cache_directory))
