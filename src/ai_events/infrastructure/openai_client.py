from __future__ import annotations

import importlib
import time
from typing import Protocol, cast

from src.ai_events.application.ports import AIModelCompletion, AIModelRequest
from src.ai_events.domain.exceptions import (
    AIConfigurationError,
    AIModelError,
    AIModelTransientError,
)
from src.ai_events.domain.schema import AIEventOutput


class _OpenAIUsage(Protocol):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class _OpenAIResponse(Protocol):
    id: str
    model: str
    output_parsed: object | None
    usage: _OpenAIUsage | None


class _OpenAIResponses(Protocol):
    async def parse(self, **kwargs: object) -> _OpenAIResponse: ...


class _OpenAIClient(Protocol):
    responses: _OpenAIResponses


class _AsyncOpenAIFactory(Protocol):
    def __call__(self, *, api_key: str, max_retries: int) -> _OpenAIClient: ...


class OpenAIEventModelClient:
    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        try:
            module = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            raise AIConfigurationError(
                "OpenAI backend requires the optional dependency; run: uv sync --extra openai"
            ) from exc
        factory = cast("_AsyncOpenAIFactory", module.__dict__["AsyncOpenAI"])
        self._client = factory(api_key=api_key, max_retries=0)

    async def analyze(self, request: AIModelRequest) -> AIModelCompletion:
        started = time.perf_counter()
        reasoning: dict[str, str] | None = None
        if request.reasoning_effort is not None:
            reasoning = {"effort": request.reasoning_effort}
        try:
            response = await self._client.responses.parse(
                model=request.requested_model,
                instructions=request.instructions,
                input=request.raw_content,
                text_format=AIEventOutput,
                max_output_tokens=request.max_output_tokens,
                reasoning=reasoning,
                store=False,
            )
        except Exception as exc:
            error_name = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if error_name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}:
                raise AIModelTransientError("temporary OpenAI API failure") from exc
            if isinstance(status_code, int) and status_code >= 500:
                raise AIModelTransientError("temporary OpenAI API failure") from exc
            if isinstance(status_code, int):
                raise AIModelError("OpenAI API rejected the request") from exc
            raise AIModelError("OpenAI API request failed") from exc
        parsed = response.output_parsed
        if not isinstance(parsed, AIEventOutput):
            raise AIModelError("OpenAI response did not contain structured output")
        usage = response.usage
        return AIModelCompletion(
            output=parsed,
            response_id=response.id,
            actual_model=response.model,
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            total_tokens=None if usage is None else usage.total_tokens,
            provider_metadata={},
            cloud_cost_usd=None,
        )
