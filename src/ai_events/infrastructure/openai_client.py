from __future__ import annotations

import time
from typing import cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning

from src.ai_events.application.ports import AIModelCompletion, AIModelRequest
from src.ai_events.domain.exceptions import AIModelError, AIModelTransientError
from src.ai_events.domain.schema import AIEventOutput


class OpenAIEventModelClient:
    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    async def analyze(self, request: AIModelRequest) -> AIModelCompletion:
        started = time.perf_counter()
        reasoning: Reasoning | None = None
        if request.reasoning_effort is not None:
            reasoning = {"effort": cast("ReasoningEffort", request.reasoning_effort)}
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
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            raise AIModelTransientError("temporary OpenAI API failure") from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise AIModelTransientError("temporary OpenAI API failure") from exc
            raise AIModelError("OpenAI API rejected the request") from exc
        except OpenAIError as exc:
            raise AIModelError("OpenAI API request failed") from exc
        parsed = response.output_parsed
        if parsed is None:
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
        )
