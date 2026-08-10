from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from src.ai_events.application.ports import (
    AIEventModelClient,
    AIModelCompletion,
    AIModelRequest,
)
from src.ai_events.domain.exceptions import AIModelError, AIModelTransientError


class ReliableAIEventModelClient:
    def __init__(
        self,
        client: AIEventModelClient,
        *,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep

    async def complete(self, request: AIModelRequest) -> AIModelCompletion:
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    async with asyncio.timeout(self._timeout_seconds):
                        return await self._client.complete(request)
                except TimeoutError as exc:
                    transient: AIModelTransientError = AIModelTransientError("AI request timed out")
                    transient.__cause__ = exc
                except AIModelTransientError as exc:
                    transient = exc
                except AIModelError:
                    raise
                except Exception as exc:
                    raise AIModelError("AI model request failed") from exc
                if attempt >= self._max_retries:
                    raise AIModelError("AI request failed after bounded retries") from transient
                await self._sleep(min(0.5 * (2**attempt), 8.0))
        raise AIModelError("AI request failed")
