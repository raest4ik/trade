from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ai_events.domain.schema import AIEventOutput


@dataclass(frozen=True, slots=True)
class AIModelRequest:
    provider: str
    raw_content: str
    requested_model: str
    instructions: str
    prompt_version: str
    prompt_hash: str
    schema_version: str
    schema_hash: str
    analyzer_version: str
    reasoning_effort: str | None
    max_output_tokens: int
    think: bool


@dataclass(frozen=True, slots=True)
class AIModelCompletion:
    output: AIEventOutput
    response_id: str
    actual_model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    provider_metadata: dict[str, int | str | bool | None]
    cloud_cost_usd: str | None


class AIEventModelClient(Protocol):
    async def analyze(self, request: AIModelRequest) -> AIModelCompletion: ...


class AIEventCache(Protocol):
    async def get(self, key: str) -> AIModelCompletion | None: ...

    async def put(self, key: str, completion: AIModelCompletion) -> None: ...
