from __future__ import annotations

import json
import time
from typing import cast

import httpx
from pydantic import ValidationError

from src.ai_events.application.ports import AIModelCompletion, AIModelRequest
from src.ai_events.domain.exceptions import (
    AIModelError,
    OllamaInvalidStructuredOutputError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.ai_events.domain.prompt import output_schema
from src.ai_events.domain.schema import AIEventOutput

_USAGE_FIELDS = (
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)


class OllamaEventModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._http_client = http_client

    async def analyze(self, request: AIModelRequest) -> AIModelCompletion:
        started = time.perf_counter()
        payload: dict[str, object] = {
            "model": request.requested_model,
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.raw_content},
            ],
            "stream": False,
            "think": request.think,
            "format": output_schema(),
            "options": {
                "num_predict": request.max_output_tokens,
                "seed": 0,
                "temperature": 0,
            },
        }
        try:
            response = await self._post(payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError() from exc
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(self._base_url) from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(request.requested_model)
        if response.status_code >= 500:
            raise OllamaUnavailableError(self._base_url)
        if response.status_code >= 400:
            raise AIModelError(f"Ollama rejected the request with status {response.status_code}")

        try:
            body = cast("dict[str, object]", response.json())
            message = cast("dict[str, object]", body["message"])
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("message.content must be a string")
            structured = json.loads(content)
            _canonicalize_null_change_units(structured)
            parsed = AIEventOutput.model_validate(structured)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise OllamaInvalidStructuredOutputError() from exc

        provider_metadata = _provider_metadata(body)
        input_tokens = _optional_int(body.get("prompt_eval_count"))
        output_tokens = _optional_int(body.get("eval_count"))
        token_counts = [value for value in (input_tokens, output_tokens) if value is not None]
        total_tokens = sum(token_counts) if token_counts else None
        total_duration = _optional_int(body.get("total_duration"))
        measured_latency = round((time.perf_counter() - started) * 1000)
        return AIModelCompletion(
            output=parsed,
            response_id=f"ollama:{body.get('created_at', 'local')}",
            actual_model=str(body.get("model") or request.requested_model),
            latency_ms=measured_latency
            if total_duration is None
            else round(total_duration / 1_000_000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider_metadata=provider_metadata,
            cloud_cost_usd="0",
        )

    async def _post(self, payload: dict[str, object]) -> httpx.Response:
        url = f"{self._base_url}/api/chat"
        if self._http_client is not None:
            return await self._http_client.post(url, json=payload, timeout=self._timeout)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, json=payload)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _provider_metadata(body: dict[str, object]) -> dict[str, int | str | bool | None]:
    return {
        field: value
        for field in _USAGE_FIELDS
        if (value := _optional_int(body.get(field))) is not None
    }


def _canonicalize_null_change_units(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    structured = cast("dict[object, object]", payload)
    facts = structured.get("financial_facts")
    warnings = structured.get("warnings")
    if not isinstance(facts, list) or not isinstance(warnings, list):
        return
    fact_values = cast("list[object]", facts)
    warning_values = cast("list[object]", warnings)
    canonicalized = 0
    for value in fact_values:
        if not isinstance(value, dict):
            continue
        fact = cast("dict[object, object]", value)
        if fact.get("change_value") is None and fact.get("change_unit") == "UNSPECIFIED":
            fact["change_unit"] = None
            canonicalized += 1
    if canonicalized:
        warning_values.append(
            "canonicalized change_unit UNSPECIFIED to null because change_value is null "
            f"for {canonicalized} fact(s)"
        )
