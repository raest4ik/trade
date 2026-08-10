from __future__ import annotations

import json
from typing import cast

import httpx
import pytest

from src.ai_events.application.use_cases import (
    AnalyzeAIEventCommand,
    build_model_request,
    cache_key,
)
from src.ai_events.domain.exceptions import (
    AIModelTransientError,
    OllamaInvalidStructuredOutputError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.ai_events.domain.prompt import output_schema
from src.ai_events.infrastructure.ollama_client import (
    OllamaEventModelClient,
    fetch_ollama_model_identity,
)


def _request(*, think: bool = False, provider: str = "ollama"):
    return build_model_request(
        AnalyzeAIEventCommand(
            provider=provider,
            raw_content="Revenue was 100",
            requested_model="qwen3:8b",
            reasoning_effort=None,
            max_output_tokens=1000,
            think=think,
            random_seed=0,
            context_length=4096,
        )
    )


def _valid_output() -> dict[str, object]:
    return {
        "events": [
            {
                "event_type": "FINANCIAL_RESULTS",
                "is_primary": True,
                "confidence": 0.9,
                "evidence_text": "Revenue was 100",
            }
        ],
        "financial_facts": [
            {
                "metric": "REVENUE",
                "metric_name": None,
                "normalized_value": "100",
                "unit": "MONEY",
                "currency": "RUB",
                "scale": "ONE",
                "fact_role": "ACTUAL",
                "period_type": "UNKNOWN",
                "period_year": None,
                "period_quarter": None,
                "comparison_type": "NONE",
                "change_direction": "UNKNOWN",
                "change_value": None,
                "change_unit": None,
                "evidence_text": "Revenue was 100",
                "confidence": 0.8,
            }
        ],
        "warnings": [],
    }


def _success_response() -> dict[str, object]:
    return {
        "model": "qwen3:8b",
        "created_at": "2026-08-10T00:00:00Z",
        "message": {"role": "assistant", "content": json.dumps(_valid_output())},
        "done": True,
        "total_duration": 2_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_count": 120,
        "prompt_eval_duration": 500_000_000,
        "eval_count": 80,
        "eval_duration": 1_400_000_000,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [False, True])
async def test_ollama_chat_request_and_structured_response(think: bool) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = cast("dict[str, object]", json.loads(request.content))
        assert payload["model"] == "qwen3:8b"
        assert payload["stream"] is False
        assert payload["think"] is think
        assert payload["format"] == output_schema()
        assert payload["options"] == {
            "num_ctx": 4096,
            "num_predict": 1000,
            "seed": 0,
            "temperature": 0,
        }
        return httpx.Response(200, json=_success_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        completion = await client.analyze(_request(think=think))

    assert completion.output.events[0].event_type.value == "FINANCIAL_RESULTS"
    assert completion.actual_model == "qwen3:8b"
    assert completion.input_tokens == 120
    assert completion.output_tokens == 80
    assert completion.total_tokens == 200
    assert completion.latency_ms == 2000
    assert completion.cloud_cost_usd == "0"
    assert completion.provider_metadata["eval_count"] == 80


@pytest.mark.asyncio
async def test_ollama_invalid_json_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not-json"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaInvalidStructuredOutputError):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_json_that_violates_pydantic_schema_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"events": []}'}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaInvalidStructuredOutputError):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_canonicalizes_unspecified_unit_for_null_change_value() -> None:
    response = _success_response()
    output = _valid_output()
    fact = cast("list[dict[str, object]]", output["financial_facts"])[0]
    fact["change_unit"] = "UNSPECIFIED"
    message = cast("dict[str, object]", response["message"])
    message["content"] = json.dumps(output)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        completion = await client.analyze(_request())

    assert completion.output.financial_facts[0].change_unit is None
    assert "canonicalized change_unit UNSPECIFIED to null" in completion.output.warnings[0]


@pytest.mark.asyncio
async def test_ollama_does_not_canonicalize_real_unit_without_change_value() -> None:
    response = _success_response()
    output = _valid_output()
    fact = cast("list[dict[str, object]]", output["financial_facts"])[0]
    fact["change_unit"] = "PERCENT"
    message = cast("dict[str, object]", response["message"])
    message["content"] = json.dumps(output)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaInvalidStructuredOutputError):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_canonicalizes_quarter_for_non_quarter_period() -> None:
    response = _success_response()
    output = _valid_output()
    fact = cast("list[dict[str, object]]", output["financial_facts"])[0]
    fact["period_type"] = "YEAR"
    fact["period_year"] = 2025
    fact["period_quarter"] = 1
    message = cast("dict[str, object]", response["message"])
    message["content"] = json.dumps(output)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        completion = await client.analyze(_request())

    assert completion.output.financial_facts[0].period_quarter is None
    assert "canonicalized period_quarter to null" in completion.output.warnings[0]


@pytest.mark.asyncio
async def test_ollama_still_rejects_quarter_period_without_quarter() -> None:
    response = _success_response()
    output = _valid_output()
    fact = cast("list[dict[str, object]]", output["financial_facts"])[0]
    fact["period_type"] = "QUARTER"
    fact["period_quarter"] = None
    message = cast("dict[str, object]", response["message"])
    message["content"] = json.dumps(output)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaInvalidStructuredOutputError):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_unavailable_is_classified() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaUnavailableError, match="localhost:11434"):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_missing_model_is_classified() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaModelNotFoundError, match="ollama pull qwen3:8b"):
            await client.analyze(_request())


@pytest.mark.asyncio
async def test_ollama_timeout_is_classified() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaEventModelClient(
            base_url="http://localhost:11434",
            timeout_seconds=1,
            http_client=http_client,
        )
        with pytest.raises(OllamaTimeoutError):
            await client.analyze(_request())


def test_provider_specific_cache_keys_do_not_collide() -> None:
    ollama = _request()
    openai = _request(provider="openai")
    assert cache_key(ollama) != cache_key(openai)


def test_invalid_ollama_structured_output_is_retryable() -> None:
    assert issubclass(OllamaInvalidStructuredOutputError, AIModelTransientError)


@pytest.mark.asyncio
async def test_ollama_model_identity_uses_registry_digest() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3.5:9b",
                            "digest": "digest-123",
                            "details": {
                                "parameter_size": "9.7B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        identity = await fetch_ollama_model_identity(
            base_url="http://localhost:11434",
            model_tag="qwen3.5:9b",
            timeout_seconds=1,
            http_client=http_client,
        )

    assert identity.model_tag == "qwen3.5:9b"
    assert identity.model_digest == "digest-123"
    assert identity.parameter_size == "9.7B"
    assert identity.quantization_level == "Q4_K_M"
    assert identity.ollama_version == "0.12.3"
