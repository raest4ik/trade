from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from src.ai_events.application.evaluation import to_evaluation_inputs
from src.ai_events.application.frozen_test import FrozenTestGuardError, validate_test_access
from src.ai_events.application.ports import AIModelCompletion, AIModelRequest
from src.ai_events.application.reliability import ReliableAIEventModelClient
from src.ai_events.application.use_cases import (
    AnalyzeAIEvent,
    AnalyzeAIEventCommand,
    build_model_request,
    cache_key,
    sanitize_failure,
)
from src.ai_events.domain.evidence import resolve_exact_evidence
from src.ai_events.domain.exceptions import (
    AIConfigurationError,
    AIModelError,
    AIModelTransientError,
    AIOutputValidationError,
)
from src.ai_events.domain.prompt import SYSTEM_PROMPT, output_schema
from src.ai_events.domain.schema import AIEventOutput, parse_decimal_string
from src.ai_events.infrastructure.cache import JsonFileAIEventCache
from src.ai_events.infrastructure.factory import create_ai_event_analyzer
from src.evaluation.domain.entities import GoldEvent
from src.evaluation.domain.enums import DatasetSplit
from src.events.domain.enums import EventType
from src.shared.config.settings import Settings


class FakeModelClient:
    def __init__(self, output: AIEventOutput) -> None:
        self.output = output
        self.calls = 0

    async def complete(self, request: AIModelRequest) -> AIModelCompletion:
        self.calls += 1
        return AIModelCompletion(
            output=self.output,
            response_id="resp_fake",
            actual_model=request.requested_model,
            latency_ms=12,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
        )


class TransientFakeClient:
    def __init__(self, failures: int, output: AIEventOutput) -> None:
        self.failures = failures
        self.output = output
        self.calls = 0

    async def complete(self, request: AIModelRequest) -> AIModelCompletion:
        self.calls += 1
        if self.calls <= self.failures:
            raise AIModelTransientError("temporary")
        return AIModelCompletion(
            output=self.output,
            response_id="resp_retry",
            actual_model=request.requested_model,
            latency_ms=1,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )


def _event_payload(*, primary: bool = True) -> dict[str, object]:
    return {
        "event_type": "FINANCIAL_RESULTS",
        "is_primary": primary,
        "confidence": 0.95,
        "evidence_text": "Revenue was 100",
    }


def _fact_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "metric": "REVENUE",
        "metric_name": None,
        "normalized_value": "100",
        "unit": "MONEY",
        "currency": "RUB",
        "scale": "MILLION",
        "fact_role": "ACTUAL",
        "period_type": "YEAR",
        "year": 2025,
        "quarter": None,
        "comparison_type": "NONE",
        "change_direction": "UNKNOWN",
        "change_value": None,
        "change_unit": None,
        "evidence_text": "Revenue was 100",
        "confidence": 0.9,
    }
    payload.update(changes)
    return payload


def _output(*, events: list[dict[str, object]] | None = None) -> AIEventOutput:
    return AIEventOutput.model_validate(
        {
            "events": [_event_payload()] if events is None else events,
            "financial_facts": [_fact_payload()],
            "warnings": [],
        }
    )


def _command(text: str = "Revenue was 100") -> AnalyzeAIEventCommand:
    return AnalyzeAIEventCommand(
        raw_content=text,
        requested_model="gpt-5-mini",
        reasoning_effort="low",
        max_output_tokens=1000,
    )


def test_structured_output_schema_is_strict_and_all_fields_are_required() -> None:
    schema = output_schema()
    assert schema["additionalProperties"] is False
    assert set(cast("list[str]", schema["required"])) == {
        "events",
        "financial_facts",
        "warnings",
    }
    definitions = cast("dict[str, object]", schema["$defs"])
    for value in definitions.values():
        definition = cast("dict[str, object]", value)
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            required = cast("list[str]", definition["required"])
            properties = cast("dict[str, object]", definition["properties"])
            assert set(required) == set(properties)


def test_schema_rejects_extra_fields_and_invalid_enum() -> None:
    payload = {
        "events": [{**_event_payload(), "unexpected": True}],
        "financial_facts": [_fact_payload(metric="NOT_A_METRIC")],
        "warnings": [],
    }
    with pytest.raises(ValidationError):
        AIEventOutput.model_validate(payload)


def test_decimal_strings_are_lossless_and_reject_noncanonical_values() -> None:
    assert parse_decimal_string("1234567890.0010") == Decimal("1234567890.0010")
    with pytest.raises(ValueError):
        parse_decimal_string("1,25")
    with pytest.raises(ValidationError):
        AIEventOutput.model_validate(
            {
                "events": [],
                "financial_facts": [_fact_payload(normalized_value="1e3")],
                "warnings": [],
            }
        )


def test_evidence_offsets_are_exact_and_duplicate_is_warned() -> None:
    unique = resolve_exact_evidence("prefix Revenue was 100 suffix", "Revenue was 100")
    assert (unique.start, unique.end, unique.warning) == (7, 22, None)
    duplicate = resolve_exact_evidence("same and same", "same")
    assert (duplicate.start, duplicate.end) == (0, 4)
    assert duplicate.warning == "duplicate evidence occurrence; selected first of 2"
    with pytest.raises(AIOutputValidationError):
        resolve_exact_evidence("Revenue was 100", "revenue was 100")


def test_primary_event_constraint_and_zero_events() -> None:
    with pytest.raises(ValidationError):
        _output(events=[_event_payload(), _event_payload()])
    zero = AIEventOutput.model_validate({"events": [], "financial_facts": [], "warnings": []})
    assert zero.events == []


def test_unknown_is_default_semantic_and_unchanged_requires_explicit_evidence() -> None:
    ordinary = AIEventOutput.model_validate(
        {"events": [], "financial_facts": [_fact_payload()], "warnings": []}
    )
    assert ordinary.financial_facts[0].change_direction.value == "UNKNOWN"
    with pytest.raises(ValidationError):
        AIEventOutput.model_validate(
            {
                "events": [],
                "financial_facts": [_fact_payload(change_direction="UNCHANGED")],
                "warnings": [],
            }
        )
    explicit = _fact_payload(
        change_direction="UNCHANGED",
        evidence_text="Revenue was unchanged",
    )
    parsed = AIEventOutput.model_validate(
        {"events": [], "financial_facts": [explicit], "warnings": []}
    )
    assert parsed.financial_facts[0].change_direction.value == "UNCHANGED"


@pytest.mark.asyncio
async def test_fake_client_and_file_cache_hit_miss(tmp_path: Path) -> None:
    client = FakeModelClient(_output())
    analyzer = AnalyzeAIEvent(client, JsonFileAIEventCache(tmp_path / "cache"))
    first = await analyzer.execute(_command())
    second = await analyzer.execute(_command())
    refreshed = await analyzer.execute(replace(_command(), force_refresh=True))
    assert client.calls == 2
    assert first.metadata.cached is False
    assert second.metadata.cached is True
    assert refreshed.metadata.cached is False
    assert first.analysis.financial_facts[0].normalized_value == Decimal("100")
    assert first.analysis.events[0].start_position == 0


def test_cache_key_changes_with_every_material_input() -> None:
    request = build_model_request(_command())
    base = cache_key(request)
    variants = (
        replace(request, raw_content="different"),
        replace(request, requested_model="different-model"),
        replace(request, prompt_hash="different-prompt"),
        replace(request, schema_hash="different-schema"),
        replace(request, analyzer_version="different-version"),
        replace(request, reasoning_effort="medium"),
        replace(request, max_output_tokens=999),
    )
    assert all(cache_key(item) != base for item in variants)


def test_failure_is_sanitized() -> None:
    failure = sanitize_failure(AIModelError("secret response body and sensitive token"))
    assert failure.error_code == "AIModelError"
    assert failure.message == "AI event analysis failed"
    assert "secret" not in failure.message


@pytest.mark.asyncio
async def test_retry_is_bounded_and_succeeds_after_transient_errors() -> None:
    client = TransientFakeClient(2, _output())

    async def no_sleep(_: float) -> None:
        return None

    reliable = ReliableAIEventModelClient(
        client,
        timeout_seconds=1,
        max_retries=2,
        max_concurrency=1,
        sleep=no_sleep,
    )
    completion = await reliable.complete(build_model_request(_command()))
    assert completion.response_id == "resp_retry"
    assert client.calls == 3

    exhausted = TransientFakeClient(3, _output())
    reliable_exhausted = ReliableAIEventModelClient(
        exhausted,
        timeout_seconds=1,
        max_retries=2,
        max_concurrency=1,
        sleep=no_sleep,
    )
    with pytest.raises(AIModelError, match="bounded retries"):
        await reliable_exhausted.complete(build_model_request(_command()))
    assert exhausted.calls == 3


def test_no_api_key_is_a_configuration_error() -> None:
    with pytest.raises(AIConfigurationError, match="OPENAI_API_KEY"):
        create_ai_event_analyzer(Settings(openai_api_key=None))


@pytest.mark.asyncio
async def test_prediction_converts_to_existing_evaluator_inputs(tmp_path: Path) -> None:
    analyzer = AnalyzeAIEvent(
        FakeModelClient(_output()),
        JsonFileAIEventCache(tmp_path / "cache"),
    )
    result = await analyzer.execute(_command())
    gold = GoldEvent(
        event_type=EventType.FINANCIAL_RESULTS,
        evidence_text="Revenue was 100",
        start_position=0,
        end_position=15,
        is_primary=True,
    )
    event_input, fact_input = to_evaluation_inputs(
        gold_events=[gold],
        gold_facts=[],
        prediction=result,
    )
    assert event_input.predicted_events == result.analysis.events
    assert event_input.predicted_primary_event_type == EventType.FINANCIAL_RESULTS
    assert fact_input.predicted_facts == result.analysis.financial_facts


def test_frozen_test_guard_requires_flag_file_and_matching_config(tmp_path: Path) -> None:
    expected: dict[str, object] = {"prompt_hash": "abc", "dataset_id": "dataset"}
    with pytest.raises(FrozenTestGuardError, match="allow-frozen-test"):
        validate_test_access(
            split=DatasetSplit.TEST,
            allow_frozen_test=False,
            frozen_config_path=None,
            expected_config=expected,
        )
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert (
        validate_test_access(
            split=DatasetSplit.TEST,
            allow_frozen_test=True,
            frozen_config_path=path,
            expected_config=expected,
        )
        == expected
    )
    path.write_text(json.dumps({**expected, "prompt_hash": "changed"}), encoding="utf-8")
    with pytest.raises(FrozenTestGuardError, match="prompt_hash"):
        validate_test_access(
            split=DatasetSplit.TEST,
            allow_frozen_test=True,
            frozen_config_path=path,
            expected_config=expected,
        )


def test_prompt_is_zero_shot_and_contains_no_dataset_identifiers() -> None:
    assert "zero-shot" in SYSTEM_PROMPT
    assert "7dbd116f" not in SYSTEM_PROMPT
    assert "TRAIN" not in SYSTEM_PROMPT
    assert "VALIDATION" not in SYSTEM_PROMPT
    assert "TEST" not in SYSTEM_PROMPT
