from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.ai_events.application.evaluation import evaluate_ai_metric_views, to_evaluation_inputs
from src.ai_events.application.frozen_test import FrozenTestGuardError, validate_test_access
from src.ai_events.application.ports import AIModelCompletion, AIModelRequest
from src.ai_events.application.reliability import ReliableAIEventModelClient
from src.ai_events.application.serialization import analysis_result_to_json, failure_to_json
from src.ai_events.application.use_cases import (
    AnalyzeAIEvent,
    AnalyzeAIEventCommand,
    build_model_request,
    cache_key,
    sanitize_failure,
)
from src.ai_events.domain.evidence import resolve_evidence, resolve_exact_evidence
from src.ai_events.domain.exceptions import (
    AIConfigurationError,
    AIModelError,
    AIModelTransientError,
    AIOutputValidationError,
    OllamaUnavailableError,
)
from src.ai_events.domain.prompt import SYSTEM_PROMPT, output_schema
from src.ai_events.domain.schema import AIEventOutput, parse_decimal_string
from src.ai_events.infrastructure.cache import JsonFileAIEventCache
from src.ai_events.infrastructure.factory import (
    create_ai_event_analyzer,
    resolve_ai_provider_config,
)
from src.evaluation.domain.entities import GoldEvent
from src.evaluation.domain.enums import DatasetSplit
from src.evaluation.domain.metrics import EventEvaluationInput, FactEvaluationInput
from src.events.domain.entities import DetectedEvent
from src.events.domain.enums import EventType
from src.shared.config.settings import Settings


class FakeModelClient:
    def __init__(self, output: AIEventOutput) -> None:
        self.output = output
        self.calls = 0

    async def analyze(self, request: AIModelRequest) -> AIModelCompletion:
        self.calls += 1
        return AIModelCompletion(
            output=self.output,
            response_id="resp_fake",
            actual_model=request.requested_model,
            latency_ms=12,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            provider_metadata={},
            cloud_cost_usd=None,
        )


class TransientFakeClient:
    def __init__(self, failures: int, output: AIEventOutput) -> None:
        self.failures = failures
        self.output = output
        self.calls = 0

    async def analyze(self, request: AIModelRequest) -> AIModelCompletion:
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
            provider_metadata={},
            cloud_cost_usd=None,
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
        "period_year": 2025,
        "period_quarter": None,
        "comparison_type": "NONE",
        "change_direction": "UNKNOWN",
        "change_value": None,
        "change_unit": None,
        "evidence_text": "Revenue was 100",
        "confidence": 0.9,
    }
    payload.update(changes)
    return payload


def _output(
    *,
    events: list[dict[str, object]] | None = None,
    facts: list[dict[str, object]] | None = None,
) -> AIEventOutput:
    return AIEventOutput.model_validate(
        {
            "events": [_event_payload()] if events is None else events,
            "financial_facts": [_fact_payload()] if facts is None else facts,
            "warnings": [],
        }
    )


def _command(text: str = "Revenue was 100") -> AnalyzeAIEventCommand:
    return AnalyzeAIEventCommand(
        provider="openai",
        raw_content=text,
        requested_model="gpt-5-mini",
        reasoning_effort="low",
        max_output_tokens=1000,
        think=False,
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
    fact_schema = cast("dict[str, object]", definitions["AIFinancialFactPrediction"])
    fact_properties = cast("dict[str, object]", fact_schema["properties"])
    assert "period_year" in fact_properties
    assert "period_quarter" in fact_properties
    assert "year" not in fact_properties


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
    with pytest.raises(ValueError):
        parse_decimal_string("01")
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


def test_safe_evidence_alignment_maps_offsets_to_original_text() -> None:
    nbsp = resolve_evidence("Revenue\u00a0was 100", "Revenue was 100")
    assert (nbsp.valid, nbsp.start, nbsp.end) == (True, 0, 15)
    assert nbsp.warning == "evidence aligned after safe whitespace/unicode normalization"

    multiline = resolve_evidence("Revenue\r\n  was 100", "Revenue was 100")
    assert (multiline.valid, multiline.start, multiline.end) == (True, 0, 18)

    quote = resolve_evidence("Revenue was \u201c100\u201d", 'Revenue was "100"')
    assert (quote.valid, quote.start, quote.end) == (True, 0, 17)


def test_evidence_alignment_does_not_fuzzy_match_words_or_punctuation() -> None:
    word = resolve_evidence("Revenue was 100", "Reveneu was 100")
    punctuation = resolve_evidence("EBITDA was 100", "EBIT,DA was 100")
    assert (word.valid, word.start, word.end) == (False, None, None)
    assert (punctuation.valid, punctuation.start, punctuation.end) == (False, None, None)


def test_primary_event_constraint_and_zero_events() -> None:
    with pytest.raises(ValidationError):
        _output(events=[_event_payload(), _event_payload()])
    secondary = {
        **_event_payload(primary=False),
        "event_type": "DIVIDEND",
    }
    multiple = _output(events=[_event_payload(), secondary])
    assert sum(event.is_primary for event in multiple.events) == 1
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
    payload = analysis_result_to_json(first)
    metadata = cast("dict[str, object]", payload["metadata"])
    assert metadata["actual_response_model"] == "gpt-5-mini"
    assert "prompt_sha256" in metadata
    fact = cast("list[dict[str, object]]", payload["financial_facts"])[0]
    assert fact["period_year"] == 2025


@pytest.mark.asyncio
async def test_invalid_evidence_preserves_semantics_and_serializes_null_offsets(
    tmp_path: Path,
) -> None:
    output = _output(
        events=[{**_event_payload(), "evidence_text": "Revenue was 101"}],
        facts=[_fact_payload(evidence_text="Revenue was 101")],
    )
    result = await AnalyzeAIEvent(
        FakeModelClient(output),
        JsonFileAIEventCache(tmp_path / "cache"),
    ).execute(_command())
    assert result.analysis.primary_event_type == EventType.FINANCIAL_RESULTS
    assert result.analysis.financial_facts[0].normalized_value == Decimal("100")
    assert result.analysis.events[0].start_position == -1
    assert result.analysis.financial_facts[0].end_position == -1
    assert sum("offsets unavailable" in warning for warning in result.warnings) == 2

    payload = analysis_result_to_json(result)
    event = cast("list[dict[str, object]]", payload["events"])[0]
    fact = cast("list[dict[str, object]]", payload["financial_facts"])[0]
    assert event["evidence_valid"] is False
    assert event["start_position"] is None
    assert event["end_position"] is None
    assert fact["evidence_valid"] is False
    assert fact["start_position"] is None
    assert fact["end_position"] is None


@pytest.mark.asyncio
async def test_missing_primary_is_a_nonfatal_validation_warning(tmp_path: Path) -> None:
    output = _output(events=[_event_payload(primary=False)])
    result = await AnalyzeAIEvent(
        FakeModelClient(output),
        JsonFileAIEventCache(tmp_path / "cache"),
    ).execute(_command())
    assert result.analysis.primary_event_type == EventType.UNKNOWN
    assert "events are present but no primary event was marked" in result.warnings


def test_absolute_scaled_and_percentage_values_remain_lossless_decimals() -> None:
    facts = [
        _fact_payload(normalized_value="141200000000", scale="BILLION"),
        _fact_payload(normalized_value="72500000", scale="MILLION"),
        _fact_payload(
            normalized_value="15.4",
            unit="PERCENT",
            currency="UNSPECIFIED",
            scale="ONE",
            change_direction="UP",
            change_value="8.4",
            change_unit="PERCENT",
        ),
    ]
    output = _output(facts=facts)
    assert [item.decimal_value() for item in output.financial_facts] == [
        Decimal("141200000000"),
        Decimal("72500000"),
        Decimal("15.4"),
    ]
    assert output.financial_facts[2].decimal_change_value() == Decimal("8.4")
    change_unit = output.financial_facts[2].change_unit
    assert change_unit is not None
    assert change_unit.value == "PERCENT"


@pytest.mark.asyncio
async def test_dividend_semantics_survive_hardening(tmp_path: Path) -> None:
    text = "".join(
        (
            "\u0421\u043e\u0432\u0435\u0442 ",
            "\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u043e\u0432 ",
            "\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438 ",
            "\u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u043e\u0432\u0430\u043b ",
            "\u0432\u044b\u043f\u043b\u0430\u0442\u0438\u0442\u044c ",
            "\u0434\u0438\u0432\u0438\u0434\u0435\u043d\u0434\u044b ",
            "\u0437\u0430 2025 \u0433\u043e\u0434 ",
            "\u0432 \u0440\u0430\u0437\u043c\u0435\u0440\u0435 35 ",
            "\u0440\u0443\u0431\u043b\u0435\u0439 \u043d\u0430 ",
            "\u0430\u043a\u0446\u0438\u044e.",
        )
    )
    output = _output(
        events=[
            {
                "event_type": "DIVIDEND",
                "is_primary": True,
                "confidence": 1,
                "evidence_text": text,
            }
        ],
        facts=[
            _fact_payload(
                metric="DIVIDEND_PER_SHARE",
                normalized_value="35",
                currency="RUB",
                scale="ONE",
                fact_role="FORECAST",
                period_type="YEAR",
                period_year=2025,
                evidence_text=text,
            )
        ],
    )
    result = await AnalyzeAIEvent(
        FakeModelClient(output),
        JsonFileAIEventCache(tmp_path / "cache"),
    ).execute(_command(text))
    fact = result.analysis.financial_facts[0]
    assert result.analysis.primary_event_type == EventType.DIVIDEND
    assert fact.metric.value == "DIVIDEND_PER_SHARE"
    assert fact.normalized_value == Decimal("35")
    assert fact.currency.value == "RUB"
    assert fact.fact_role.value == "FORECAST"
    assert (fact.period_type.value, fact.year) == ("YEAR", 2025)


def test_successful_only_and_end_to_end_metric_views_count_failures() -> None:
    gold = GoldEvent(
        event_type=EventType.FINANCIAL_RESULTS,
        evidence_text="Revenue was 100",
        start_position=0,
        end_position=15,
        is_primary=True,
    )
    predicted = DetectedEvent(
        id=uuid4(),
        analysis_id=uuid4(),
        event_type=EventType.FINANCIAL_RESULTS,
        confidence=Decimal("1"),
        rule_id="test",
        matched_rule="test",
        evidence_text="Revenue was 100",
        start_position=0,
        end_position=15,
    )
    successful_event = EventEvaluationInput(
        gold_events=[gold],
        predicted_events=[predicted],
        gold_primary_event_type=EventType.FINANCIAL_RESULTS,
        predicted_primary_event_type=EventType.FINANCIAL_RESULTS,
        prediction_status="COMPLETE",
    )
    failed_event = EventEvaluationInput(
        gold_events=[gold],
        predicted_events=[],
        gold_primary_event_type=EventType.FINANCIAL_RESULTS,
        predicted_primary_event_type=None,
        prediction_status="FAILED",
    )
    views = evaluate_ai_metric_views(
        successful_event_inputs=[successful_event],
        successful_fact_inputs=[FactEvaluationInput(gold_facts=[], predicted_facts=[])],
        failed_event_inputs=[failed_event],
        failed_fact_inputs=[FactEvaluationInput(gold_facts=[], predicted_facts=[])],
    )
    successful_micro = cast("dict[str, object]", views.successful_events.metrics["micro"])
    end_to_end_micro = cast("dict[str, object]", views.end_to_end_events.metrics["micro"])
    assert successful_micro["f1"] == 1.0
    assert end_to_end_micro["f1"] == 0.666667
    assert (views.requested_count, views.successful_count, views.failed_count) == (2, 1, 1)
    assert views.item_success_rate == 0.5


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
        replace(request, provider="ollama"),
        replace(request, think=True),
    )
    assert all(cache_key(item) != base for item in variants)


def test_failure_is_sanitized() -> None:
    failure = sanitize_failure(AIModelError("secret response body and sensitive token"))
    assert failure.error_code == "AIModelError"
    assert failure.message == "AI event analysis failed"
    assert "secret" not in failure.message
    payload = failure_to_json(failure)
    assert payload["status"] == "FAILED"
    assert payload["error_type"] == "AIModelError"

    ollama_failure = sanitize_failure(OllamaUnavailableError("http://localhost:11434"))
    assert ollama_failure.error_code == "OLLAMA_UNAVAILABLE"
    assert ollama_failure.message == "Ollama is unavailable at http://localhost:11434"


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
    completion = await reliable.analyze(build_model_request(_command()))
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
    with pytest.raises(AIModelTransientError, match="temporary"):
        await reliable_exhausted.analyze(build_model_request(_command()))
    assert exhausted.calls == 3


def test_default_ollama_provider_needs_no_api_key() -> None:
    settings = Settings(openai_api_key=None)
    provider = resolve_ai_provider_config(settings)
    assert provider.provider.value == "ollama"
    assert provider.requested_model == "qwen3:8b"
    assert provider.think is False
    create_ai_event_analyzer(settings)


def test_openai_provider_without_key_is_a_configuration_error() -> None:
    with pytest.raises(AIConfigurationError, match="OPENAI_API_KEY"):
        create_ai_event_analyzer(Settings(openai_api_key=None), provider_override="openai")


def test_openai_sdk_is_an_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(_: str) -> object:
        raise ModuleNotFoundError("openai")

    monkeypatch.setattr(
        "src.ai_events.infrastructure.openai_client.importlib.import_module",
        missing_sdk,
    )
    with pytest.raises(AIConfigurationError, match="uv sync --extra openai"):
        create_ai_event_analyzer(
            Settings(openai_api_key="test-only"),
            provider_override="openai",
        )


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
    assert "converted to BASE UNITS" in SYSTEM_PROMPT
    assert "BILLION = 1,000,000,000" in SYSTEM_PROMPT
    assert "never divide it by 100" in SYSTEM_PROMPT
    assert "Extract each economically meaningful supported KPI once" in SYSTEM_PROMPT
    assert "period_quarter only when period_type is QUARTER" in SYSTEM_PROMPT
    assert "Never invent zero" in SYSTEM_PROMPT
    assert "Never translate, normalize, reconstruct, or paraphrase evidence_text" in SYSTEM_PROMPT
    assert "7dbd116f" not in SYSTEM_PROMPT
    assert "TRAIN" not in SYSTEM_PROMPT
    assert "VALIDATION" not in SYSTEM_PROMPT
    assert "TEST" not in SYSTEM_PROMPT
