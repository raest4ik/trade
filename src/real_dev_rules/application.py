from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from src.ai_events.application.serialization import analysis_result_to_json, failure_to_json
from src.ai_events.application.use_cases import (
    AIEventAnalysisResult,
    AIItemFailure,
    AnalyzeAIEvent,
    AnalyzeAIEventCommand,
    sanitize_failure,
)
from src.ai_events.domain.prompt import (
    ANALYSIS_VERSION as AI_ANALYSIS_VERSION,
)
from src.ai_events.domain.prompt import (
    FACT_EXTRACTOR_VERSION as AI_FACT_VERSION,
)
from src.ai_events.domain.prompt import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    prompt_hash,
    schema_hash,
)
from src.ai_events.infrastructure.factory import (
    AIProviderConfig,
    create_ai_event_analyzer,
    resolve_ai_provider_config,
)
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.events.domain.entities import NewsEventAnalysis
from src.events.domain.v3 import (
    EVENT_ANALYSIS_V3_VERSION,
    FINANCIAL_FACTS_V3_VERSION,
    rules_v3_fingerprint,
)
from src.real_dev_rules.domain import DevelopmentGoldDataset, DevelopmentGoldRecord
from src.shared.config.settings import Settings, get_settings

FROZEN_QWEN_MODEL = "qwen3.5:9b"
FROZEN_QWEN_CONFIG = {
    "context_length": 4096,
    "model": FROZEN_QWEN_MODEL,
    "provider": "ollama",
    "random_seed": 0,
    "think": False,
}


class DeterministicAnalyzer(Protocol):
    def analyze(self, *, news_id: Any, raw_content: str) -> NewsEventAnalysis: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    predictions: tuple[dict[str, Any], ...]
    analyses: tuple[NewsEventAnalysis | None, ...]
    metrics: dict[str, Any]
    errors: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class QwenRunResult:
    evaluation: EvaluationResult
    successful: int
    failed: int
    mean_latency_ms: float


def evaluate_deterministic(
    dataset: DevelopmentGoldDataset,
    analyzer: DeterministicAnalyzer,
    *,
    label: str,
) -> EvaluationResult:
    analyses = tuple(
        analyzer.analyze(news_id=record.news_id, raw_content=record.annotation_text)
        for record in dataset.records
    )
    return _evaluate_analyses(dataset.records, analyses, label=label)


async def evaluate_frozen_qwen(
    dataset: DevelopmentGoldDataset,
    *,
    cache_directory: Path,
    settings: Settings | None = None,
) -> QwenRunResult:
    resolved_settings = settings or get_settings()
    provider = resolve_ai_provider_config(
        resolved_settings,
        provider_override="ollama",
        model_override=FROZEN_QWEN_MODEL,
    )
    _validate_qwen_config(provider)
    analyzer = create_ai_event_analyzer(
        resolved_settings,
        cache_directory=cache_directory,
        provider_override="ollama",
        model_override=FROZEN_QWEN_MODEL,
    )
    results = await asyncio.gather(
        *[
            _analyze_qwen_record(analyzer, provider, resolved_settings, record)
            for record in dataset.records
        ]
    )
    analyses = tuple(result.analysis if result is not None else None for result, _ in results)
    evaluation = _evaluate_analyses(dataset.records, analyses, label="qwen-frozen")
    predictions: list[dict[str, Any]] = []
    latencies: list[int] = []
    failed = 0
    for record, (result, failure) in zip(dataset.records, results, strict=True):
        if result is not None:
            payload = cast("dict[str, Any]", analysis_result_to_json(result))
            payload["news_id"] = str(record.news_id)
            predictions.append(payload)
            latencies.append(result.metadata.latency_ms)
        else:
            assert failure is not None
            failed += 1
            payload = cast("dict[str, Any]", failure_to_json(failure))
            payload["news_id"] = str(record.news_id)
            predictions.append(payload)
    evaluation = EvaluationResult(
        predictions=tuple(predictions),
        analyses=evaluation.analyses,
        metrics={
            **evaluation.metrics,
            "config": frozen_qwen_manifest(),
            "failed": failed,
            "mean_latency_ms": 0.0 if not latencies else sum(latencies) / len(latencies),
            "successful": len(dataset.records) - failed,
        },
        errors=evaluation.errors,
    )
    return QwenRunResult(
        evaluation=evaluation,
        successful=len(dataset.records) - failed,
        failed=failed,
        mean_latency_ms=0.0 if not latencies else sum(latencies) / len(latencies),
    )


def write_baseline_artifacts(
    *,
    output_directory: Path,
    dataset: DevelopmentGoldDataset,
    rules_v2: EvaluationResult,
    qwen: QwenRunResult,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_directory / "rules-v2-predictions.jsonl", rules_v2.predictions)
    _write_jsonl(output_directory / "qwen-predictions.jsonl", qwen.evaluation.predictions)
    comparison = _comparison_rows(dataset.records, rules_v2.analyses, qwen.evaluation.analyses)
    _write_jsonl(output_directory / "comparison.jsonl", comparison)
    errors = [
        *({"system": "rules-v2", **item} for item in rules_v2.errors),
        *({"system": "qwen-frozen", **item} for item in qwen.evaluation.errors),
    ]
    _write_jsonl(output_directory / "errors.jsonl", errors)
    _write_json(
        output_directory / "metrics.json",
        {
            "dataset_sha256": dataset.dataset_sha256,
            "development_performance_only": True,
            "qwen_frozen": qwen.evaluation.metrics,
            "rules_v2": rules_v2.metrics,
            "split_sha256": dataset.split_sha256,
        },
    )


def write_candidate_artifacts(
    *,
    output_directory: Path,
    dataset: DevelopmentGoldDataset,
    rules_v2: EvaluationResult,
    rules_v3: EvaluationResult,
    qwen: QwenRunResult,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_directory / "rules-v3-predictions.jsonl", rules_v3.predictions)
    _write_json(
        output_directory / "metrics.json",
        {
            "development_performance_only": True,
            "qwen_frozen": qwen.evaluation.metrics,
            "rules_v2": rules_v2.metrics,
            "rules_v3": rules_v3.metrics,
        },
    )
    _write_json(
        output_directory / "manifest.json",
        {
            "analysis_version": EVENT_ANALYSIS_V3_VERSION,
            "candidate_name": "event-rules-v3-real-dev-candidate",
            "config": {
                "development_only": True,
                "financial_facts_version": FINANCIAL_FACTS_V3_VERSION,
                "hybrid": False,
                "qwen_frozen": frozen_qwen_manifest(),
                "qwen_used_for_rule_design": False,
            },
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "development_gold_sha256": dataset.dataset_sha256,
            "development_metrics": rules_v3.metrics,
            "frozen": True,
            "git_sha": _git_sha(),
            "holdout_predictions_created": False,
            "rules_fingerprint_sha256": rules_v3_fingerprint(),
            "split_sha256": dataset.split_sha256,
            "version": EVENT_ANALYSIS_V3_VERSION,
        },
    )


def frozen_qwen_manifest() -> dict[str, Any]:
    return {
        **FROZEN_QWEN_CONFIG,
        "analysis_version": AI_ANALYSIS_VERSION,
        "fact_version": AI_FACT_VERSION,
        "prompt_sha256": prompt_hash(),
        "prompt_version": PROMPT_VERSION,
        "schema_sha256": schema_hash(),
        "schema_version": SCHEMA_VERSION,
    }


async def _analyze_qwen_record(
    analyzer: AnalyzeAIEvent,
    provider: AIProviderConfig,
    settings: Settings,
    record: DevelopmentGoldRecord,
) -> tuple[AIEventAnalysisResult | None, AIItemFailure | None]:
    try:
        result = await analyzer.execute(
            AnalyzeAIEventCommand(
                provider=provider.provider.value,
                raw_content=record.annotation_text,
                requested_model=provider.requested_model,
                reasoning_effort=provider.reasoning_effort,
                max_output_tokens=settings.ai_max_output_tokens,
                think=provider.think,
                random_seed=provider.random_seed,
                context_length=provider.context_length,
                news_id=record.news_id,
                record_id=str(record.news_id),
            )
        )
    except Exception as exc:
        return None, sanitize_failure(exc, record_id=str(record.news_id), news_id=record.news_id)
    return result, None


def _validate_qwen_config(provider: AIProviderConfig) -> None:
    actual = {
        "context_length": provider.context_length,
        "model": provider.requested_model,
        "provider": provider.provider.value,
        "random_seed": provider.random_seed,
        "think": provider.think,
    }
    if actual != FROZEN_QWEN_CONFIG:
        raise ValueError("Qwen config differs from the frozen DEVELOPMENT baseline")


def _evaluate_analyses(
    records: Sequence[DevelopmentGoldRecord],
    analyses: Sequence[NewsEventAnalysis | None],
    *,
    label: str,
) -> EvaluationResult:
    event_inputs: list[EventEvaluationInput] = []
    fact_inputs: list[FactEvaluationInput] = []
    predictions: list[dict[str, Any]] = []
    for record, analysis in zip(records, analyses, strict=True):
        event_inputs.append(
            EventEvaluationInput(
                gold_events=record.events,
                predicted_events=[] if analysis is None else analysis.events,
                gold_primary_event_type=record.primary_event,
                predicted_primary_event_type=None
                if analysis is None
                else analysis.primary_event_type,
                prediction_status="FAILED" if analysis is None else analysis.status.value,
            )
        )
        fact_inputs.append(
            FactEvaluationInput(
                gold_facts=record.facts,
                predicted_facts=[] if analysis is None else analysis.financial_facts,
            )
        )
        if analysis is not None:
            predictions.append(_analysis_payload(record.news_id, analysis))
    event_result = evaluate_event_predictions(event_inputs)
    fact_result = evaluate_fact_predictions(fact_inputs)
    errors = tuple(
        _tag_error(item, records, category="event") for item in event_result.errors
    ) + tuple(_tag_error(item, records, category="fact") for item in fact_result.errors)
    return EvaluationResult(
        predictions=tuple(predictions),
        analyses=tuple(analyses),
        metrics={
            "development_performance_only": True,
            "events": event_result.metrics,
            "facts": fact_result.metrics,
            "label": label,
        },
        errors=errors,
    )


def _analysis_payload(news_id: Any, analysis: NewsEventAnalysis) -> dict[str, Any]:
    return {
        "analysis_version": analysis.analysis_version,
        "events": [
            {
                "confidence": str(item.confidence),
                "end_position": item.end_position,
                "event_type": item.event_type.value,
                "evidence_text": item.evidence_text,
                "rule_id": item.rule_id,
                "start_position": item.start_position,
            }
            for item in analysis.events
        ],
        "financial_facts": [
            {
                "change_direction": item.change_direction.value,
                "change_unit": None if item.change_unit is None else item.change_unit.value,
                "change_value": None if item.change_value is None else str(item.change_value),
                "comparison_type": item.comparison_type.value,
                "currency": item.currency.value,
                "end_position": item.end_position,
                "extractor_version": item.extractor_version,
                "fact_role": item.fact_role.value,
                "metric": item.metric.value,
                "normalized_value": str(item.normalized_value),
                "period_type": item.period_type.value,
                "period_year": item.year,
                "rule_id": item.rule_id,
                "scale": item.scale.value,
                "start_position": item.start_position,
                "unit": item.unit.value,
            }
            for item in analysis.financial_facts
        ],
        "news_id": str(news_id),
        "primary_event": analysis.primary_event_type.value,
        "status": analysis.status.value,
    }


def _comparison_rows(
    records: Sequence[DevelopmentGoldRecord],
    rules: Sequence[NewsEventAnalysis | None],
    qwen: Sequence[NewsEventAnalysis | None],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "gold": record.primary_event.value,
            "news_id": str(record.news_id),
            "qwen": None if ai is None else ai.primary_event_type.value,
            "rules_v2": None if deterministic is None else deterministic.primary_event_type.value,
        }
        for record, deterministic, ai in zip(records, rules, qwen, strict=True)
    )


def _tag_error(
    error: dict[str, object], records: Sequence[DevelopmentGoldRecord], *, category: str
) -> dict[str, Any]:
    payload = cast("dict[str, Any]", dict(error))
    index = payload.get("example_index")
    if isinstance(index, int) and 0 <= index < len(records):
        payload["news_id"] = str(records[index].news_id)
    payload["category"] = category
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for item in payloads
        ),
        encoding="utf-8",
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
