from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast
from urllib.parse import urlparse
from uuid import UUID

from src.evaluation.domain.entities import (
    GOLD_SCHEMA_VERSION,
    AnnotationExample,
    GoldEvent,
    GoldFinancialFact,
)
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.evaluation.domain.serialization import annotation_from_json, annotation_to_json
from src.evaluation.domain.validation import validate_jsonl_payloads
from src.events.domain.entities import NewsEventAnalysis
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)

DATASET_NAME = "ru-corporate-events-real-batch-003-gold-v1"
DATASET_VERSION = "1"
SOURCE_BATCH = "003"
PROVENANCE = "REAL"
REVIEW_BASIS = "EXCERPT_ONLY"
HUMAN_REVIEW_BASIS = "annotation_text_excerpt_only"
EXPECTED_RECORDS = 26
EXPECTED_EVENT_DISTRIBUTION = {
    EventType.OTHER.value: 24,
    EventType.FINANCIAL_RESULTS.value: 1,
    EventType.GUIDANCE.value: 1,
}
OLD_BATCH_001_SHA256 = "4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196"
ALLOWED_SOURCE_TICKERS = {
    "ROSNEFT_PRESS_RELEASES_RSS": "ROSN",
    "YANDEX_IR_PRESS_RELEASES_RSS": "YDEX",
}
BENCHMARK_WARNINGS = (
    "SMALL_SAMPLE",
    "CLASS_IMBALANCE",
    "LOW_SOURCE_DIVERSITY",
    "LOW_TICKER_DIVERSITY",
)

_FACT_EVIDENCE_SUFFIX = {
    "developers_regularly_using_ai_share": "разработчиков",
    "company_changes_with_ai_participation_share": "изменений",
    "ai_generated_code_share_within_ai_assisted_changes": "кода",
}
_FORBIDDEN_INPUT_FIELDS = {
    "abnormal_return",
    "future_volume",
    "gold_events",
    "gold_financial_facts",
    "human_events",
    "human_financial_facts",
    "human_primary_event",
    "market_reaction",
    "post_event_price",
    "predicted_events",
    "reaction_labels",
    "rules_primary_event",
}


class BenchmarkValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalizationIssue:
    news_id: str
    code: str
    message: str
    severity: str = "WARNING"

    def payload(self) -> dict[str, str]:
        return {
            "news_id": self.news_id,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    record_id: str
    ticker: str
    source_code: str
    source_item_id: str
    source_url: str
    storage_policy: str
    review_basis: str
    annotation: AnnotationExample

    @property
    def gold_primary_event(self) -> EventType:
        primary = next((item for item in self.annotation.gold_events if item.is_primary), None)
        if primary is None:
            raise BenchmarkValidationError(f"gold primary event missing for {self.record_id}")
        return primary.event_type

    def index_payload(self) -> dict[str, object]:
        return {
            "news_id": str(self.annotation.news_id),
            "record_id": self.record_id,
            "ticker": self.ticker,
            "source_code": self.source_code,
            "source_item_id": self.source_item_id,
            "source_url": self.source_url,
            "storage_policy": self.storage_policy,
            "review_basis": self.review_basis,
        }


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    source_file_sha256: str
    examples: tuple[BenchmarkExample, ...]
    issues: tuple[CanonicalizationIssue, ...]
    canonical_bytes: bytes
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    dataset_path: Path
    manifest_path: Path
    dataset_sha256: str
    source_file_sha256: str
    examples: tuple[BenchmarkExample, ...]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    record_id: str
    news_id: UUID
    analysis: NewsEventAnalysis
    runtime: dict[str, object]
    failure: dict[str, object] | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    metrics: dict[str, object]
    errors: tuple[dict[str, object], ...]


def canonicalize_human_review(
    source_path: Path,
    *,
    expected_records: int = EXPECTED_RECORDS,
    expected_event_distribution: dict[str, int] | None = None,
) -> CanonicalDataset:
    if not source_path.is_file():
        raise BenchmarkValidationError(f"human review file does not exist: {source_path}")
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rows = _read_jsonl(source_bytes.decode("utf-8"))
    distribution = expected_event_distribution or EXPECTED_EVENT_DISTRIBUTION
    _validate_human_rows(rows, expected_records=expected_records, distribution=distribution)
    examples: list[BenchmarkExample] = []
    issues: list[CanonicalizationIssue] = []
    for row in rows:
        example, row_issues = _canonical_example(row)
        examples.append(example)
        issues.extend(row_issues)
    examples.sort(key=lambda item: (item.annotation.published_at, str(item.annotation.news_id)))
    canonical_bytes = _canonical_jsonl([annotation_to_json(item.annotation) for item in examples])
    validation = validate_jsonl_payloads(
        canonical_bytes.decode("utf-8").splitlines(),
        raw_content_by_news_id={
            item.annotation.news_id: cast("str", item.annotation.raw_content) for item in examples
        },
        raw_hash_by_news_id={
            item.annotation.news_id: item.annotation.raw_content_hash for item in examples
        },
        strict=True,
    )
    if not validation.ok:
        details = "; ".join(f"{item.code}: {item.message}" for item in validation.errors)
        raise BenchmarkValidationError(f"canonical event-gold-v1 validation failed: {details}")
    return CanonicalDataset(
        source_file_sha256=source_hash,
        examples=tuple(examples),
        issues=tuple(issues),
        canonical_bytes=canonical_bytes,
        dataset_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )


def freeze_canonical_dataset(
    canonical: CanonicalDataset,
    *,
    output_directory: Path,
    old_batch_001_path: Path,
    git_sha: str,
    expected_old_batch_001_sha256: str = OLD_BATCH_001_SHA256,
) -> FrozenDataset:
    old_hash = _sha256_file(old_batch_001_path)
    if old_hash != expected_old_batch_001_sha256:
        raise BenchmarkValidationError("frozen Batch 001 SHA-256 changed")
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = output_directory / "dataset.jsonl"
    manifest_path = output_directory / "manifest.json"
    issues_path = output_directory / "validation-issues.json"
    if dataset_path.exists() and dataset_path.read_bytes() != canonical.canonical_bytes:
        raise BenchmarkValidationError("frozen real-gold dataset differs from canonical input")
    dataset_path.write_bytes(canonical.canonical_bytes)
    if manifest_path.exists():
        existing = _read_object(manifest_path)
        if existing.get("dataset_sha256") != canonical.dataset_sha256:
            raise BenchmarkValidationError("existing frozen manifest has a different dataset hash")
        created_at = str(existing["created_at"])
    else:
        created_at = _utc_now()
    manifest = _dataset_manifest(
        canonical,
        created_at=created_at,
        old_batch_001_hash=old_hash,
        git_sha=git_sha,
    )
    _write_json(manifest_path, manifest)
    _write_json(
        issues_path,
        {
            "dataset_sha256": canonical.dataset_sha256,
            "issue_count": len(canonical.issues),
            "issues": [item.payload() for item in canonical.issues],
        },
    )
    return load_frozen_dataset(dataset_path, manifest_path)


def load_frozen_dataset(dataset_path: Path, manifest_path: Path) -> FrozenDataset:
    dataset_bytes = dataset_path.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    manifest = _read_object(manifest_path)
    if manifest.get("freeze_state") != "FROZEN_BEFORE_PREDICTIONS":
        raise BenchmarkValidationError("dataset was not frozen before predictions")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise BenchmarkValidationError("frozen dataset hash does not match its manifest")
    rows = _read_jsonl(dataset_bytes.decode("utf-8"))
    annotations = [annotation_from_json(row) for row in rows]
    index_values = manifest.get("record_index")
    if not isinstance(index_values, list):
        raise BenchmarkValidationError("frozen manifest record index is missing")
    index_by_id: dict[str, dict[str, object]] = {}
    for value in cast("list[object]", index_values):
        if not isinstance(value, dict):
            raise BenchmarkValidationError("invalid frozen manifest record index")
        item = {str(key): val for key, val in cast("dict[object, object]", value).items()}
        index_by_id[str(item["news_id"])] = item
    examples: list[BenchmarkExample] = []
    for annotation in annotations:
        metadata = index_by_id.get(str(annotation.news_id))
        if metadata is None:
            raise BenchmarkValidationError("frozen manifest does not cover every news_id")
        examples.append(
            BenchmarkExample(
                record_id=str(metadata["record_id"]),
                ticker=str(metadata["ticker"]),
                source_code=str(metadata["source_code"]),
                source_item_id=str(metadata["source_item_id"]),
                source_url=str(metadata["source_url"]),
                storage_policy=str(metadata["storage_policy"]),
                review_basis=str(metadata["review_basis"]),
                annotation=annotation,
            )
        )
    if len(examples) != int(str(manifest["records"])):
        raise BenchmarkValidationError("frozen dataset record count mismatch")
    return FrozenDataset(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        dataset_sha256=dataset_hash,
        source_file_sha256=str(manifest["source_file_sha256"]),
        examples=tuple(examples),
        manifest=manifest,
    )


def analyzer_input(example: BenchmarkExample) -> dict[str, object]:
    payload: dict[str, object] = {
        "news_id": str(example.annotation.news_id),
        "record_id": example.record_id,
        "raw_content": example.annotation.raw_content,
    }
    forbidden = set(payload) & _FORBIDDEN_INPUT_FIELDS
    if forbidden:
        raise BenchmarkValidationError(
            "analyzer input contains forbidden fields: " + ", ".join(sorted(forbidden))
        )
    return payload


def evaluate_prediction_set(
    frozen: FrozenDataset,
    predictions: tuple[BenchmarkPrediction, ...],
    *,
    system_name: str,
) -> PredictionEvaluation:
    prediction_by_id = {item.news_id: item for item in predictions}
    if len(prediction_by_id) != len(predictions):
        raise BenchmarkValidationError("duplicate prediction news_id")
    event_inputs: list[EventEvaluationInput] = []
    fact_inputs: list[FactEvaluationInput] = []
    ordered_predictions: list[BenchmarkPrediction] = []
    for example in frozen.examples:
        prediction = prediction_by_id.get(example.annotation.news_id)
        if prediction is None:
            raise BenchmarkValidationError("prediction set does not cover frozen dataset")
        ordered_predictions.append(prediction)
        analysis = prediction.analysis
        event_inputs.append(
            EventEvaluationInput(
                gold_events=example.annotation.gold_events,
                predicted_events=analysis.events,
                gold_primary_event_type=example.gold_primary_event,
                predicted_primary_event_type=analysis.primary_event_type,
                prediction_status=(
                    "FAILED" if prediction.failure is not None else analysis.status.value
                ),
            )
        )
        fact_inputs.append(
            FactEvaluationInput(
                gold_facts=example.annotation.gold_financial_facts,
                predicted_facts=analysis.financial_facts,
            )
        )
    events = evaluate_event_predictions(event_inputs)
    facts = evaluate_fact_predictions(fact_inputs)
    errors = _benchmark_errors(
        frozen.examples,
        tuple(ordered_predictions),
        fact_errors=facts.errors,
        system_name=system_name,
    )
    metrics: dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "dataset_sha256": frozen.dataset_sha256,
        "system": system_name,
        "requested": len(frozen.examples),
        "successful": sum(item.succeeded for item in ordered_predictions),
        "failed": sum(not item.succeeded for item in ordered_predictions),
        "events": events.metrics,
        "facts": facts.metrics,
        "runtime": _runtime_metrics(ordered_predictions),
    }
    return PredictionEvaluation(metrics=metrics, errors=errors)


def prediction_payload(
    example: BenchmarkExample,
    prediction: BenchmarkPrediction,
) -> dict[str, object]:
    analysis = prediction.analysis
    return {
        "record_id": example.record_id,
        "news_id": str(example.annotation.news_id),
        "ticker": example.ticker,
        "source": example.source_code,
        "input_sha256": example.annotation.raw_content_hash,
        "status": "FAILED" if prediction.failure else analysis.status.value,
        "primary_event": analysis.primary_event_type.value,
        "event_count": len(analysis.events),
        "fact_count": len(analysis.financial_facts),
        "events": [
            {
                "event_type": item.event_type.value,
                "evidence_text": item.evidence_text,
                "start_position": item.start_position,
                "end_position": item.end_position,
                "confidence": str(item.confidence),
            }
            for item in analysis.events
        ],
        "financial_facts": [
            {
                "metric": item.metric.value,
                "normalized_value": str(item.normalized_value),
                "unit": item.unit.value,
                "currency": item.currency.value,
                "scale": item.scale.value,
                "period_type": item.period_type.value,
                "period_year": item.year,
                "period_quarter": item.quarter,
                "fact_role": item.fact_role.value,
                "comparison_type": item.comparison_type.value,
                "change_direction": item.change_direction.value,
                "change_value": None if item.change_value is None else str(item.change_value),
                "change_unit": None if item.change_unit is None else item.change_unit.value,
                "evidence_text": item.evidence_text,
                "start_position": item.start_position,
                "end_position": item.end_position,
            }
            for item in analysis.financial_facts
        ],
        "runtime": prediction.runtime,
        "failure": prediction.failure,
    }


def compare_prediction_sets(
    frozen: FrozenDataset,
    rules: tuple[BenchmarkPrediction, ...],
    qwen: tuple[BenchmarkPrediction, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rules_by_id = {item.news_id: item for item in rules}
    qwen_by_id = {item.news_id: item for item in qwen}
    rows: list[dict[str, object]] = []
    outcomes = Counter[str]()
    disagreements = 0
    for example in frozen.examples:
        rules_item = rules_by_id[example.annotation.news_id]
        qwen_item = qwen_by_id[example.annotation.news_id]
        gold = example.gold_primary_event
        rules_primary = rules_item.analysis.primary_event_type
        qwen_primary = qwen_item.analysis.primary_event_type
        rules_correct = rules_primary == gold
        qwen_correct = qwen_primary == gold
        if rules_correct and qwen_correct:
            outcome = "BOTH_CORRECT"
        elif rules_correct:
            outcome = "RULES_ONLY_CORRECT"
        elif qwen_correct:
            outcome = "QWEN_ONLY_CORRECT"
        else:
            outcome = "BOTH_WRONG"
        outcomes[outcome] += 1
        disagreements += int(rules_primary != qwen_primary)
        rows.append(
            {
                "news_id": str(example.annotation.news_id),
                "ticker": example.ticker,
                "source": example.source_code,
                "gold_primary_event": gold.value,
                "rules_primary_event": rules_primary.value,
                "qwen_primary_event": qwen_primary.value,
                "rules_correct": rules_correct,
                "qwen_correct": qwen_correct,
                "rules_event_count": len(rules_item.analysis.events),
                "qwen_event_count": len(qwen_item.analysis.events),
                "gold_fact_count": len(example.annotation.gold_financial_facts),
                "rules_fact_count": len(rules_item.analysis.financial_facts),
                "qwen_fact_count": len(qwen_item.analysis.financial_facts),
                "outcome": outcome,
            }
        )
    total = len(rows)
    summary: dict[str, object] = {
        "dataset_sha256": frozen.dataset_sha256,
        "records": total,
        "four_way": {
            name: {
                "count": outcomes[name],
                "percentage": _ratio(outcomes[name], total),
            }
            for name in (
                "BOTH_CORRECT",
                "RULES_ONLY_CORRECT",
                "QWEN_ONLY_CORRECT",
                "BOTH_WRONG",
            )
        },
        "ORACLE_UPPER_BOUND": {
            "diagnostic_only": True,
            "primary_accuracy": _ratio(total - outcomes["BOTH_WRONG"], total),
        },
        "rules_vs_qwen_disagreement_count": disagreements,
        "hybrid_predictions_emitted": False,
    }
    return rows, summary


def taxonomy_summary(
    rules_errors: tuple[dict[str, object], ...],
    qwen_errors: tuple[dict[str, object], ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = list(normalize_taxonomy_errors((*rules_errors, *qwen_errors)))
    counts: dict[str, dict[str, int]] = {}
    for system in ("rules-v2", "qwen3.5:9b"):
        counter = Counter(str(item["category"]) for item in rows if item.get("system") == system)
        counts[system] = dict(sorted(counter.items()))
    summary: dict[str, object] = {
        "research_only": True,
        "models_unchanged": True,
        "counts": counts,
        "most_common": {system: _most_common(value) for system, value in counts.items()},
    }
    return rows, summary


def normalize_taxonomy_errors(
    errors: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    for error in errors:
        item = dict(error)
        details = item.get("details")
        if isinstance(details, dict):
            values = cast("dict[object, object]", details)
            error_type = values.get("type")
            if error_type is not None:
                item["category"] = _fact_error_category(str(error_type))
        normalized.append(item)
    return tuple(normalized)


def write_json(path: Path, payload: object) -> None:
    _write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_jsonl(rows))


def sha256_file(path: Path) -> str:
    return _sha256_file(path)


def _validate_human_rows(
    rows: list[dict[str, object]],
    *,
    expected_records: int,
    distribution: dict[str, int],
) -> None:
    errors: list[str] = []
    if len(rows) != expected_records:
        errors.append(f"expected {expected_records} records, got {len(rows)}")
    seen_news: set[str] = set()
    seen_logical: set[tuple[str, str]] = set()
    events = Counter[str]()
    for index, row in enumerate(rows, start=1):
        news_id = str(row.get("news_id", ""))
        try:
            UUID(news_id)
        except ValueError:
            errors.append(f"line {index}: invalid news_id")
        if news_id in seen_news:
            errors.append(f"line {index}: duplicate news_id")
        seen_news.add(news_id)
        source = str(row.get("source_code", ""))
        source_item = str(row.get("source_item_id", ""))
        logical = (source, source_item)
        if logical in seen_logical:
            errors.append(f"line {index}: duplicate logical source record")
        seen_logical.add(logical)
        ticker = str(row.get("ticker", ""))
        if ALLOWED_SOURCE_TICKERS.get(source) != ticker:
            errors.append(f"line {index}: source/ticker is not approved REAL provenance")
        if any(token in source for token in ("SYNTHETIC", "SEED", "BATCH_001")):
            errors.append(f"line {index}: synthetic or seed provenance is forbidden")
        if row.get("timestamp_quality") != "EXACT":
            errors.append(f"line {index}: timestamp must be EXACT")
        if row.get("human_review_status") != "REVIEWED":
            errors.append(f"line {index}: human review must be REVIEWED")
        if row.get("human_review_basis") != HUMAN_REVIEW_BASIS:
            errors.append(f"line {index}: review basis must be excerpt-only")
        if row.get("storage_policy") != "EXCERPT_ALLOWED":
            errors.append(f"line {index}: excerpt storage policy is required")
        text = str(row.get("annotation_text", ""))
        if not text.strip():
            errors.append(f"line {index}: annotation_text is required")
        source_url = urlparse(str(row.get("source_url", "")))
        if source_url.scheme != "https" or not source_url.netloc:
            errors.append(f"line {index}: source URL must use HTTPS")
        primary = str(row.get("human_primary_event", ""))
        try:
            EventType(primary)
        except ValueError:
            errors.append(f"line {index}: unsupported human primary event")
        events[primary] += 1
        human_events = row.get("human_events")
        if not isinstance(human_events, list) or primary not in human_events:
            errors.append(f"line {index}: human events do not contain primary event")
        forbidden = set(row) & {
            "abnormal_return",
            "future_volume",
            "market_reaction",
            "post_event_price",
            "reaction_labels",
        }
        if forbidden:
            errors.append(f"line {index}: future market fields are forbidden")
    if dict(events) != distribution:
        errors.append(f"unexpected human event distribution: {dict(events)}")
    if errors:
        raise BenchmarkValidationError("; ".join(errors))


def _canonical_example(
    row: dict[str, object],
) -> tuple[BenchmarkExample, list[CanonicalizationIssue]]:
    news_id = UUID(str(row["news_id"]))
    raw_content = str(row["annotation_text"])
    primary = EventType(str(row["human_primary_event"]))
    gold_event = GoldEvent(
        event_type=primary,
        evidence_text=raw_content,
        start_position=0,
        end_position=len(raw_content),
        is_primary=True,
        notes="Human primary label reviewed from the complete stored excerpt only.",
    )
    issues: list[CanonicalizationIssue] = []
    facts: list[GoldFinancialFact] = []
    raw_facts = row.get("human_financial_facts")
    if not isinstance(raw_facts, list):
        raise BenchmarkValidationError(f"human_financial_facts must be a list for {news_id}")
    for value in cast("list[object]", raw_facts):
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"invalid human financial fact for {news_id}")
        payload = {str(key): item for key, item in cast("dict[object, object]", value).items()}
        fact, issue = _canonical_fact(str(news_id), raw_content, payload)
        issues.append(issue)
        if fact is not None:
            facts.append(fact)
    published_at = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
    annotation = AnnotationExample(
        schema_version=GOLD_SCHEMA_VERSION,
        news_id=news_id,
        published_at=published_at.astimezone(UTC),
        raw_content_hash=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        split=DatasetSplit.TEST,
        review_status=ReviewStatus.REVIEWED,
        annotator="independent-human-review-batch-003-v1",
        notes=(
            "source_batch=003; provenance=REAL; review_basis=EXCERPT_ONLY; "
            f"human_notes={row.get('human_review_notes') or 'none'}"
        ),
        predicted_events=[],
        predicted_financial_facts=[],
        gold_events=[gold_event],
        gold_financial_facts=facts,
        raw_content=raw_content,
    )
    return (
        BenchmarkExample(
            record_id=str(row["record_id"]),
            ticker=str(row["ticker"]),
            source_code=str(row["source_code"]),
            source_item_id=str(row["source_item_id"]),
            source_url=str(row["source_url"]),
            storage_policy=str(row["storage_policy"]),
            review_basis=REVIEW_BASIS,
            annotation=annotation,
        ),
        issues,
    )


def _canonical_fact(
    news_id: str,
    raw_content: str,
    payload: dict[str, object],
) -> tuple[GoldFinancialFact | None, CanonicalizationIssue]:
    metric_name = str(payload.get("metric_name", ""))
    suffix = _FACT_EVIDENCE_SUFFIX.get(metric_name)
    if suffix is None:
        return None, CanonicalizationIssue(
            news_id,
            "FACT_EXCLUDED_FROM_STRICT_SCORING",
            f"unsupported human metric_name: {metric_name}",
        )
    try:
        raw_value = Decimal(str(payload["value"]))
        metric = FinancialMetric(str(payload["metric"]))
        unit = FactUnit(str(payload["unit"]))
        fact_role = FactRole(str(payload["role"]))
        change_direction = ChangeDirection(str(payload["change_direction"]))
    except (KeyError, ValueError, InvalidOperation) as exc:
        return None, CanonicalizationIssue(
            news_id,
            "FACT_EXCLUDED_FROM_STRICT_SCORING",
            f"fact cannot be represented without coercion: {exc}",
        )
    if str(payload.get("period_text")) != "by end of 2026":
        return None, CanonicalizationIssue(
            news_id,
            "FACT_EXCLUDED_FROM_STRICT_SCORING",
            "period text has no strict event-gold-v1 mapping",
        )
    escaped_value = re.escape(format(raw_value, "f"))
    pattern = re.compile(rf"не&nbsp;менее {escaped_value}% {re.escape(suffix)}")
    match = pattern.search(raw_content)
    if match is None:
        return None, CanonicalizationIssue(
            news_id,
            "FACT_EXCLUDED_FROM_STRICT_SCORING",
            f"exact evidence fragment not found for {metric_name}",
        )
    if unit == FactUnit.PERCENTAGE_POINTS:
        return None, CanonicalizationIssue(
            news_id,
            "FACT_EXCLUDED_FROM_STRICT_SCORING",
            "percentage-points fact cannot be silently converted to percent",
        )
    fact = GoldFinancialFact(
        metric=metric,
        raw_value=raw_value,
        normalized_value=raw_value,
        unit=unit,
        currency=Currency.UNSPECIFIED,
        scale=ValueScale.ONE,
        period_type=PeriodType.YEAR,
        period_year=2026,
        period_quarter=None,
        period_month=None,
        raw_period="by end of 2026",
        fact_role=fact_role,
        comparison_type=ComparisonType.UNKNOWN,
        change_direction=change_direction,
        change_value=None,
        change_unit=None,
        evidence_text=match.group(0),
        start_position=match.start(),
        end_position=match.end(),
        notes=(
            f"metric_name={metric_name}; technical schema mapping only; "
            f"human_notes={payload.get('notes') or 'none'}"
        ),
    )
    return fact, CanonicalizationIssue(
        news_id,
        "TECHNICAL_FACT_SCHEMA_MAPPING",
        (
            f"{metric_name}: OTHER/PERCENT/FORECAST/YEAR-2026; value preserved; "
            "currency UNSPECIFIED; scale ONE; no change value invented"
        ),
    )


def _dataset_manifest(
    canonical: CanonicalDataset,
    *,
    created_at: str,
    old_batch_001_hash: str,
    git_sha: str,
) -> dict[str, object]:
    examples = canonical.examples
    dates = [item.annotation.published_at for item in examples]
    events = Counter(item.gold_primary_event.value for item in examples)
    tickers = Counter(item.ticker for item in examples)
    sources = Counter(item.source_code for item in examples)
    months = Counter(item.annotation.published_at.strftime("%Y-%m") for item in examples)
    return {
        "name": DATASET_NAME,
        "version": DATASET_VERSION,
        "schema_version": GOLD_SCHEMA_VERSION,
        "created_at": created_at,
        "freeze_state": "FROZEN_BEFORE_PREDICTIONS",
        "source_batch": SOURCE_BATCH,
        "provenance": PROVENANCE,
        "review_basis": REVIEW_BASIS,
        "full_text_human_gold": False,
        "records": len(examples),
        "source_file_sha256": canonical.source_file_sha256,
        "dataset_sha256": canonical.dataset_sha256,
        "batch_001_sha256": old_batch_001_hash,
        "batch_001_unchanged": True,
        "tickers": dict(sorted(tickers.items())),
        "sources": dict(sorted(sources.items())),
        "date_range": {
            "from": min(dates).isoformat().replace("+00:00", "Z"),
            "to": max(dates).isoformat().replace("+00:00", "Z"),
        },
        "month_distribution": dict(sorted(months.items())),
        "event_distribution": dict(sorted(events.items())),
        "warnings": list(BENCHMARK_WARNINGS),
        "validation_issue_count": len(canonical.issues),
        "strict_fact_exclusion_count": sum(
            item.code == "FACT_EXCLUDED_FROM_STRICT_SCORING" for item in canonical.issues
        ),
        "analyzer_input_policy": "raw_content only; publication-time excerpt",
        "future_market_fields_included": False,
        "observed_evaluation_set": True,
        "git_commit_sha": git_sha,
        "record_index": [item.index_payload() for item in examples],
    }


def _benchmark_errors(
    examples: tuple[BenchmarkExample, ...],
    predictions: tuple[BenchmarkPrediction, ...],
    *,
    fact_errors: list[dict[str, object]],
    system_name: str,
) -> tuple[dict[str, object], ...]:
    errors: list[dict[str, object]] = []
    for index, (example, prediction) in enumerate(zip(examples, predictions, strict=True)):
        gold = example.gold_primary_event
        predicted = prediction.analysis.primary_event_type
        event_count = len(prediction.analysis.events)
        if prediction.failure is not None:
            errors.append(
                _error_payload(
                    system_name,
                    example,
                    "OTHER_ERROR",
                    gold,
                    predicted,
                    details=prediction.failure,
                )
            )
        elif predicted != gold:
            errors.append(
                _error_payload(
                    system_name,
                    example,
                    _primary_error_category(gold, predicted, event_count),
                    gold,
                    predicted,
                )
            )
        elif event_count > 1:
            errors.append(
                _error_payload(
                    system_name,
                    example,
                    "MULTI_EVENT_OVERDETECTION",
                    gold,
                    predicted,
                )
            )
        for item in fact_errors:
            if item.get("example_index") != index:
                continue
            category = _fact_error_category(str(item.get("type")))
            errors.append(
                _error_payload(
                    system_name,
                    example,
                    category,
                    gold,
                    predicted,
                    details=item,
                )
            )
    return tuple(errors)


def _primary_error_category(
    gold: EventType,
    predicted: EventType,
    event_count: int,
) -> str:
    if gold == EventType.OTHER:
        if predicted == EventType.UNKNOWN:
            return "UNKNOWN_INSTEAD_OF_OTHER"
        return "FALSE_SPECIFIC_EVENT"
    if predicted == EventType.OTHER:
        return "OTHER_INSTEAD_OF_SPECIFIC"
    if predicted == EventType.UNKNOWN or event_count == 0:
        return "MISSED_EVENT"
    return "SPECIFIC_CLASS_CONFUSION"


def _fact_error_category(error_type: str) -> str:
    mapping = {
        "MISSED_FACT": "OTHER_ERROR",
        "EXTRA_FACT": "OTHER_ERROR",
        "WRONG_METRIC": "FACT_METRIC_ERROR",
        "WRONG_VALUE": "FACT_VALUE_ERROR",
        "WRONG_ROLE": "FACT_ROLE_ERROR",
        "WRONG_PERIOD": "FACT_PERIOD_ERROR",
        "WRONG_CHANGE_UNIT": "FACT_UNIT_ERROR",
        "WRONG_CURRENCY": "FACT_UNIT_ERROR",
        "WRONG_SCALE": "FACT_UNIT_ERROR",
    }
    return mapping.get(error_type, "OTHER_ERROR")


def _error_payload(
    system: str,
    example: BenchmarkExample,
    category: str,
    gold: EventType,
    predicted: EventType,
    *,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "system": system,
        "news_id": str(example.annotation.news_id),
        "ticker": example.ticker,
        "source": example.source_code,
        "category": category,
        "gold_primary_event": gold.value,
        "predicted_primary_event": predicted.value,
        "details": details or {},
    }


def _runtime_metrics(predictions: list[BenchmarkPrediction]) -> dict[str, object]:
    latencies = [
        int(str(item.runtime["latency_ms"]))
        for item in predictions
        if isinstance(item.runtime.get("latency_ms"), int)
    ]
    return {
        "mean_latency_ms": (round(sum(latencies) / len(latencies), 3) if latencies else None),
        "input_tokens": _sum_optional(predictions, "input_tokens"),
        "output_tokens": _sum_optional(predictions, "output_tokens"),
        "total_tokens": _sum_optional(predictions, "total_tokens"),
    }


def _sum_optional(predictions: list[BenchmarkPrediction], key: str) -> int | None:
    values = [item.runtime.get(key) for item in predictions]
    integers = [item for item in values if isinstance(item, int) and not isinstance(item, bool)]
    return sum(integers) if integers else None


def _read_jsonl(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkValidationError(f"invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"JSONL line {line_number} is not an object")
        rows.append({str(key): item for key, item in cast("dict[object, object]", value).items()})
    return rows


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return text.encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise BenchmarkValidationError(f"required frozen file missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _most_common(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts, key=lambda item: (-counts[item], item))[0]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
