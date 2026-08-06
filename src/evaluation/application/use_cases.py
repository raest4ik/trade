from __future__ import annotations

import asyncio
import json
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

from src.evaluation.application.exceptions import (
    AnnotationDatasetValidationError,
    EvaluationDatasetNotFoundError,
    EvaluationThresholdError,
)
from src.evaluation.domain.entities import EvaluationRun, GoldEvent
from src.evaluation.domain.enums import DatasetSplit, EvaluationRunStatus
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.evaluation.domain.serialization import annotation_from_json
from src.evaluation.domain.validation import ValidationIssue, validate_jsonl_payloads
from src.evaluation.infrastructure.repositories import (
    ImportDatasetResult,
    SqlAlchemyEvaluationRepository,
    dataset_from_examples,
)
from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.events.domain.enums import EventType


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    run: EvaluationRun
    report_directory: Path
    errors: list[dict[str, object]]


async def import_annotation_dataset(
    *,
    repository: SqlAlchemyEvaluationRepository,
    path: Path,
    name: str,
    description: str | None = None,
    allow_missing_news: bool = False,
) -> ImportDatasetResult:
    lines = await asyncio.to_thread(_read_lines, path)
    examples = [annotation_from_json(json.loads(line)) for line in lines if line.strip()]
    content_by_id, hash_by_id = await repository.raw_news_maps(
        [example.news_id for example in examples]
    )
    validation = validate_jsonl_payloads(
        lines,
        raw_content_by_news_id=content_by_id,
        raw_hash_by_news_id=hash_by_id,
        allow_missing_news=allow_missing_news,
    )
    if not validation.ok:
        raise AnnotationDatasetValidationError(_issue_text(validation.errors))
    source_hash = _file_hash(path)
    dataset = dataset_from_examples(
        name=name,
        source_file_hash=source_hash,
        examples=examples,
        description=description,
    )
    return await repository.import_dataset(dataset=dataset, examples=examples)


async def run_evaluation(
    *,
    repository: SqlAlchemyEvaluationRepository,
    dataset_id: UUID,
    split: DatasetSplit,
    thresholds_path: Path,
    output_dir: Path,
    fail_below_thresholds: bool = False,
) -> EvaluationRunResult:
    dataset = await repository.get_dataset(dataset_id)
    if dataset is None:
        raise EvaluationDatasetNotFoundError("evaluation dataset not found")
    thresholds = _load_thresholds(thresholds_path)
    started = await repository.save_run(
        EvaluationRun.running(
            dataset_id=dataset_id,
            split=split,
            git_commit_sha=_git_commit_sha(),
            config_json={"thresholds": thresholds, "thresholds_path": str(thresholds_path)},
        )
    )
    rows = await repository.list_examples_with_news(dataset_id=dataset_id, split=split)
    analyzer = EventAnalyzer()
    event_inputs: list[EventEvaluationInput] = []
    fact_inputs: list[FactEvaluationInput] = []
    for row in rows:
        analysis = analyzer.analyze(news_id=row.news.id, raw_content=row.news.raw_content)
        gold_events = [record.to_entity() for record in row.example.gold_events]
        gold_facts = [record.to_entity() for record in row.example.gold_financial_facts]
        event_inputs.append(
            EventEvaluationInput(
                gold_events=gold_events,
                predicted_events=analysis.events,
                gold_primary_event_type=_primary_gold_event_type(gold_events),
                predicted_primary_event_type=analysis.primary_event_type,
                prediction_status=analysis.status.value,
            )
        )
        fact_inputs.append(
            FactEvaluationInput(
                gold_facts=gold_facts,
                predicted_facts=analysis.financial_facts,
            )
        )
    event_result = evaluate_event_predictions(event_inputs)
    fact_result = evaluate_fact_predictions(fact_inputs)
    errors = event_result.errors + fact_result.errors
    metrics: dict[str, object] = {
        "dataset_id": str(dataset_id),
        "dataset_name": dataset.name,
        "split": split.value,
        "analysis_version": EVENT_ANALYSIS_VERSION,
        "extractor_version": FINANCIAL_FACTS_VERSION,
        "events": event_result.metrics,
        "facts": fact_result.metrics,
        "thresholds": thresholds,
    }
    report_directory = output_dir / str(started.id)
    _write_reports(report_directory, metrics, errors)
    status = EvaluationRunStatus.SUCCEEDED
    failed_thresholds = _failed_thresholds(metrics, thresholds)
    if failed_thresholds:
        status = EvaluationRunStatus.FAILED
        metrics["failed_thresholds"] = failed_thresholds
    finished = await repository.finish_run(
        run_id=started.id,
        status=status,
        example_count=len(rows),
        metrics_json=metrics,
        error_count=len(errors),
    )
    if failed_thresholds and fail_below_thresholds:
        raise EvaluationThresholdError(", ".join(failed_thresholds))
    return EvaluationRunResult(run=finished, report_directory=report_directory, errors=errors)


def _primary_gold_event_type(gold_events: list[GoldEvent]) -> EventType | None:
    for event in gold_events:
        if event.is_primary:
            return event.event_type
    for event in gold_events:
        return event.event_type
    return None


def _load_thresholds(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return cast("dict[str, object]", tomllib.loads(path.read_text(encoding="utf-8")))


def _failed_thresholds(
    metrics: Mapping[str, object], thresholds: Mapping[str, object]
) -> list[str]:
    failed: list[str] = []
    event_thresholds = _dict(thresholds.get("event"))
    fact_thresholds = _dict(thresholds.get("fact"))
    event_metrics = _dict(metrics.get("events"))
    fact_metrics = _dict(metrics.get("facts"))
    event_micro = _dict(event_metrics.get("micro"))
    fact_strict = _dict(fact_metrics.get("strict"))
    fact_value = _dict(fact_metrics.get("value"))
    fact_metric = _dict(fact_metrics.get("metric"))
    checks = (
        ("event.micro_f1", event_micro.get("f1"), event_thresholds.get("micro_f1")),
        (
            "event.primary_accuracy",
            event_metrics.get("primary_accuracy"),
            event_thresholds.get("primary_accuracy"),
        ),
        ("fact.strict_f1", fact_strict.get("f1"), fact_thresholds.get("strict_f1")),
        ("fact.value_f1", fact_value.get("f1"), fact_thresholds.get("value_f1")),
        ("fact.metric_f1", fact_metric.get("f1"), fact_thresholds.get("metric_f1")),
    )
    for name, actual, expected in checks:
        if expected is None or actual is None:
            continue
        if float(cast("float | int", actual)) < float(cast("float | int", expected)):
            failed.append(f"{name} {actual} < {expected}")
    return failed


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _write_reports(
    report_directory: Path,
    metrics: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Event Extraction Evaluation",
        "",
        f"- dataset: {metrics['dataset_name']}",
        f"- split: {metrics['split']}",
        f"- examples: {_dict(metrics['events']).get('example_count', 0)}",
        f"- errors: {len(errors)}",
    ]
    (report_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_directory / "errors.jsonl").write_text(
        "".join(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n" for error in errors),
        encoding="utf-8",
    )


def _issue_text(issues: list[ValidationIssue]) -> str:
    return "; ".join(f"line {issue.line_number} {issue.code}: {issue.message}" for issue in issues)


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
