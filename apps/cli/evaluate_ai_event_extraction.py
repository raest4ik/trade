from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from src.ai_events.application.evaluation import to_evaluation_inputs
from src.ai_events.application.frozen_test import FrozenTestGuardError, validate_test_access
from src.ai_events.application.serialization import analysis_result_to_json, failure_to_json
from src.ai_events.application.use_cases import (
    AIEventAnalysisResult,
    AIItemFailure,
    AnalyzeAIEventCommand,
    sanitize_failure,
)
from src.ai_events.domain.prompt import (
    ANALYSIS_VERSION,
    FACT_EXTRACTOR_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    prompt_hash,
    schema_hash,
)
from src.ai_events.infrastructure.factory import (
    DEFAULT_AI_CACHE_DIRECTORY,
    create_ai_event_analyzer,
)
from src.evaluation.domain.enums import DatasetSplit
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.evaluation.infrastructure.repositories import (
    EvaluationExampleWithNews,
    SqlAlchemyEvaluationRepository,
)
from src.shared.config.settings import Settings, get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    split = DatasetSplit(args.split)
    if args.limit is not None and split != DatasetSplit.TRAIN:
        print("error=--limit is allowed only for TRAIN")
        return 2
    try:
        analyzer = create_ai_event_analyzer(
            settings,
            cache_directory=Path(args.cache_dir),
        )
    except Exception as exc:
        failure = sanitize_failure(exc)
        print(json.dumps(failure_to_json(failure), sort_keys=True))
        return 2

    dataset_id = UUID(args.dataset_id)
    engine = create_engine(settings.database_url)
    try:
        async with create_session_factory(engine)() as session:
            repository = SqlAlchemyEvaluationRepository(session)
            dataset = await repository.get_dataset(dataset_id)
            if dataset is None:
                print("error=evaluation dataset not found")
                return 2
            expected_frozen = _frozen_config(settings, dataset_id, dataset.source_file_hash)
            validate_test_access(
                split=split,
                allow_frozen_test=args.allow_frozen_test,
                frozen_config_path=None if args.frozen_config is None else Path(args.frozen_config),
                expected_config=expected_frozen,
            )
            rows = await repository.list_examples_with_news(dataset_id=dataset_id, split=split)
    except FrozenTestGuardError as exc:
        print(f"error={exc}")
        return 2
    finally:
        await engine.dispose()

    if args.limit is not None:
        rows = rows[: args.limit]
    completed = await asyncio.gather(
        *[
            _analyze_row(
                analyzer,
                row,
                settings=settings,
                force_refresh=args.force_refresh,
            )
            for row in rows
        ]
    )
    successes = [(row, result) for row, result, _ in completed if result is not None]
    failures = [failure for _, _, failure in completed if failure is not None]
    output_dir = Path(args.output_dir or f"artifacts/seed/ai-event-v0/{split.value.lower()}")
    metrics = _evaluate(
        dataset_id=dataset_id,
        dataset_name=dataset.name,
        dataset_hash=dataset.source_file_hash,
        split=split,
        requested_count=len(rows),
        successes=successes,
        failures=failures,
        baseline_path=None if args.baseline_metrics is None else Path(args.baseline_metrics),
    )
    _write_artifacts(
        output_dir=output_dir,
        successes=successes,
        failures=failures,
        metrics=metrics,
        settings=settings,
        dataset_id=dataset_id,
        dataset_hash=dataset.source_file_hash,
        split=split,
        cache_directory=Path(args.cache_dir),
    )
    if args.freeze_config_output is not None:
        if split != DatasetSplit.VALIDATION:
            print("error=frozen config can be created only from VALIDATION")
            return 2
        if failures:
            print("error=cannot freeze a validation run with item failures")
            return 1
        _write_json(Path(args.freeze_config_output), expected_frozen)
    print(
        f"split={split.value} requested={len(rows)} succeeded={len(successes)} "
        f"failed={len(failures)} output_dir={output_dir}"
    )
    return 0 if not failures else 1


async def _analyze_row(
    analyzer: object,
    row: EvaluationExampleWithNews,
    *,
    settings: Settings,
    force_refresh: bool,
) -> tuple[EvaluationExampleWithNews, AIEventAnalysisResult | None, AIItemFailure | None]:
    from src.ai_events.application.use_cases import AnalyzeAIEvent

    typed_analyzer = cast("AnalyzeAIEvent", analyzer)
    record_id = str(row.example.id)
    news_id = row.news.id
    try:
        result = await typed_analyzer.execute(
            AnalyzeAIEventCommand(
                raw_content=row.news.raw_content,
                requested_model=settings.openai_model,
                reasoning_effort=settings.ai_reasoning_effort,
                max_output_tokens=settings.ai_max_output_tokens,
                news_id=news_id,
                record_id=record_id,
                force_refresh=force_refresh,
            )
        )
    except Exception as exc:
        return row, None, sanitize_failure(exc, record_id=record_id, news_id=news_id)
    return row, result, None


def _evaluate(
    *,
    dataset_id: UUID,
    dataset_name: str,
    dataset_hash: str,
    split: DatasetSplit,
    requested_count: int,
    successes: list[tuple[EvaluationExampleWithNews, AIEventAnalysisResult]],
    failures: list[AIItemFailure],
    baseline_path: Path | None,
) -> dict[str, object]:
    event_inputs: list[EventEvaluationInput] = []
    fact_inputs: list[FactEvaluationInput] = []
    record_ids: list[str] = []
    for row, prediction in successes:
        event_input, fact_input = to_evaluation_inputs(
            gold_events=[item.to_entity() for item in row.example.gold_events],
            gold_facts=[item.to_entity() for item in row.example.gold_financial_facts],
            prediction=prediction,
        )
        event_inputs.append(event_input)
        fact_inputs.append(fact_input)
        record_ids.append(str(row.example.id))
    event_result = evaluate_event_predictions(event_inputs)
    fact_result = evaluate_fact_predictions(fact_inputs)
    errors = _tag_errors(event_result.errors + fact_result.errors, record_ids)
    metrics: dict[str, object] = {
        "dataset_id": str(dataset_id),
        "dataset_name": dataset_name,
        "dataset_source_hash": dataset_hash,
        "split": split.value,
        "analysis_version": ANALYSIS_VERSION,
        "extractor_version": FACT_EXTRACTOR_VERSION,
        "requested_example_count": requested_count,
        "evaluated_example_count": len(successes),
        "failed_example_count": len(failures),
        "events": event_result.metrics,
        "facts": fact_result.metrics,
        "errors": errors,
    }
    if baseline_path is not None and baseline_path.is_file():
        baseline = cast(
            "dict[str, object]",
            json.loads(baseline_path.read_text(encoding="utf-8")),
        )
        metrics["deterministic_baseline"] = baseline
        metrics["comparison"] = _comparison(metrics, baseline)
    return metrics


def _tag_errors(errors: list[dict[str, object]], record_ids: list[str]) -> list[dict[str, object]]:
    tagged: list[dict[str, object]] = []
    for error in errors:
        item = dict(error)
        index = item.get("example_index")
        if isinstance(index, int) and 0 <= index < len(record_ids):
            item["record_id"] = record_ids[index]
        tagged.append(item)
    return tagged


def _write_artifacts(
    *,
    output_dir: Path,
    successes: list[tuple[EvaluationExampleWithNews, AIEventAnalysisResult]],
    failures: list[AIItemFailure],
    metrics: dict[str, object],
    settings: Settings,
    dataset_id: UUID,
    dataset_hash: str,
    split: DatasetSplit,
    cache_directory: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    actual_models = Counter[str]()
    input_tokens = 0
    output_tokens = 0
    for row, result in successes:
        payload = analysis_result_to_json(result)
        payload["record_id"] = str(row.example.id)
        payload["news_id"] = str(row.news.id)
        predictions.append(payload)
        actual_models[result.metadata.actual_model] += 1
        input_tokens += result.metadata.input_tokens or 0
        output_tokens += result.metadata.output_tokens or 0
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    public_metrics = {key: value for key, value in metrics.items() if key != "errors"}
    _write_json(output_dir / "metrics.json", public_metrics)
    errors = cast("list[dict[str, object]]", metrics["errors"])
    errors.extend({"type": "AI_ITEM_FAILURE", **failure_to_json(item)} for item in failures)
    _write_jsonl(output_dir / "errors.jsonl", errors)
    manifest = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": _git_commit_sha(),
        "dataset_id": str(dataset_id),
        "dataset_source_hash": dataset_hash,
        "split": split.value,
        "requested_model": settings.openai_model,
        "actual_models": dict(sorted(actual_models.items())),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "analyzer_version": ANALYSIS_VERSION,
        "fact_extractor_version": FACT_EXTRACTOR_VERSION,
        "reasoning_effort": settings.ai_reasoning_effort,
        "max_output_tokens": settings.ai_max_output_tokens,
        "request_timeout_seconds": settings.ai_request_timeout_seconds,
        "max_retries": settings.ai_max_retries,
        "max_concurrency": settings.ai_max_concurrency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_directory": str(cache_directory),
    }
    _write_json(output_dir / "run-manifest.json", manifest)
    event_metrics = cast("dict[str, object]", public_metrics["events"])
    fact_metrics = cast("dict[str, object]", public_metrics["facts"])
    event_micro = cast("dict[str, object]", event_metrics["micro"])
    fact_semantic = cast("dict[str, object]", fact_metrics["semantic_strict"])
    summary = [
        "# AI Event Extraction v0",
        "",
        f"- split: {split.value}",
        f"- requested: {public_metrics['requested_example_count']}",
        f"- evaluated: {public_metrics['evaluated_example_count']}",
        f"- failures: {public_metrics['failed_example_count']}",
        f"- event micro F1: {event_micro['f1']}",
        f"- fact semantic strict F1: {fact_semantic['f1']}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _frozen_config(
    settings: Settings,
    dataset_id: UUID,
    dataset_hash: str,
) -> dict[str, object]:
    return {
        "dataset_id": str(dataset_id),
        "dataset_source_hash": dataset_hash,
        "requested_model": settings.openai_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "analyzer_version": ANALYSIS_VERSION,
        "fact_extractor_version": FACT_EXTRACTOR_VERSION,
        "reasoning_effort": settings.ai_reasoning_effort,
        "max_output_tokens": settings.ai_max_output_tokens,
    }


def _comparison(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    paths = (
        "events.micro.f1",
        "events.primary_accuracy",
        "facts.strict.f1",
        "facts.semantic_strict.f1",
        "facts.value.f1",
        "facts.metric.f1",
        "facts.evidence_span_accuracy",
    )
    comparison: dict[str, object] = {}
    for path in paths:
        current_value = _nested_number(current, path)
        baseline_value = _nested_number(baseline, path)
        if current_value is not None and baseline_value is not None:
            comparison[path] = {
                "ai": current_value,
                "deterministic": baseline_value,
                "delta": current_value - baseline_value,
            }
    return comparison


def _nested_number(payload: dict[str, object], path: str) -> float | None:
    value: object = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = cast("Mapping[str, object]", value).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in payloads),
        encoding="utf-8",
    )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AI event extraction.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--split",
        default=DatasetSplit.VALIDATION.value,
        choices=[item.value for item in DatasetSplit if item != DatasetSplit.UNASSIGNED],
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default=str(DEFAULT_AI_CACHE_DIRECTORY))
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--allow-frozen-test", action="store_true")
    parser.add_argument("--frozen-config")
    parser.add_argument("--freeze-config-output")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
