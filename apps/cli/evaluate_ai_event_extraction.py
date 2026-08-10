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
from src.ai_events.domain.enums import AIProvider
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
    AIProviderConfig,
    create_ai_event_analyzer,
    resolve_ai_provider_config,
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
        provider = resolve_ai_provider_config(settings, args.provider)
        analyzer = create_ai_event_analyzer(
            settings,
            cache_directory=Path(args.cache_dir),
            provider_override=args.provider,
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
            expected_frozen = _frozen_config(
                settings,
                provider,
                dataset_id,
                dataset.source_file_hash,
            )
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
                provider=provider,
                force_refresh=args.force_refresh,
            )
            for row in rows
        ]
    )
    successes = [(row, result) for row, result, _ in completed if result is not None]
    failures = [failure for _, _, failure in completed if failure is not None]
    output_dir = Path(
        args.output_dir
        or f"artifacts/seed/ai-event-v0/{provider.artifact_slug}/{split.value.lower()}"
    )
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
        provider=provider,
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
        frozen = {
            **expected_frozen,
            "validation_metrics": _metric_snapshot(metrics),
            "frozen_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        _write_json(Path(args.freeze_config_output), frozen)
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
    provider: AIProviderConfig,
    force_refresh: bool,
) -> tuple[EvaluationExampleWithNews, AIEventAnalysisResult | None, AIItemFailure | None]:
    from src.ai_events.application.use_cases import AnalyzeAIEvent

    typed_analyzer = cast("AnalyzeAIEvent", analyzer)
    record_id = str(row.example.id)
    news_id = row.news.id
    try:
        result = await typed_analyzer.execute(
            AnalyzeAIEventCommand(
                provider=provider.provider.value,
                raw_content=row.news.raw_content,
                requested_model=provider.requested_model,
                reasoning_effort=provider.reasoning_effort,
                max_output_tokens=settings.ai_max_output_tokens,
                think=provider.think,
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
        "runtime": _runtime_metrics(successes, failures),
        "errors": errors,
    }
    baseline = _default_baseline(split)
    if baseline_path is not None and baseline_path.is_file():
        baseline = cast(
            "dict[str, object]",
            json.loads(baseline_path.read_text(encoding="utf-8")),
        )
    if baseline is not None:
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
    provider: AIProviderConfig,
    dataset_id: UUID,
    dataset_hash: str,
    split: DatasetSplit,
    cache_directory: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    actual_models = Counter[str]()
    for row, result in successes:
        payload = analysis_result_to_json(result)
        payload["record_id"] = str(row.example.id)
        payload["news_id"] = str(row.news.id)
        predictions.append(payload)
        actual_models[result.metadata.actual_model] += 1
    if split != DatasetSplit.TEST:
        _write_jsonl(output_dir / "predictions.jsonl", predictions)
    public_metrics = {key: value for key, value in metrics.items() if key != "errors"}
    _write_json(output_dir / "metrics.json", public_metrics)
    if split != DatasetSplit.TEST:
        errors = cast("list[dict[str, object]]", metrics["errors"])
        errors.extend({"type": "AI_ITEM_FAILURE", **failure_to_json(item)} for item in failures)
        _write_jsonl(output_dir / "errors.jsonl", errors)
    runtime = cast("dict[str, object]", public_metrics["runtime"])
    manifest = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": _git_commit_sha(),
        "dataset_id": str(dataset_id),
        "dataset_source_hash": dataset_hash,
        "split": split.value,
        "provider": provider.provider.value,
        "requested_model": provider.requested_model,
        "actual_model": next(iter(actual_models)) if len(actual_models) == 1 else None,
        "actual_response_models": dict(sorted(actual_models.items())),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_hash(),
        "analyzer_version": ANALYSIS_VERSION,
        "fact_version": FACT_EXTRACTOR_VERSION,
        "reasoning_effort": provider.reasoning_effort,
        "think": provider.think,
        "max_output_tokens": settings.ai_max_output_tokens,
        "request_timeout_seconds": settings.ai_request_timeout_seconds,
        "max_retries": settings.ai_max_retries,
        "max_concurrency": settings.ai_max_concurrency,
        "input_tokens": runtime["input_tokens"],
        "output_tokens": runtime["output_tokens"],
        "total_tokens": runtime["total_tokens"],
        "api_calls": runtime["api_calls"],
        "cache_hits": runtime["cache_hits"],
        "failed_calls": runtime["failed_calls"],
        "mean_latency_ms": runtime["mean_latency_ms"],
        "p50_latency_ms": runtime["p50_latency_ms"],
        "p95_latency_ms": runtime["p95_latency_ms"],
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
    _write_comparison_markdown(
        public_metrics,
        split,
        provider=provider,
        target=output_dir.parent / f"comparison-{split.value.lower()}.md",
    )


def _frozen_config(
    settings: Settings,
    provider: AIProviderConfig,
    dataset_id: UUID,
    dataset_hash: str,
) -> dict[str, object]:
    return {
        "dataset_id": str(dataset_id),
        "gold_dataset_sha256": dataset_hash,
        "provider": provider.provider.value,
        "model_requested": provider.requested_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_hash(),
        "analyzer_version": ANALYSIS_VERSION,
        "fact_version": FACT_EXTRACTOR_VERSION,
        "reasoning_effort": provider.reasoning_effort,
        "think": provider.think,
        "max_output_tokens": settings.ai_max_output_tokens,
    }


def _comparison(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    paths = (
        "events.micro.precision",
        "events.micro.recall",
        "events.micro.f1",
        "events.macro_f1",
        "events.primary_accuracy",
        "facts.value.f1",
        "facts.metric.f1",
        "facts.semantic_strict.f1",
        "facts.field_accuracy.period_type",
        "facts.field_accuracy.fact_role",
        "facts.field_accuracy.change_direction",
        "facts.field_accuracy.change_value",
        "facts.field_accuracy.change_unit",
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


def _runtime_metrics(
    successes: list[tuple[EvaluationExampleWithNews, AIEventAnalysisResult]],
    failures: list[AIItemFailure],
) -> dict[str, object]:
    live = [result for _, result in successes if not result.metadata.cached]
    latencies = sorted(result.metadata.latency_ms for result in live)
    input_tokens = sum(result.metadata.input_tokens or 0 for result in live)
    output_tokens = sum(result.metadata.output_tokens or 0 for result in live)
    total_tokens = sum(result.metadata.total_tokens or 0 for result in live)
    return {
        "api_calls": len(live) + len(failures),
        "cache_hits": sum(result.metadata.cached for _, result in successes),
        "failed_calls": len(failures),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "mean_latency_ms": 0.0 if not latencies else sum(latencies) / len(latencies),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _default_baseline(split: DatasetSplit) -> dict[str, object] | None:
    if split == DatasetSplit.VALIDATION:
        event_f1 = primary = fact_value = fact_metric = semantic = 1.0
    elif split == DatasetSplit.TEST:
        event_f1 = 0.909091
        primary = 0.909091
        fact_value = 0.923077
        fact_metric = 0.820513
        semantic = 0.615385
    else:
        return None
    return {
        "events": {
            "micro": {"precision": event_f1, "recall": event_f1, "f1": event_f1},
            "macro_f1": event_f1,
            "primary_accuracy": primary,
        },
        "facts": {
            "value": {"f1": fact_value},
            "metric": {"f1": fact_metric},
            "semantic_strict": {"f1": semantic},
            "field_accuracy": {
                "period_type": 1.0,
                "fact_role": 1.0,
                "change_direction": 1.0,
                "change_value": 1.0,
                "change_unit": 1.0,
            },
            "evidence_span_accuracy": 0.0,
        },
    }


def _metric_snapshot(metrics: dict[str, object]) -> dict[str, object]:
    paths = (
        "events.micro.precision",
        "events.micro.recall",
        "events.micro.f1",
        "events.macro_f1",
        "events.primary_accuracy",
        "facts.value.f1",
        "facts.metric.f1",
        "facts.semantic_strict.f1",
        "facts.evidence_span_accuracy",
    )
    snapshot: dict[str, object] = {}
    for path in paths:
        value = _nested_number(metrics, path)
        if value is not None:
            snapshot[path] = value
    snapshot["runtime"] = metrics["runtime"]
    return snapshot


def _write_comparison_markdown(
    metrics: dict[str, object],
    split: DatasetSplit,
    *,
    provider: AIProviderConfig,
    target: Path,
) -> None:
    if split not in {DatasetSplit.VALIDATION, DatasetSplit.TEST}:
        return
    comparison = cast("dict[str, object]", metrics.get("comparison", {}))
    labels = {
        "events.micro.precision": "event micro precision",
        "events.micro.recall": "event micro recall",
        "events.micro.f1": "event micro F1",
        "events.macro_f1": "event macro F1",
        "events.primary_accuracy": "primary accuracy",
        "facts.value.f1": "fact value F1",
        "facts.metric.f1": "fact metric F1",
        "facts.semantic_strict.f1": "semantic_strict_f1",
        "facts.field_accuracy.period_type": "period accuracy",
        "facts.field_accuracy.fact_role": "role accuracy",
        "facts.field_accuracy.change_direction": "change_direction accuracy",
        "facts.field_accuracy.change_value": "change_value accuracy",
        "facts.field_accuracy.change_unit": "change_unit accuracy",
        "facts.evidence_span_accuracy": "evidence_span_accuracy",
    }
    provider_label = (
        f"Local AI {provider.requested_model}"
        if provider.provider == AIProvider.OLLAMA
        else f"OpenAI {provider.requested_model}"
    )
    lines = [
        f"# Deterministic v2 vs {provider_label} ({split.value})",
        "",
        f"| Metric | Deterministic v2 | {provider.requested_model} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for path, label in labels.items():
        values = cast("dict[str, object]", comparison.get(path, {}))
        if values:
            deterministic = _number(values.get("deterministic"))
            ai_value = _number(values.get("ai"))
            delta = _number(values.get("delta"))
            if deterministic is None or ai_value is None or delta is None:
                continue
            lines.append(f"| {label} | {deterministic:.6f} | {ai_value:.6f} | {delta:+.6f} |")
    runtime = cast("dict[str, object]", metrics["runtime"])
    mean_latency = _number(runtime.get("mean_latency_ms")) or 0.0
    p50_latency = _number(runtime.get("p50_latency_ms")) or 0.0
    p95_latency = _number(runtime.get("p95_latency_ms")) or 0.0
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Metric | Local AI |",
            "| --- | ---: |",
            f"| total calls | {runtime['api_calls']} |",
            f"| cache hits | {runtime['cache_hits']} |",
            f"| failures | {runtime['failed_calls']} |",
            f"| input tokens | {runtime['input_tokens']} |",
            f"| output tokens | {runtime['output_tokens']} |",
            f"| mean latency ms | {mean_latency:.3f} |",
            f"| p50 latency ms | {p50_latency:.3f} |",
            f"| p95 latency ms | {p95_latency:.3f} |",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


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
    parser.add_argument("--provider", choices=[item.value for item in AIProvider])
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
