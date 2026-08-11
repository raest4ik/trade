from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import cast

from src.ai_events.domain.exceptions import AIEventError
from src.real_gold_benchmark.domain import (
    BenchmarkValidationError,
    PredictionEvaluation,
    canonicalize_human_review,
    compare_prediction_sets,
    freeze_canonical_dataset,
    load_frozen_dataset,
    normalize_taxonomy_errors,
    taxonomy_summary,
    write_json,
    write_jsonl,
)
from src.real_gold_benchmark.reporting import write_benchmark_report
from src.real_gold_benchmark.runner import (
    QWEN_MODEL,
    evaluate,
    qwen_manifest,
    rules_manifest,
    run_qwen,
    run_rules,
    write_prediction_artifacts,
)
from src.shared.config.settings import get_settings


async def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if args.refresh_derived_only:
        return _refresh_derived(output, Path(args.report))
    try:
        canonical = canonicalize_human_review(Path(args.human_review))
        frozen = freeze_canonical_dataset(
            canonical,
            output_directory=output / "gold",
            old_batch_001_path=Path(args.batch_001_gold),
            git_sha=_git_sha(),
        )
        if args.freeze_only:
            print(
                json.dumps(
                    {
                        "status": "FROZEN",
                        "dataset_name": frozen.manifest["name"],
                        "dataset_sha256": frozen.dataset_sha256,
                        "records": len(frozen.examples),
                        "source_file_sha256": frozen.source_file_sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0

        rules_predictions = run_rules(frozen)
        rules_evaluation = evaluate(frozen, rules_predictions, system_name="rules-v2")
        write_prediction_artifacts(
            output / "rules-v2",
            frozen=frozen,
            predictions=rules_predictions,
            evaluation=rules_evaluation,
            manifest=rules_manifest(),
        )

        qwen_directory = output / "qwen3.5-9b"
        qwen_predictions, qwen_config = await run_qwen(
            frozen,
            settings=get_settings(),
            output_directory=qwen_directory,
            frozen_config_path=qwen_directory / "frozen-config.json",
        )
        qwen_evaluation = evaluate(frozen, qwen_predictions, system_name=QWEN_MODEL)
        write_prediction_artifacts(
            qwen_directory,
            frozen=frozen,
            predictions=qwen_predictions,
            evaluation=qwen_evaluation,
            manifest=qwen_manifest(qwen_config, qwen_predictions),
        )

        comparison_rows, comparison = compare_prediction_sets(
            frozen,
            rules_predictions,
            qwen_predictions,
        )
        write_jsonl(output / "comparison.jsonl", comparison_rows)
        write_json(output / "comparison-summary.json", comparison)
        taxonomy_rows, taxonomy = taxonomy_summary(
            rules_evaluation.errors,
            qwen_evaluation.errors,
        )
        write_jsonl(output / "error-taxonomy.jsonl", taxonomy_rows)
        write_json(output / "error-taxonomy.json", taxonomy)
        write_benchmark_report(
            Path(args.report),
            frozen=frozen,
            rules=rules_evaluation,
            qwen=qwen_evaluation,
            comparison=comparison,
            taxonomy=taxonomy,
        )
    except (BenchmarkValidationError, AIEventError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    summary = {
        "status": "SUCCEEDED",
        "dataset_name": frozen.manifest["name"],
        "dataset_sha256": frozen.dataset_sha256,
        "records": len(frozen.examples),
        "rules_metrics": rules_evaluation.metrics,
        "qwen_metrics": qwen_evaluation.metrics,
        "comparison": comparison,
        "taxonomy": taxonomy,
        "output": str(output),
        "report": str(args.report),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if qwen_evaluation.metrics["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and evaluate the excerpt-only REAL Batch 003 benchmark."
    )
    parser.add_argument(
        "--human-review",
        default="artifacts/corpus-quality-v1/batch-003-human-review-v1.jsonl",
    )
    parser.add_argument(
        "--batch-001-gold",
        default="artifacts/seed/batch-001-gold-v1-reviewed-only.jsonl",
    )
    parser.add_argument("--output", default="artifacts/real-gold-benchmark-v2")
    parser.add_argument("--report", default="docs/real-gold-benchmark-v2.md")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--refresh-derived-only", action="store_true")
    return parser


def _refresh_derived(output: Path, report: Path) -> int:
    frozen = load_frozen_dataset(
        output / "gold" / "dataset.jsonl", output / "gold" / "manifest.json"
    )
    rules = _existing_evaluation(output / "rules-v2")
    qwen = _existing_evaluation(output / "qwen3.5-9b")
    rules_errors = normalize_taxonomy_errors(rules.errors)
    qwen_errors = normalize_taxonomy_errors(qwen.errors)
    rules = PredictionEvaluation(metrics=rules.metrics, errors=rules_errors)
    qwen = PredictionEvaluation(metrics=qwen.metrics, errors=qwen_errors)
    write_jsonl(output / "rules-v2" / "errors.jsonl", [dict(item) for item in rules_errors])
    write_jsonl(output / "qwen3.5-9b" / "errors.jsonl", [dict(item) for item in qwen_errors])
    comparison = _read_object(output / "comparison-summary.json")
    taxonomy_rows, taxonomy = taxonomy_summary(rules.errors, qwen.errors)
    write_jsonl(output / "error-taxonomy.jsonl", taxonomy_rows)
    write_json(output / "error-taxonomy.json", taxonomy)
    write_benchmark_report(
        report,
        frozen=frozen,
        rules=rules,
        qwen=qwen,
        comparison=comparison,
        taxonomy=taxonomy,
    )
    print(
        json.dumps(
            {"status": "DERIVED_ARTIFACTS_REFRESHED", "model_inference_performed": False},
            sort_keys=True,
        )
    )
    return 0


def _existing_evaluation(directory: Path) -> PredictionEvaluation:
    metrics = _read_object(directory / "metrics.json")
    errors = tuple(_read_jsonl(directory / "errors.jsonl"))
    return PredictionEvaluation(metrics=metrics, errors=errors)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"expected JSON object: {path}")
    items = cast("dict[object, object]", value)
    return {str(key): item for key, item in items.items()}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"expected JSONL objects: {path}")
        items = cast("dict[object, object]", value)
        rows.append({str(key): item for key, item in items.items()})
    return rows


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
