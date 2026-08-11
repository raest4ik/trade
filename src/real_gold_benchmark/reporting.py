from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from src.real_gold_benchmark.domain import FrozenDataset, PredictionEvaluation


def write_benchmark_report(
    path: Path,
    *,
    frozen: FrozenDataset,
    rules: PredictionEvaluation,
    qwen: PredictionEvaluation,
    comparison: dict[str, object],
    taxonomy: dict[str, object],
) -> None:
    rules_events = _object(rules.metrics["events"])
    qwen_events = _object(qwen.metrics["events"])
    rules_facts = _object(rules.metrics["facts"])
    qwen_facts = _object(qwen.metrics["facts"])
    four_way = _object(comparison["four_way"])
    oracle = _object(comparison["ORACLE_UPPER_BOUND"])
    lines = [
        "# Real Gold Benchmark v2",
        "",
        "## Dataset",
        "",
        (
            "Batch 003 is separate from frozen Batch 001. It contains REAL issuer-owned news "
            "records with EXACT publication timestamps, but its human labels were reviewed only "
            "from the stored excerpts. It is EXCERPT_ONLY gold, not full-text human gold."
        ),
        "",
        f"- dataset: `{frozen.manifest['name']}`",
        f"- dataset SHA-256: `{frozen.dataset_sha256}`",
        f"- human-review source SHA-256: `{frozen.source_file_sha256}`",
        f"- frozen Batch 001 SHA-256: `{frozen.manifest['batch_001_sha256']}` (unchanged)",
        f"- records: {frozen.manifest['records']}",
        f"- review basis: `{frozen.manifest['review_basis']}`",
        f"- tickers: `{_compact(frozen.manifest['tickers'])}`",
        f"- sources: `{_compact(frozen.manifest['sources'])}`",
        f"- months: `{_compact(frozen.manifest['month_distribution'])}`",
        f"- event distribution: `{_compact(frozen.manifest['event_distribution'])}`",
        "- warnings: `SMALL_SAMPLE`, `CLASS_IMBALANCE`, `LOW_SOURCE_DIVERSITY`, "
        "`LOW_TICKER_DIVERSITY`",
        "",
        (
            "The 26 rows are too small and too concentrated by class, ticker, source, and time "
            "for broad conclusions. No future market returns, prices, volume, or reaction labels "
            "are analyzer inputs."
        ),
        "",
        "## Rules v2",
        "",
        *_runtime_lines(rules.metrics),
        *_metric_lines(rules_events, rules_facts),
        "",
        "## Qwen 3.5 9B",
        "",
        *_runtime_lines(qwen.metrics),
        *_metric_lines(qwen_events, qwen_facts),
        "",
        (
            "Any comparison describes performance only on this 26-example real "
            "excerpt-reviewed benchmark. It is not a claim of general superiority."
        ),
        "",
        "## Four-way primary outcome",
        "",
        *[
            f"- {name}: {_object(value)['count']} ({_object(value)['percentage']:.6f})"
            for name, value in four_way.items()
        ],
        (
            "- ORACLE_UPPER_BOUND primary accuracy: "
            f"{_number(oracle['primary_accuracy']):.6f} (diagnostic only)"
        ),
        (
            "- Rules/Qwen primary disagreement count: "
            f"{comparison['rules_vs_qwen_disagreement_count']}"
        ),
        "",
        (
            "ORACLE_UPPER_BOUND is not a hybrid, fallback, ensemble, reconciliation, or emitted "
            "prediction. It only counts records where at least one frozen system was correct."
        ),
        "",
        "## Error taxonomy",
        "",
        "```json",
        json.dumps(taxonomy, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "The taxonomy is research-only. No error was used to change Rules or Qwen in this PR.",
        "",
        "## Limitations and test policy",
        "",
        "- Human review is excerpt-only and may miss facts available on linked full articles.",
        "- OTHER dominates the benchmark, so micro metrics can hide minority-class failures.",
        "- ROSN and YDEX are the only represented tickers and issuer sources.",
        "- Evidence accuracy is reported separately from semantic strict fact scoring.",
        "- The three GUIDANCE percentage targets preserve PERCENT units and are not converted to "
        "percentage points.",
        "- Batch 003 is OBSERVED after this evaluation and must not be used to tune v2 or Qwen.",
        "- Future extractor changes require a fresh reviewed Batch 004 or another holdout.",
        "- No prompt, schema, model, rules, reaction, or ML feature semantics were changed.",
        "- No hybrid, predictive model training, backtest, or trading signal was created.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_lines(events: dict[str, object], facts: dict[str, object]) -> list[str]:
    micro = _object(events["micro"])
    per_class = _object(events["per_class"])
    other = _object(per_class.get("OTHER", {}))
    financial = _object(per_class.get("FINANCIAL_RESULTS", {}))
    guidance = _object(per_class.get("GUIDANCE", {}))
    value = _object(facts["value"])
    metric = _object(facts["metric"])
    semantic = _object(facts["semantic_strict"])
    evidence_note = ""
    if facts.get("matched_pair_count") == 0:
        evidence_note = " (vacuous: no matched fact pairs)"
    return [
        (
            "- event micro precision/recall/F1: "
            f"{micro['precision']}/{micro['recall']}/{micro['f1']}"
        ),
        f"- event macro F1: {events['macro_f1']}",
        f"- primary accuracy: {events['primary_accuracy']}",
        f"- OTHER precision/recall/F1: {_prf(other)}",
        f"- FINANCIAL_RESULTS precision/recall/F1: {_prf(financial)}",
        f"- GUIDANCE precision/recall/F1: {_prf(guidance)}",
        f"- fact value precision/recall/F1: {_prf(value)}",
        f"- fact metric precision/recall/F1: {_prf(metric)}",
        f"- fact semantic strict precision/recall/F1: {_prf(semantic)}",
        f"- fact evidence span accuracy: {facts['evidence_span_accuracy']}{evidence_note}",
        f"- fact field accuracies: `{_compact(facts['field_accuracy'])}`",
        f"- primary confusion: `{json.dumps(events['confusion_matrix'], sort_keys=True)}`",
    ]


def _runtime_lines(metrics: dict[str, object]) -> list[str]:
    runtime = _object(metrics["runtime"])
    return [
        f"- successful/failed: {metrics['successful']}/{metrics['failed']}",
        f"- mean latency ms: {runtime.get('mean_latency_ms')}",
        (
            "- input/output/total tokens: "
            f"{runtime.get('input_tokens')}/{runtime.get('output_tokens')}/"
            f"{runtime.get('total_tokens')}"
        ),
    ]


def _prf(value: dict[str, object]) -> str:
    return f"{value.get('precision', 0.0)}/{value.get('recall', 0.0)}/{value.get('f1', 0.0)}"


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _number(value: object) -> float:
    return float(str(value))
