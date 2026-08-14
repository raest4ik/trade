from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.event_predictive_baseline.data import (
    build_temporal_split,
    exact_event_rows,
    load_comparison_cohort,
    load_targets,
)
from src.event_predictive_baseline.diagnostics import (
    comparison_deltas,
    grouped_diagnostics,
    incremental_value_status,
)
from src.event_predictive_baseline.domain import (
    DATASET_VERSION,
    EXPECTED_DATASET_SHA,
    EXPECTED_FEATURE_SCHEMA_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    FEATURE_FAMILIES,
    MODEL_VERSION,
    PRICE_ADJUSTMENT_STATUS,
    REACTION_FAMILY,
    TEST_STATUS,
    ComparisonCohort,
    EventFeatureRow,
    FrozenModelConfig,
    research_safety_flags,
    sha256_payload,
)
from src.event_predictive_baseline.modeling import (
    evaluate_all_families,
    fit_all_families,
    metrics_from_records,
    serialize_models,
)


def run_event_predictive_baseline(
    dataset_root: Path,
    output_root: Path,
    *,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable event baseline output already exists")
    cohort, exclusions = load_comparison_cohort(dataset_root)
    split = build_temporal_split(cohort)
    assignments = {
        str(item["event_id"]): str(item["split"])
        for item in cast("list[dict[str, Any]]", split["assignments"])
    }
    output_root.mkdir(parents=True, exist_ok=True)
    cohort_manifest = _cohort_manifest(cohort, exclusions)
    _write_exclusive(output_root / "comparison-cohort-manifest.json", cohort_manifest)
    _write_exclusive(output_root / "event-split-manifest.json", split)

    train_rows = cohort.rows_for(assignments, "TRAIN")
    validation_rows = cohort.rows_for(assignments, "VALIDATION")
    development_ids = {row.event_id for row in (*train_rows, *validation_rows)}
    development_targets = load_targets(dataset_root / "targets.jsonl", development_ids)
    train_targets = {row.event_id: development_targets[row.event_id] for row in train_rows}
    validation_targets = {
        row.event_id: development_targets[row.event_id] for row in validation_rows
    }
    config = FrozenModelConfig()
    development_models = fit_all_families(train_rows, train_targets, config)
    validation_metrics, validation_records = evaluate_all_families(
        development_models, validation_rows, validation_targets, train_targets
    )
    validation_diagnostics = grouped_diagnostics(validation_records)
    config_payload = {
        **config.payload(),
        "comparison_cohort_sha": cohort.cohort_sha,
        "split_sha": split["split_sha"],
        "event_feature_schema_sha": cohort.event_schema_sha,
        "market_feature_schema_sha": cohort.market_schema_sha,
    }
    config_payload["locked_config_sha"] = sha256_payload(config_payload)
    _write_exclusive(output_root / "final-model-config.json", config_payload)
    state_path = output_root / "test-evaluation-state.json"
    _write_exclusive(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 0,
            "TEST_STATUS": "BLIND_LOCKED_NOT_EVALUATED",
            "locked_config_sha": config_payload["locked_config_sha"],
        },
    )

    development_rows = (*train_rows, *validation_rows)
    final_models = fit_all_families(development_rows, development_targets, config)
    loio = _loio_diagnostics(development_rows, development_targets, config)
    _replace_json(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 1,
            "TEST_STATUS": "EVALUATION_STARTED_NO_RETRY_ALLOWED",
            "locked_config_sha": config_payload["locked_config_sha"],
        },
    )
    test_rows = cohort.rows_for(assignments, "TEST")
    test_ids = {row.event_id for row in test_rows}
    test_targets = load_targets(dataset_root / "targets.jsonl", test_ids)
    test_metrics, test_records = evaluate_all_families(
        final_models, test_rows, test_targets, development_targets
    )
    test_diagnostics = grouped_diagnostics(test_records)
    deltas = {
        "C_vs_A": comparison_deltas(test_metrics, "C_EVENT_PLUS_MARKET", "A_MARKET_ONLY"),
        "B_vs_A": comparison_deltas(test_metrics, "B_EVENT_ONLY", "A_MARKET_ONLY"),
        "C_vs_B": comparison_deltas(test_metrics, "C_EVENT_PLUS_MARKET", "B_EVENT_ONLY"),
    }
    incremental_status = incremental_value_status(test_metrics, test_diagnostics)
    exact_report = _exact_report(dataset_root)
    model_bytes = serialize_models(final_models)
    model_binary_sha = hashlib.sha256(model_bytes).hexdigest()
    (output_root / "models.pkl").write_bytes(model_bytes)
    _write_json(output_root / "validation-metrics.json", validation_metrics)
    _write_json(output_root / "validation-diagnostics.json", validation_diagnostics)
    _write_json(output_root / "test-metrics.json", test_metrics)
    _write_json(output_root / "test-diagnostics.json", test_diagnostics)
    _write_json(output_root / "incremental-event-value.json", deltas)
    _write_json(output_root / "loio-development-diagnostics.json", loio)
    _write_json(output_root / "exact-events-descriptive.json", exact_report)
    _write_jsonl(output_root / "validation-predictions.jsonl", validation_records)
    _write_jsonl(output_root / "test-predictions.jsonl", test_records)

    generated_at = (created_at or datetime.now(UTC)).isoformat()
    manifest: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        "dataset_version": DATASET_VERSION,
        "dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_sha": EXPECTED_PROVENANCE_SHA,
        "frozen_feature_schema_sha": EXPECTED_FEATURE_SCHEMA_SHA,
        "comparison_cohort_sha": cohort.cohort_sha,
        "event_feature_schema_sha": cohort.event_schema_sha,
        "market_feature_schema_sha": cohort.market_schema_sha,
        "event_feature_count": len(cohort.event_feature_names),
        "market_feature_count": len(cohort.market_feature_names),
        "combined_feature_count": len(cohort.event_feature_names)
        + len(cohort.market_feature_names),
        "reaction_family": REACTION_FAMILY,
        "predictive_unit": "EVENT",
        "comparison_cohort_rows": len(cohort.rows),
        "comparison_cohort_tickers": sorted({row.ticker for row in cohort.rows}),
        "comparison_cohort_date_range": {
            "from": min(row.publication_date for row in cohort.rows).isoformat(),
            "to": max(row.publication_date for row in cohort.rows).isoformat(),
        },
        "split_sha": split["split_sha"],
        "split_counts": split["counts"],
        "split_tickers": split["ticker_counts"],
        "split_date_ranges": split["date_ranges"],
        "target_class_distribution": {
            split_name: _class_distribution(
                cohort.rows_for(assignments, split_name),
                development_targets if split_name != "TEST" else test_targets,
            )
            for split_name in ("TRAIN", "VALIDATION", "TEST")
        },
        "models": {
            "classification": "multiclass LogisticRegression",
            "regression": "Ridge",
            "feature_families": list(FEATURE_FAMILIES),
            "model_binary_sha": model_binary_sha,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_diagnostics": test_diagnostics,
        "incremental_deltas": deltas,
        "loio": loio,
        "exact_events": exact_report,
        "TEST_CONFIG_LOCKED": "YES",
        "TEST_EVALUATION_COUNT": 1,
        "TEST_STATUS": TEST_STATUS,
        "EVENT_INCREMENTAL_VALUE_STATUS": incremental_status,
        "CONFIRMED_SIGNAL": False,
        "EXACT_MODEL_STATUS": "INSUFFICIENT_DATA_FOR_BASELINE",
        "EXACT_MODEL_TRAINED": False,
        "NLP_FROZEN": True,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "live_collector_preserved": True,
        "market_only_baseline_status": "FROZEN_NEGATIVE_BASELINE",
        "market_only_daily_rows_as_event_examples": False,
        "EVENT_ISSUER_CONCENTRATION_RISK": "PRESENT",
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "warnings": [
            "EVENT_ISSUER_CONCENTRATION_RISK=PRESENT",
            f"PRICE_ADJUSTMENT_STATUS={PRICE_ADJUSTMENT_STATUS}",
            "ASSOCIATIONAL_RESEARCH_BASELINE_ONLY",
        ],
        "test_reuse_policy": (
            "Observed after event baseline v1. Never tune features, data, split, thresholds, "
            "issuers, or hyperparameters against this TEST; use a new forward holdout."
        ),
        "strategy_backtest_executed": False,
        "paper_trading_executed": False,
        "production_order_executed": False,
        "sandbox_order_executed": False,
        "buy_sell_generated": False,
        "paid_services_used": False,
        "safety": research_safety_flags(),
    }
    manifest["artifact_sha"] = sha256_payload({**manifest, "artifact_sha": None})
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    _replace_json(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 1,
            "TEST_STATUS": TEST_STATUS,
            "locked_config_sha": config_payload["locked_config_sha"],
            "artifact_sha": manifest["artifact_sha"],
        },
    )
    return manifest


def _cohort_manifest(cohort: ComparisonCohort, exclusions: dict[str, Any]) -> dict[str, Any]:
    return {
        "cohort_version": "COMPARISON_COHORT_V1",
        "dataset_sha": EXPECTED_DATASET_SHA,
        "reaction_family": REACTION_FAMILY,
        "rows": len(cohort.rows),
        "tickers": sorted({row.ticker for row in cohort.rows}),
        "date_range": {
            "from": min(row.publication_date for row in cohort.rows).isoformat(),
            "to": max(row.publication_date for row in cohort.rows).isoformat(),
        },
        "event_feature_names": list(cohort.event_feature_names),
        "market_feature_names": list(cohort.market_feature_names),
        "event_feature_schema_sha": cohort.event_schema_sha,
        "market_feature_schema_sha": cohort.market_schema_sha,
        "comparison_cohort_sha": cohort.cohort_sha,
        "same_event_ids_for_A_B_C": True,
        **exclusions,
    }


def _loio_diagnostics(
    rows: tuple[EventFeatureRow, ...],
    targets: dict[str, Any],
    config: FrozenModelConfig,
) -> dict[str, Any]:
    counts = Counter(row.ticker for row in rows)
    eligible = [ticker for ticker, count in sorted(counts.items()) if count >= 20]
    records: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for ticker in eligible:
        fit_rows = tuple(row for row in rows if row.ticker != ticker)
        held_rows = tuple(row for row in rows if row.ticker == ticker)
        fit_targets = {row.event_id: targets[row.event_id] for row in fit_rows}
        held_targets = {row.event_id: targets[row.event_id] for row in held_rows}
        if len(fit_rows) < 100 or len({target.direction for target in fit_targets.values()}) < 3:
            skipped[ticker] = "INSUFFICIENT_TRAINING_ROWS_OR_CLASSES"
            continue
        models = fit_all_families(fit_rows, fit_targets, config)
        _, held_records = evaluate_all_families(models, held_rows, held_targets, fit_targets)
        records.extend(held_records)
        completed.append({"ticker": ticker, "rows": len(held_rows)})
    if not records:
        return {
            "LOIO_STATUS": "INSUFFICIENT_DATA",
            "minimum_heldout_rows": 20,
            "completed": completed,
            "skipped": skipped,
        }
    return {
        "LOIO_STATUS": "COMPLETED_DEVELOPMENT_ONLY",
        "minimum_heldout_rows": 20,
        "completed": completed,
        "skipped": skipped,
        "aggregate": metrics_from_records(records),
        "diagnostics": grouped_diagnostics(records),
        "test_rows_used": False,
    }


def _exact_report(dataset_root: Path) -> dict[str, Any]:
    features, targets = exact_event_rows(dataset_root)
    metadata = [cast("dict[str, Any]", row["metadata"]) for row in features]
    event_values = [cast("dict[str, Any]", row["event_features"]) for row in features]
    horizons: dict[str, list[float]] = {}
    for target in targets:
        for horizon, values in cast("dict[str, dict[str, Any]]", target["horizons"]).items():
            if values.get("available") and values.get("abnormal_simple_return") is not None:
                horizons.setdefault(horizon, []).append(float(values["abnormal_simple_return"]))
    return {
        "EXACT_TIMESTAMP_EVENTS": 42,
        "REACTION_READY_EXACT": 36,
        "feature_ready_descriptive_rows": len(features),
        "ticker_counts": dict(sorted(Counter(str(row["ticker"]) for row in metadata).items())),
        "event_type_counts": dict(
            sorted(Counter(str(row["primary_event_type"]) for row in event_values).items())
        ),
        "date_range": {
            "from": min(str(row["publication_date"]) for row in metadata),
            "to": max(str(row["publication_date"]) for row in metadata),
        }
        if metadata
        else None,
        "abnormal_reaction_distribution": {
            horizon: _value_summary(values) for horizon, values in sorted(horizons.items())
        },
        "EXACT_MODEL_STATUS": "INSUFFICIENT_DATA_FOR_BASELINE",
        "EXACT_MODEL_TRAINED": False,
    }


def _class_distribution(
    rows: tuple[EventFeatureRow, ...], targets: dict[str, Any]
) -> dict[str, int]:
    return dict(sorted(Counter(str(targets[row.event_id].direction) for row in rows).items()))


def _value_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    split_counts = manifest["split_counts"]
    lines = [
        "# Event predictive baseline v1",
        "",
        "This is a research-only event-level baseline. It estimates post-event association, not "
        "causality, and never generates BUY/SELL decisions.",
        "",
        "## Frozen design",
        "",
        f"- Predictive unit: `{manifest['predictive_unit']}`",
        f"- Primary reaction family: `{manifest['reaction_family']}`",
        f"- Comparison cohort: {manifest['comparison_cohort_rows']} rows",
        "- TRAIN / VALIDATION / TEST: "
        f"{split_counts['TRAIN']} / {split_counts['VALIDATION']} / {split_counts['TEST']}",
        f"- TEST status: `{manifest['TEST_STATUS']}`; evaluation count: "
        f"{manifest['TEST_EVALUATION_COUNT']}",
        "- A = pre-event market context only",
        "- B = frozen Rules v3 event features only",
        "- C = the exact union of A and B",
        "",
        "Preprocessing is fit only on TRAIN for validation and TRAIN+VALIDATION for the single "
        "final TEST evaluation. Dates, issuer-date groups, and same-story groups never cross "
        "splits.",
        "",
        "## Interpretation",
        "",
        f"`EVENT_INCREMENTAL_VALUE_STATUS={manifest['EVENT_INCREMENTAL_VALUE_STATUS']}`.",
        "This result is not a confirmed signal. The corpus has material issuer concentration, and "
        "T-Invest daily candle price-adjustment status remains unverified.",
        "",
        "The TEST set is now observed and must never be reused for feature selection, source or "
        "issuer selection, threshold tuning, model selection, or hyperparameter tuning. A new "
        "forward holdout is required for confirmation.",
        "",
        "No backtest, PnL, Sharpe, portfolio construction, paper trading, broker mutation, sandbox "
        "order, or production order is part of this artifact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _replace_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
