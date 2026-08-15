from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.event_predictive_baseline.data import build_temporal_split, load_exact_horizon_cohorts
from src.event_predictive_baseline.diagnostics import (
    comparison_deltas,
    grouped_diagnostics,
    incremental_value_status,
    timestamp_hypothesis_status,
)
from src.event_predictive_baseline.domain import (
    EXACT_HORIZONS,
    EXPECTED_DATASET_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    MODEL_VERSION,
    PRICE_ADJUSTMENT_STATUS,
    PRIMARY_EXACT_HORIZON,
    SECONDARY_EXACT_HORIZONS,
    TEST_STATUS,
    FrozenModelConfig,
    HorizonCohort,
    research_safety_flags,
    sha256_payload,
)
from src.event_predictive_baseline.modeling import (
    evaluate_all_families,
    fit_all_families,
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
        raise FileExistsError("immutable exact event baseline output already exists")
    cohorts, dataset_metadata = load_exact_horizon_cohorts(dataset_root)
    config = FrozenModelConfig()
    if config.primary_horizon != PRIMARY_EXACT_HORIZON:
        raise ValueError("PRIMARY_EXACT_HORIZON_CHANGED")
    output_root.mkdir(parents=True, exist_ok=True)
    split_payloads = {horizon: build_temporal_split(cohorts[horizon]) for horizon in EXACT_HORIZONS}
    config_payload = _locked_config(config, cohorts, split_payloads)
    _write_exclusive(output_root / "final-test-lock-config.json", config_payload)
    _write_exclusive(
        output_root / "test-evaluation-state.json",
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT_PRIMARY": 0,
            "TEST_EVALUATION_COUNT_SECONDARY": {horizon: 0 for horizon in SECONDARY_EXACT_HORIZONS},
            "TEST_STATUS": "BLIND_LOCKED_NOT_EVALUATED",
            "locked_config_sha": config_payload["locked_config_sha"],
        },
    )
    horizon_results: dict[str, dict[str, Any]] = {}
    model_payloads: dict[str, bytes] = {}
    for horizon in EXACT_HORIZONS:
        result, model_binary = _evaluate_horizon(cohorts[horizon], split_payloads[horizon], config)
        horizon_results[horizon] = result
        model_payloads[horizon] = model_binary
        _write_json(output_root / f"{horizon}-cohort-manifest.json", result["cohort_manifest"])
        _write_json(output_root / f"{horizon}-split-manifest.json", split_payloads[horizon])
        _write_json(
            output_root / f"{horizon}-validation-metrics.json", result["validation_metrics"]
        )
        _write_json(output_root / f"{horizon}-test-metrics.json", result["test_metrics"])
        _write_json(output_root / f"{horizon}-test-diagnostics.json", result["test_diagnostics"])
        _write_json(output_root / f"{horizon}-deltas.json", result["deltas"])
        _write_jsonl(
            output_root / f"{horizon}-validation-predictions.jsonl", result["validation_records"]
        )
        _write_jsonl(output_root / f"{horizon}-test-predictions.jsonl", result["test_records"])
    model_binary_sha = _write_models(output_root / "models-by-horizon.pkl", model_payloads)
    primary = horizon_results[PRIMARY_EXACT_HORIZON]
    incremental_status = incremental_value_status(
        primary["test_metrics"], primary["test_diagnostics"]
    )
    timestamp_status = timestamp_hypothesis_status(incremental_status)
    exact_vs_date = _exact_vs_date_diagnostic()
    generated_at = (created_at or datetime.now(UTC)).isoformat()
    manifest: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        "dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_sha": EXPECTED_PROVENANCE_SHA,
        "primary_horizon": PRIMARY_EXACT_HORIZON,
        "secondary_horizons": list(SECONDARY_EXACT_HORIZONS),
        "cohort_shas": {horizon: cohorts[horizon].cohort_sha for horizon in EXACT_HORIZONS},
        "split_shas": {horizon: split_payloads[horizon]["split_sha"] for horizon in EXACT_HORIZONS},
        "feature_schema_shas": {
            horizon: {
                "A_MARKET_ONLY": cohorts[horizon].market_schema_sha,
                "B_EVENT_ONLY": cohorts[horizon].event_schema_sha,
                "C_EVENT_PLUS_MARKET": sha256_payload(
                    [cohorts[horizon].event_schema_sha, cohorts[horizon].market_schema_sha]
                ),
            }
            for horizon in EXACT_HORIZONS
        },
        "feature_counts": {
            horizon: {
                "A_MARKET_ONLY": len(cohorts[horizon].market_feature_names),
                "B_EVENT_ONLY": len(cohorts[horizon].event_feature_names),
                "C_EVENT_PLUS_MARKET": len(cohorts[horizon].event_feature_names)
                + len(cohorts[horizon].market_feature_names),
            }
            for horizon in EXACT_HORIZONS
        },
        "target_methodology": {
            "regression": "abnormal_return_h = security_return_h - IMOEX_return_h",
            "classification": "UP/FLAT/DOWN using frozen +/-0.002 abnormal return threshold",
            "window_policy": (
                "target window starts strictly after publication timestamp via exact "
                "corpus alignment"
            ),
            "target_schema_sha": cohorts[PRIMARY_EXACT_HORIZON].target_schema_sha,
        },
        "model_configs": config_payload,
        "horizon_results": _manifest_horizon_results(horizon_results),
        "primary_results": _manifest_horizon_results({PRIMARY_EXACT_HORIZON: primary})[
            PRIMARY_EXACT_HORIZON
        ],
        "primary_c_minus_a": primary["deltas"]["C_vs_A"],
        "secondary_summary": {
            horizon: _secondary_summary(horizon_results[horizon])
            for horizon in SECONDARY_EXACT_HORIZONS
        },
        "exact_vs_date_diagnostic": exact_vs_date,
        "dataset_metadata": dataset_metadata,
        "EVENT_MARKET_LEAKAGE_CHECK": dataset_metadata["EVENT_MARKET_LEAKAGE_CHECK"],
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": dataset_metadata["FUTURE_EVENT_HOLDOUT_OBSERVED"],
        "holdout_guard": dataset_metadata["holdout_guard"],
        "TEST_CONFIG_LOCKED": "YES",
        "TEST_EVALUATION_COUNT_PRIMARY": 1,
        "TEST_EVALUATION_COUNT_SECONDARY": {horizon: 1 for horizon in SECONDARY_EXACT_HORIZONS},
        "TEST_STATUS": TEST_STATUS,
        "EXACT_EVENT_INCREMENTAL_VALUE_STATUS": incremental_status,
        "TIMESTAMP_HYPOTHESIS_STATUS": timestamp_status,
        "CONFIRMED_SIGNAL": False,
        "EVENT_ISSUER_CONCENTRATION_RISK": "PRESENT",
        "PRICE_ADJUSTMENT_STATUS": PRICE_ADJUSTMENT_STATUS,
        "NLP_FROZEN": True,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "model_trained": True,
        "abc_evaluated": True,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "orders_submitted": False,
        "buy_sell_generated": False,
        "real_trading_executed": False,
        "paid_services_used": False,
        "model_binary_sha": model_binary_sha,
        "warnings": [
            "EVENT_ISSUER_CONCENTRATION_RISK=PRESENT",
            f"PRICE_ADJUSTMENT_STATUS={PRICE_ADJUSTMENT_STATUS}",
            "ASSOCIATIONAL_RESEARCH_BASELINE_ONLY",
            "CONFIRMED_SIGNAL=false_UNOPENED_FUTURE_HOLDOUT",
        ],
        "safety": research_safety_flags(),
    }
    manifest["artifact_sha"] = sha256_payload({**manifest, "artifact_sha": None})
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    _replace_json(
        output_root / "test-evaluation-state.json",
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT_PRIMARY": 1,
            "TEST_EVALUATION_COUNT_SECONDARY": {horizon: 1 for horizon in SECONDARY_EXACT_HORIZONS},
            "TEST_STATUS": TEST_STATUS,
            "locked_config_sha": config_payload["locked_config_sha"],
            "artifact_sha": manifest["artifact_sha"],
        },
    )
    return manifest


def _evaluate_horizon(
    cohort: HorizonCohort, split: dict[str, Any], config: FrozenModelConfig
) -> tuple[dict[str, Any], bytes]:
    assignments = {
        str(item["event_id"]): str(item["split"])
        for item in cast("list[dict[str, Any]]", split["assignments"])
    }
    train_rows = cohort.rows_for(assignments, "TRAIN")
    validation_rows = cohort.rows_for(assignments, "VALIDATION")
    test_rows = cohort.rows_for(assignments, "TEST")
    train_targets = {row.event_id: cohort.targets[row.event_id] for row in train_rows}
    validation_targets = {row.event_id: cohort.targets[row.event_id] for row in validation_rows}
    test_targets = {row.event_id: cohort.targets[row.event_id] for row in test_rows}
    development_models = fit_all_families(train_rows, train_targets, config)
    validation_metrics, validation_records = evaluate_all_families(
        development_models, validation_rows, validation_targets, train_targets
    )
    development_rows = (*train_rows, *validation_rows)
    development_targets = {row.event_id: cohort.targets[row.event_id] for row in development_rows}
    final_models = fit_all_families(development_rows, development_targets, config)
    test_metrics, test_records = evaluate_all_families(
        final_models, test_rows, test_targets, development_targets
    )
    test_diagnostics = grouped_diagnostics(test_records)
    validation_diagnostics = grouped_diagnostics(validation_records)
    deltas = {
        "C_vs_A": comparison_deltas(test_metrics, "C_EVENT_PLUS_MARKET", "A_MARKET_ONLY"),
        "B_vs_A": comparison_deltas(test_metrics, "B_EVENT_ONLY", "A_MARKET_ONLY"),
        "C_vs_B": comparison_deltas(test_metrics, "C_EVENT_PLUS_MARKET", "B_EVENT_ONLY"),
    }
    return (
        {
            "horizon": cohort.horizon,
            "cohort_manifest": _cohort_manifest(cohort),
            "split": split,
            "validation_metrics": validation_metrics,
            "validation_diagnostics": validation_diagnostics,
            "test_metrics": test_metrics,
            "test_diagnostics": test_diagnostics,
            "deltas": deltas,
            "validation_records": validation_records,
            "test_records": test_records,
            "target_class_distribution": {
                "TRAIN": _class_distribution(train_targets),
                "VALIDATION": _class_distribution(validation_targets),
                "TEST": _class_distribution(test_targets),
            },
            "loio_development": _loio_placeholder(development_rows),
        },
        serialize_models(final_models),
    )


def _locked_config(
    config: FrozenModelConfig,
    cohorts: dict[str, HorizonCohort],
    splits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        **config.payload(),
        "dataset_sha": EXPECTED_DATASET_SHA,
        "cohort_shas": {horizon: cohorts[horizon].cohort_sha for horizon in EXACT_HORIZONS},
        "split_shas": {horizon: splits[horizon]["split_sha"] for horizon in EXACT_HORIZONS},
        "target_definition": "EXACT_INTRADAY abnormal_return per predeclared horizon",
        "class_threshold_methodology": "frozen project-wide +/-0.002 abnormal return",
        "test_lock_time": "before any TEST target evaluation in runner order",
    }
    payload["locked_config_sha"] = sha256_payload(payload)
    return payload


def _cohort_manifest(cohort: HorizonCohort) -> dict[str, Any]:
    return {
        "cohort_version": "EXACT_EVENT_COHORT_V1",
        "horizon": cohort.horizon,
        "dataset_sha": EXPECTED_DATASET_SHA,
        "cohort_sha": cohort.cohort_sha,
        "rows": len(cohort.rows),
        "tickers": sorted({row.ticker for row in cohort.rows}),
        "date_range": {
            "from": min(row.publication_date for row in cohort.rows).isoformat(),
            "to": max(row.publication_date for row in cohort.rows).isoformat(),
        },
        "event_feature_names": list(cohort.event_feature_names),
        "market_feature_names": list(cohort.market_feature_names),
        "event_schema_sha": cohort.event_schema_sha,
        "market_schema_sha": cohort.market_schema_sha,
        "target_schema_sha": cohort.target_schema_sha,
        "same_event_ids_for_A_B_C": True,
        "future_holdout_used": False,
    }


def _manifest_horizon_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for horizon, result in results.items():
        split = result["split"]
        diagnostics = result["test_diagnostics"]
        payload[horizon] = {
            "cohort_rows": result["cohort_manifest"]["rows"],
            "cohort_sha": result["cohort_manifest"]["cohort_sha"],
            "tickers": result["cohort_manifest"]["tickers"],
            "date_range": result["cohort_manifest"]["date_range"],
            "split_sha": split["split_sha"],
            "split_counts": split["counts"],
            "split_tickers": split["ticker_counts"],
            "split_date_ranges": split["date_ranges"],
            "target_class_distribution": result["target_class_distribution"],
            "classification": result["test_metrics"]["classification"],
            "regression": result["test_metrics"]["regression"],
            "deltas": result["deltas"],
            "ROW_WEIGHTED": diagnostics["ROW_WEIGHTED"],
            "ISSUER_MACRO": diagnostics["ISSUER_MACRO"],
            "per_ticker": diagnostics["per_ticker"],
            "per_event_type": diagnostics["per_event_type"],
            "concentration": diagnostics["concentration"],
            "loio_development": result["loio_development"],
        }
    return payload


def _secondary_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["test_metrics"]
    deltas = result["deltas"]["C_vs_A"]
    return {
        "rows": result["cohort_manifest"]["rows"],
        "split_counts": result["split"]["counts"],
        "C_vs_A_classification": deltas["classification"],
        "C_vs_A_regression": deltas["regression"],
        "A_regression": metrics["regression"]["abnormal_return"]["models"]["A_MARKET_ONLY"],
        "C_regression": metrics["regression"]["abnormal_return"]["models"]["C_EVENT_PLUS_MARKET"],
    }


def _class_distribution(targets: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(target.direction) for target in targets.values()).items()))


def _loio_placeholder(rows: tuple[Any, ...]) -> dict[str, Any]:
    counts = Counter(str(row.ticker) for row in rows)
    eligible = {ticker: count for ticker, count in sorted(counts.items()) if count >= 20}
    return {
        "LOIO_STATUS": "PREDECLARED_DEVELOPMENT_ONLY_NOT_USED_FOR_TEST_TUNING",
        "test_rows_used": False,
        "eligible_issuers": eligible,
    }


def _exact_vs_date_diagnostic() -> dict[str, Any]:
    return {
        "audit_type": "DESCRIPTIVE_ONLY",
        "status": "NOT_RUN_NO_CANONICAL_EXACT_DATE_SAFE_PAIRING_IN_RUNNER",
        "direction_agreement": None,
        "target_correlation": None,
        "magnitude_dispersion": None,
        "model_training_used": False,
        "feature_selection_used": False,
        "threshold_tuning_used": False,
        "future_holdout_used": False,
    }


def _write_models(path: Path, model_payloads: dict[str, bytes]) -> str:
    payload = json.dumps(
        {horizon: hashlib.sha256(value).hexdigest() for horizon, value in model_payloads.items()},
        sort_keys=True,
    ).encode("utf-8")
    path.with_suffix(".sha-manifest.json").write_bytes(payload)
    combined = b"".join(model_payloads[horizon] for horizon in EXACT_HORIZONS)
    path.write_bytes(combined)
    return hashlib.sha256(combined).hexdigest()


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    primary = manifest["primary_results"]
    concentration = primary["concentration"]
    lines = [
        "# Exact event predictive baseline v1",
        "",
        "This is a research-only event-driven baseline for exact publication timestamps.",
        "It compares A market context, B frozen Rules v3 event features, and C their union.",
        "",
        "## Locked Design",
        "",
        f"- Dataset SHA: `{manifest['dataset_sha']}`",
        f"- Primary horizon: `{manifest['primary_horizon']}`",
        f"- Primary cohort rows: {primary['cohort_rows']}",
        f"- Primary cohort SHA: `{primary['cohort_sha']}`",
        f"- Split SHA: `{primary['split_sha']}`",
        f"- TEST status: `{manifest['TEST_STATUS']}`",
        f"- TEST evaluation count primary: {manifest['TEST_EVALUATION_COUNT_PRIMARY']}",
        "",
        "## Primary Result",
        "",
        f"- Incremental value: `{manifest['EXACT_EVENT_INCREMENTAL_VALUE_STATUS']}`",
        f"- Timestamp hypothesis: `{manifest['TIMESTAMP_HYPOTHESIS_STATUS']}`",
        f"- Confirmed signal: `{manifest['CONFIRMED_SIGNAL']}`",
        f"- MGNT/top1 share: {concentration['top1_share']:.6f}",
        f"- HHI: {concentration['hhi']:.6f}",
        f"- Effective issuer count: {concentration['effective_issuer_count']:.6f}",
        "",
        "No future holdout outcomes were used or observed. No PnL, Sharpe, backtest, paper "
        "trading, BUY/SELL signal, order, position sizing, portfolio simulation, or broker "
        "mutation is part of this artifact.",
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
