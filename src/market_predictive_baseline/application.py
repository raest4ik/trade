from __future__ import annotations

import hashlib
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.market_predictive_baseline.data import (
    load_frozen_market_dataset,
    load_targets_for_splits,
)
from src.market_predictive_baseline.domain import (
    ASSOCIATIONAL_WARNING,
    MODEL_VERSION,
    PRICE_WARNING,
    TEST_STATUS,
    FinalModelConfig,
    research_safety_flags,
    sha256_payload,
)
from src.market_predictive_baseline.modeling import (
    MarketModels,
    diagnostic_views,
    evaluate_models,
    model_quality_status,
)


def run_frozen_market_baseline(
    dataset_root: Path,
    output_root: Path,
    *,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable market baseline output already exists")
    dataset = load_frozen_market_dataset(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    targets_path = dataset_root / "targets.jsonl"
    development_targets = load_targets_for_splits(
        targets_path, dataset.assignments, frozenset({"TRAIN", "VALIDATION"})
    )
    train_rows = dataset.rows_for("TRAIN")
    validation_rows = dataset.rows_for("VALIDATION")
    train_targets = {row.row_id: development_targets[row.row_id] for row in train_rows}
    validation_targets = {row.row_id: development_targets[row.row_id] for row in validation_rows}
    stage_one = MarketModels.create(FinalModelConfig())
    stage_one.fit(train_rows, train_targets)
    validation_metrics, _ = evaluate_models(
        stage_one, validation_rows, validation_targets, train_targets
    )
    config = FinalModelConfig()
    config_payload = config.payload()
    _write_exclusive(output_root / "final_model_config.json", config_payload)
    state_path = output_root / "test-evaluation-state.json"
    _write_exclusive(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 0,
            "TEST_STATUS": "BLIND_LOCKED_NOT_EVALUATED",
            "config_sha": config_payload["config_sha"],
        },
    )
    final_rows = (*train_rows, *validation_rows)
    final_targets = {**train_targets, **validation_targets}
    final_models = MarketModels.create(config)
    final_models.fit(final_rows, final_targets)
    _replace_json(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 1,
            "TEST_STATUS": "EVALUATION_STARTED_NO_RETRY_ALLOWED",
            "config_sha": config_payload["config_sha"],
        },
    )
    test_rows = dataset.rows_for("TEST")
    test_targets = load_targets_for_splits(targets_path, dataset.assignments, frozenset({"TEST"}))
    test_metrics, predictions = evaluate_models(
        final_models, test_rows, test_targets, final_targets
    )
    diagnostics = diagnostic_views(predictions)
    coefficients = final_models.coefficients(dataset.feature_names)
    quality_status = model_quality_status(test_metrics, diagnostics)
    generated_at = (created_at or datetime.now(UTC)).isoformat()
    model_bytes = pickle.dumps(final_models, protocol=pickle.HIGHEST_PROTOCOL)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    model_path = output_root / "model.pkl"
    model_path.write_bytes(model_bytes)
    _write_jsonl(output_root / "test-predictions.jsonl", predictions)
    _write_json(output_root / "validation-metrics.json", validation_metrics)
    _write_json(output_root / "test-metrics.json", test_metrics)
    _write_json(output_root / "test-diagnostics.json", diagnostics)
    _write_json(output_root / "coefficients.json", coefficients)
    manifest: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        "dataset_version": dataset.dataset_version,
        "dataset_sha": dataset.dataset_sha,
        "split_sha": dataset.split_sha,
        "feature_schema_sha": dataset.feature_schema_sha,
        "target_version": "tinvest-next-session-targets-v1",
        "model_type": {
            "classification": "multiclass LogisticRegression",
            "regression": "Ridge",
        },
        "hyperparameters": {
            "classification": dict(config.classifier_parameters),
            "regression": dict(config.regressor_parameters),
        },
        "preprocessing": config.preprocessing,
        "seed": config.random_seed,
        "rows": dataset.counts,
        "date_ranges": dataset.date_ranges,
        "feature_names": list(dataset.feature_names),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_diagnostics": diagnostics,
        "coefficients": coefficients,
        "TEST_CONFIG_LOCKED": "YES",
        "TEST_EVALUATION_COUNT": 1,
        "TEST_STATUS": TEST_STATUS,
        "MODEL_QUALITY_STATUS": quality_status,
        "model_binary_sha": model_sha,
        "config_sha": config_payload["config_sha"],
        "price_adjustment_status": dataset.price_adjustment_status,
        "source_usage_readiness": dataset.source_usage_readiness,
        "warnings": [ASSOCIATIONAL_WARNING, PRICE_WARNING],
        "test_reuse_policy": (
            "Observed after baseline v1; do not use for iterative tuning. "
            "A new forward holdout is required for blind confirmation."
        ),
        "strategy_backtest_executed": False,
        "paper_trading_executed": False,
        "production_order_executed": False,
        "sandbox_order_executed": False,
        "buy_sell_generated": False,
        "live_automation_connected": False,
        "safety": research_safety_flags(),
    }
    fingerprint_payload = {**manifest, "artifact_sha": None}
    manifest["artifact_sha"] = sha256_payload(fingerprint_payload)
    _write_json(output_root / "manifest.json", manifest)
    _replace_json(
        state_path,
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT": 1,
            "TEST_STATUS": TEST_STATUS,
            "config_sha": config_payload["config_sha"],
            "artifact_sha": manifest["artifact_sha"],
        },
    )
    return manifest


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
