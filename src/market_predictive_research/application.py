from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.market_predictive_research.data import (
    future_holdout_coverage,
    load_development_dataset,
)
from src.market_predictive_research.diagnostics import (
    dataset_diagnostics,
    fold_feature_associations,
)
from src.market_predictive_research.domain import (
    RESEARCH_VERSION,
    TARGET_HORIZONS,
    frozen_research_metadata,
    safety_flags,
    sha256_payload,
)
from src.market_predictive_research.folds import build_rolling_folds
from src.market_predictive_research.modeling import (
    MODEL_CONFIGS,
    aggregate_research_results,
    evaluate_fold,
    stability_views,
)


def run_development_research(
    dataset_root: Path,
    output_root: Path,
    *,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable market research output already exists")
    dataset = load_development_dataset(dataset_root)
    fold_manifest = build_rolling_folds(dataset)
    diagnostics = dataset_diagnostics(dataset)
    diagnostics["feature_associations_by_fold"] = fold_feature_associations(dataset, fold_manifest)
    fold_results = [evaluate_fold(dataset, fold) for fold in fold_manifest.folds]
    aggregate = aggregate_research_results(fold_results)
    best = aggregate["best_development_candidate"]
    stability = stability_views(
        fold_results, model=str(best["model"]), horizon=int(best["horizon"])
    )
    future = future_holdout_coverage(dataset_root / "features.jsonl")
    public_fold_results = [
        {key: value for key, value in result.items() if key != "validation_predictions"}
        for result in fold_results
    ]
    generated_at = (created_at or datetime.now(UTC)).isoformat()
    manifest: dict[str, Any] = {
        "research_version": RESEARCH_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        **frozen_research_metadata(),
        "development_rows": len(dataset.rows),
        "development_tickers": sorted({row.ticker for row in dataset.rows}),
        "feature_count": len(dataset.feature_names),
        "feature_names": list(dataset.feature_names),
        "feature_schema_sha": dataset.feature_schema_sha,
        "target_horizons": list(TARGET_HORIZONS),
        "target_definitions": {
            "security_return": "compounded forward security return from target session",
            "abnormal_return": (
                "compounded security return minus compounded T-Invest IMOEX return"
            ),
            "direction_threshold": "0.002 * sqrt(horizon), declared before fold evaluation",
        },
        "model_configs": MODEL_CONFIGS,
        "fold_manifest_sha": fold_manifest.fold_manifest_sha,
        "fold_count_per_horizon": 5,
        "fold_count_total": len(fold_manifest.folds),
        "diagnostics": diagnostics,
        "fold_results": public_fold_results,
        "aggregate_results": aggregate,
        "stability": stability,
        "future_holdout": future,
        "model_trained_for_research": True,
        "CONFIRMED_SIGNAL": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "production_order_executed": False,
        "sandbox_order_executed": False,
        "buy_sell_generated": False,
        "live_automation_connected": False,
        "safety": safety_flags(),
        "warnings": [
            "ASSOCIATIONAL_RESEARCH_ONLY",
            dataset.price_adjustment_status,
            "NO_NEW_BLIND_HOLDOUT_CONFIRMATION",
        ],
    }
    manifest["artifact_sha"] = sha256_payload({**manifest, "artifact_sha": None})
    output_root.mkdir(parents=True, exist_ok=False)
    _json(output_root / "manifest.json", manifest)
    _json(output_root / "fold-manifest.json", fold_manifest.payload())
    _json(output_root / "diagnostics.json", diagnostics)
    _json(output_root / "fold-metrics.json", public_fold_results)
    _json(output_root / "stability.json", stability)
    _json(output_root / "future-holdout-status.json", future)
    return manifest


def future_status(dataset_root: Path) -> dict[str, object]:
    return future_holdout_coverage(dataset_root / "features.jsonl")


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
