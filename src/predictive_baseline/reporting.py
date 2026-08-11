from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.predictive_baseline.domain import (
    MODEL_VERSION,
    BaselineConfig,
    LoadedDataset,
    ModelArtifactManifest,
    PredictiveRow,
    RunMode,
    TrainingResult,
    sha256_payload,
)


def write_readiness(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return path


def write_training_artifacts(
    output_root: Path,
    *,
    dataset: LoadedDataset,
    config: BaselineConfig,
    result: TrainingResult,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Path]:
    if result.split is None:
        raise ValueError("blocked training has no model artifacts")
    if result.mode == RunMode.DEVELOPMENT_SMOKE and result.model_binary is not None:
        raise ValueError("development smoke must not persist a production model binary")
    if result.mode == RunMode.REAL and result.model_binary is None:
        raise ValueError("real training requires a versioned model binary")
    if result.model_binary is not None:
        actual_hash = hashlib.sha256(result.model_binary).hexdigest()
        if actual_hash != result.model_binary_sha256:
            raise ValueError("model binary hash does not match manifest")
    generated_at = created_at or datetime.now(UTC)
    run_fingerprint = sha256_payload(
        {
            "dataset_sha256": dataset.dataset_sha256,
            "feature_schema_sha256": dataset.feature_schema_sha256,
            "split_sha256": result.split.split_sha256,
            "config": config.payload(),
            "git_sha": git_sha,
            "mode": result.mode.value,
        }
    )
    run_id = f"{generated_at.strftime('%Y%m%dT%H%M%S%fZ')}-{run_fingerprint[:12]}"
    category = "development-smoke" if result.mode == RunMode.DEVELOPMENT_SMOKE else "training-runs"
    run_directory = output_root / category / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    metrics = {
        "classification": result.classification_metrics,
        "regression": result.regression_metrics,
        "test_used_for_tuning": result.test_used_for_tuning,
        "warnings": list(result.warnings),
    }
    manifest = ModelArtifactManifest(
        model_version=MODEL_VERSION,
        dataset_sha256=dataset.dataset_sha256,
        feature_schema_sha256=dataset.feature_schema_sha256,
        split_sha256=result.split.split_sha256,
        git_sha=git_sha,
        training_config=config.payload(),
        training_period=_period(result.split.train),
        validation_period=_period(result.split.validation),
        test_period=_period(result.split.test),
        metrics=metrics,
        model_binary_sha256=result.model_binary_sha256,
        created_at=generated_at.isoformat(),
        mode=result.mode.value,
        warnings=result.warnings,
    )
    paths = {
        "directory": run_directory,
        "manifest": run_directory / "manifest.json",
        "metrics": run_directory / "metrics.json",
        "predictions": run_directory / "predictions.jsonl",
    }
    _write_json(paths["manifest"], manifest.payload())
    _write_json(paths["metrics"], metrics)
    _write_jsonl(paths["predictions"], [item.payload() for item in result.predictions])
    if result.model_binary is not None:
        model_path = run_directory / "model.pkl"
        model_path.write_bytes(result.model_binary)
        paths["model"] = model_path
        manifest_directory = output_root / "model-manifests"
        manifest_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_directory / f"{run_id}.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(
                json.dumps(manifest.payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        paths["model_manifest"] = manifest_path
    return paths


def _period(rows: tuple[PredictiveRow, ...]) -> dict[str, str]:
    dates = sorted(row.publication_date for row in rows)
    return {"from": dates[0].isoformat(), "to": dates[-1].isoformat()}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
