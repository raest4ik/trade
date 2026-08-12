from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.predictive_baseline.application import dataset_readiness
from src.predictive_baseline.data import load_daily_predictive_dataset
from src.predictive_baseline.domain import BaselineConfig, RunMode, TrainingGate
from src.predictive_baseline.modeling import train_predictive_baselines
from src.predictive_baseline.reporting import write_readiness, write_training_artifacts


def run(args: argparse.Namespace) -> int:
    feature_path = Path(args.features)
    reaction_path = Path(args.reactions)
    output = Path(args.output_dir)
    dataset = load_daily_predictive_dataset(feature_path, reaction_path)
    readiness = dataset_readiness(dataset, intraday_feature_ready=args.intraday_feature_ready)
    write_readiness(output / "readiness.json", readiness)
    mode = RunMode.DEVELOPMENT_SMOKE if args.development_smoke else RunMode.REAL
    config = BaselineConfig()
    result = train_predictive_baselines(dataset, config, mode=mode)
    if result.gate == TrainingGate.TRAINING_BLOCKED and mode == RunMode.REAL:
        print(
            json.dumps(
                {
                    **readiness,
                    "status": TrainingGate.TRAINING_BLOCKED.value,
                    "reason": "daily_feature_ready < 100",
                    "application_error": False,
                    "model_trained": False,
                },
                sort_keys=True,
            )
        )
        return 0
    paths = write_training_artifacts(
        output,
        dataset=dataset,
        config=config,
        result=result,
        git_sha=_git_sha(),
    )
    summary: dict[str, Any] = {
        "status": result.status,
        "mode": result.mode.value,
        "warnings": list(result.warnings),
        "model_trained_in_memory": True,
        "production_model_saved": result.model_binary is not None,
        "test_used_for_tuning": result.test_used_for_tuning,
        "classification_test": result.classification_metrics["test"],
        "regression_test": result.regression_metrics["test"],
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe daily classification and regression baselines."
    )
    parser.add_argument(
        "--features",
        default="artifacts/free-daily-historical-v1/daily-feature-dataset.jsonl",
    )
    parser.add_argument(
        "--reactions",
        default="artifacts/free-daily-historical-v1/daily-reactions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/predictive-baseline-v1",
    )
    parser.add_argument("--intraday-feature-ready", type=int, default=21)
    parser.add_argument(
        "--development-smoke",
        action="store_true",
        help="Explicitly permit a non-production smoke fit below the data gate.",
    )
    return parser


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
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
