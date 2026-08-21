from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.exact_event_warmup_recovery.application import run_warmup_recovery


def run(args: argparse.Namespace) -> int:
    manifest = run_warmup_recovery(
        Path(args.dataset_dir),
        Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        baseline_root=Path(args.baseline_dir),
        v1_dataset_root=Path(args.v1_dataset_dir),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "WARMUP_RECOVERED": manifest["WARMUP_RECOVERED"],
                "WARMUP_REMAINING": manifest["WARMUP_REMAINING"],
                "DATA_RECOVERY_ONLY": True,
                "MODEL_TRAINING_PERFORMED": False,
                "TEST_OUTCOME_USED": False,
                "FUTURE_EVENT_HOLDOUT_USED": False,
                "BACKTEST_APPROVED": False,
                "PAPER_TRADING_APPROVED": False,
                "REAL_TRADING_APPROVED": False,
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover EXACT event warmup market history.")
    parser.add_argument("--dataset-dir", default="artifacts/exact-event-market-dataset-v2")
    parser.add_argument("--v1-dataset-dir", default="artifacts/exact-event-market-dataset-v1")
    parser.add_argument("--baseline-dir", default="artifacts/exact-event-predictive-baseline-v1")
    parser.add_argument(
        "--output-dir", default="artifacts/exact-event-market-history-warmup-recovery-v1"
    )
    parser.add_argument(
        "--base-main-sha",
        default="6f56f592f4dcb306424620c4a3f12a8d9412457d",
    )
    return parser


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
