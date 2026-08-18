from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.exact_event_diagnostics.application import run_exact_event_data_diagnostics


def run(args: argparse.Namespace) -> int:
    manifest = run_exact_event_data_diagnostics(
        Path(args.dataset_dir),
        Path(args.baseline_dir),
        Path(args.output_dir),
        git_sha=_git_sha(),
    )
    print(
        json.dumps(
            {
                "artifact_sha": manifest["artifact_sha"],
                "output_dir": args.output_dir,
                "NEXT_DATA_PRIORITY": manifest["diagnostics"]["priority_report"][
                    "NEXT_DATA_PRIORITY"
                ],
                "DIAGNOSTIC_ONLY": True,
                "MODEL_TRAINING_PERFORMED": False,
                "TEST_OUTCOME_USED": False,
                "FUTURE_EVENT_HOLDOUT_USED": False,
                "BACKTEST_APPROVED": False,
                "PAPER_TRADING_APPROVED": False,
                "REAL_TRADING_APPROVED": False,
                "CONFIRMED_SIGNAL": False,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EXACT event data diagnostics v1.")
    parser.add_argument("--dataset-dir", default="artifacts/exact-event-market-dataset-v2")
    parser.add_argument("--baseline-dir", default="artifacts/exact-event-predictive-baseline-v1")
    parser.add_argument("--output-dir", default="artifacts/exact-event-data-diagnostics-v1")
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
