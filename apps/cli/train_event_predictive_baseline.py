from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.event_predictive_baseline.application import run_event_predictive_baseline


def run(args: argparse.Namespace) -> int:
    manifest = run_event_predictive_baseline(
        Path(args.dataset_dir),
        Path(args.output_dir),
        git_sha=_git_sha(),
    )
    print(
        json.dumps(
            {
                "EVENT_INCREMENTAL_VALUE_STATUS": manifest["EVENT_INCREMENTAL_VALUE_STATUS"],
                "TEST_CONFIG_LOCKED": manifest["TEST_CONFIG_LOCKED"],
                "TEST_EVALUATION_COUNT": manifest["TEST_EVALUATION_COUNT"],
                "TEST_STATUS": manifest["TEST_STATUS"],
                "artifact_sha": manifest["artifact_sha"],
                "output_dir": args.output_dir,
                "research_only": True,
                "backtest": False,
                "trading": False,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-time frozen event-level A/B/C predictive baseline."
    )
    parser.add_argument("--dataset-dir", default="artifacts/event-market-predictive-dataset-v2")
    parser.add_argument("--output-dir", default="artifacts/event-predictive-baseline-v1")
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
