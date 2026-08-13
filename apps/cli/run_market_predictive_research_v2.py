from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.market_predictive_research.application import run_development_research


def run(args: argparse.Namespace) -> int:
    result = run_development_research(
        Path(args.dataset_dir), Path(args.output_dir), git_sha=_git_sha()
    )
    print(
        json.dumps(
            {
                "DEVELOPMENT_STATUS": result["aggregate_results"]["DEVELOPMENT_STATUS"],
                "CONFIRMED_SIGNAL": False,
                "OBSERVED_TEST_USED": False,
                "FUTURE_HOLDOUT_USED": False,
                "artifact_sha": result["artifact_sha"],
                "fold_manifest_sha": result["fold_manifest_sha"],
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run development-only market predictive research v2."
    )
    parser.add_argument("--dataset-dir", default="artifacts/tinvest-market-baseline-features-v1")
    parser.add_argument("--output-dir", default="artifacts/market-predictive-research-v2")
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
