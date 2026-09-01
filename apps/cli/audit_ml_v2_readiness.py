from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_dataset_readiness_audit.domain import (
    DEFAULT_INPUT_ARTIFACT_ROOT,
    DEFAULT_OLD_BASELINE_ROOT,
    ML_V2_ARTIFACT_VERSION,
)
from src.exact_dataset_readiness_audit.ml_v2 import run_ml_v2_readiness_audit


def run(args: argparse.Namespace) -> int:
    manifest = run_ml_v2_readiness_audit(
        input_root=Path(args.input_dir),
        old_baseline_root=Path(args.old_baseline_dir),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "CANONICAL_COHORT": manifest["CANONICAL_COHORT"],
                "FEATURE_READY_ISSUER_ROWS": manifest["ISSUER_ORIGINATED_FEATURE_READY_EVENTS"],
                "ISSUER_TICKERS": manifest["UNIQUE_ISSUER_TICKERS"],
                "ISSUER_UNKNOWN_RATE": manifest["ISSUER_UNKNOWN_RATE"],
                "TOP_1_TICKER_SHARE": manifest["TOP_1_TICKER_SHARE"],
                "SOURCE_FAMILY_HHI": manifest["SOURCE_FAMILY_HHI"],
                "PRIMARY_15M_TARGET_COVERAGE": manifest["PRIMARY_15M_TARGET_COVERAGE"],
                "LEAKAGE_AUDIT": manifest["LEAKAGE_AUDIT"],
                "OLD_BASELINE_TEST_STATUS": manifest["OLD_BASELINE_TEST_STATUS"],
                "FINAL_READINESS_DECISION": manifest["FINAL_READINESS_DECISION"],
                "CAN_START_CONTROLLED_ML_V2": manifest["CAN_START_CONTROLLED_ML_V2"],
                "MAIN_BLOCKER": manifest["MAIN_BLOCKER"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical strict-EXACT issuer/event dataset readiness for ML v2."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_ARTIFACT_ROOT)
    parser.add_argument("--old-baseline-dir", default=DEFAULT_OLD_BASELINE_ROOT)
    parser.add_argument("--output-dir", default=f"artifacts/{ML_V2_ARTIFACT_VERSION}")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
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
