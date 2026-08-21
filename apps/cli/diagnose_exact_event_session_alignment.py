from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_session_alignment_diagnostics.application import (
    run_session_alignment_diagnostics,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_session_alignment_diagnostics(
        pr38_root=Path(args.pr38_dir),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
        extra_cache_roots=tuple(Path(path) for path in args.extra_cache_dir),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "INPUT_DATASET_SHA": manifest["INPUT_DATASET_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "SESSION_DIAGNOSTIC_COHORT_SHA": manifest["SESSION_DIAGNOSTIC_COHORT_SHA"],
                "DIAGNOSTIC_EVENTS_TOTAL": manifest["DIAGNOSTIC_EVENTS_TOTAL"],
                "ROOT_CAUSE_COUNTS": manifest["ROOT_CAUSE_COUNTS"],
                "DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES": manifest[
                    "DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES"
                ],
                "ALIGNMENT_METHODOLOGY_CHANGED": False,
                "MODEL_TRAINING_PERFORMED": False,
                "TEST_OUTCOME_USED": False,
                "FUTURE_EVENT_HOLDOUT_USED": False,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen EXACT session alignment gaps using PR38 cache."
    )
    parser.add_argument("--pr38-dir", default="artifacts/exact-event-security-history-recovery-v1")
    parser.add_argument(
        "--output-dir", default="artifacts/exact-event-session-alignment-diagnostics-v1"
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--extra-cache-dir", action="append", default=[])
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
