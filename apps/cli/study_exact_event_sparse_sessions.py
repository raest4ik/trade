from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_sparse_session_study.application import run_sparse_session_study


def run(args: argparse.Namespace) -> int:
    manifest = run_sparse_session_study(
        events_path=Path(args.events),
        split_manifest_path=Path(args.split_manifest),
        pr39_root=Path(args.pr39_dir),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        cache_roots=tuple(Path(item) for item in args.cache_root),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "DEVELOPMENT_EXACT_TOTAL": manifest["DEVELOPMENT_EXACT_TOTAL"],
                "TIMESTAMP_STUDY_ELIGIBLE": manifest["TIMESTAMP_STUDY_ELIGIBLE"],
                "TIMESTAMP_STUDY_INELIGIBLE": manifest["TIMESTAMP_STUDY_INELIGIBLE"],
                "OBSERVED_TEST_ROWS_USED": manifest["OBSERVED_TEST_ROWS_USED"],
                "METHODOLOGY_STUDY_RECOMMENDATION": manifest["METHODOLOGY_STUDY_RECOMMENDATION"],
                "METHODOLOGY_CONCLUSION": manifest["METHODOLOGY_CONCLUSION"],
                "STUDY_ARTIFACT_OUTCOME_FREE": manifest["STUDY_ARTIFACT_OUTCOME_FREE"],
                "STUDY_ARTIFACT_PRICE_FREE": manifest["STUDY_ARTIFACT_PRICE_FREE"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Study sparse exact-event session delays using timestamp-only metadata."
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--events",
        default="artifacts/exact-event-security-history-recovery-v1/events.jsonl",
    )
    parser.add_argument(
        "--split-manifest",
        default="artifacts/exact-event-predictive-baseline-v1/15m-split-manifest.json",
    )
    parser.add_argument(
        "--pr39-dir",
        default="artifacts/exact-event-session-alignment-diagnostics-v1",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-sparse-session-methodology-study-v1",
    )
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[
            "artifacts/exact-event-security-history-recovery-v1/raw-minute-cache",
            "artifacts/exact-event-market-dataset-v2/raw-minute-cache",
            "artifacts/exact-event-market-dataset-v1/raw-minute-cache",
        ],
    )
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
