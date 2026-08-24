from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_live_source_snapshot.application import (
    build_live_source_snapshot_artifact,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_live_source_snapshot_artifact(
        input_root=Path(args.input_dir),
        source_registry_path=Path(args.source_registry),
        universe_path=Path(args.universe),
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
                "INPUT_DATASET_SHA": manifest["INPUT_DATASET_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "LIVE_DISCOVERY_EXECUTED": manifest["LIVE_DISCOVERY_EXECUTED"],
                "LIVE_DISCOVERY_BLOCKER": manifest["LIVE_DISCOVERY_BLOCKER"],
                "LIVE_REQUESTS_TOTAL": manifest["LIVE_REQUESTS_TOTAL"],
                "LIVE_CANDIDATES_WRITTEN": manifest["LIVE_CANDIDATES_WRITTEN"],
                "V5_DOWNSTREAM_ARTIFACT_SHA": manifest["V5_DOWNSTREAM_ARTIFACT_SHA"],
                "V5_NEW_EXACT_CAPABLE_SOURCES": manifest["V5_NEW_EXACT_CAPABLE_SOURCES"],
                "V5_NEW_EXACT_EVENTS": manifest["V5_NEW_EXACT_EVENTS"],
                "LIVE_SOURCE_DISCOVERY_CONCLUSION": manifest["LIVE_SOURCE_DISCOVERY_CONCLUSION"],
                "MODEL_TRAINING_PERFORMED": manifest["MODEL_TRAINING_PERFORMED"],
                "TEST_OUTCOME_USED": manifest["TEST_OUTCOME_USED"],
                "FUTURE_EVENT_HOLDOUT_USED": manifest["FUTURE_EVENT_HOLDOUT_USED"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire live official source snapshot v1.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--input-dir",
        default="artifacts/exact-event-official-source-discovery-v5",
    )
    parser.add_argument(
        "--source-registry",
        default="artifacts/exact-event-official-source-discovery-v5/source-registry.jsonl",
    )
    parser.add_argument(
        "--universe",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-live-official-source-snapshot-v1",
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
