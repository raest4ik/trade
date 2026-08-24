from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_source_depth_expansion.application import (
    build_source_depth_expansion_artifact,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_source_depth_expansion_artifact(
        input_root=Path(args.input_dir),
        source_registry_path=Path(args.source_registry),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        archive_cache_root=Path(args.archive_cache_root)
        if args.archive_cache_root is not None
        else None,
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
                "SOURCES_AUDITED": manifest["SOURCES_AUDITED"],
                "NEW_EXACT_EVENTS": manifest["NEW_EXACT_EVENTS"],
                "NEW_EXACT_HISTORICAL": manifest["NEW_EXACT_HISTORICAL"],
                "NEW_EXACT_FUTURE_METADATA_ONLY": manifest["NEW_EXACT_FUTURE_METADATA_ONLY"],
                "DATA_EXPANSION_CONCLUSION": manifest["DATA_EXPANSION_CONCLUSION"],
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
    parser = argparse.ArgumentParser(description="Expand EXACT event official archive depth v4.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--input-dir",
        default="artifacts/exact-event-security-history-recovery-v1",
    )
    parser.add_argument(
        "--source-registry",
        default="artifacts/exact-event-source-diversity-v3/source-registry.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-source-depth-expansion-v4",
    )
    parser.add_argument("--archive-cache-root", default=None)
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
