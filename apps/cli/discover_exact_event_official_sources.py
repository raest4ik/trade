from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_official_source_discovery.application import (
    build_official_source_discovery_artifact,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_official_source_discovery_artifact(
        input_root=Path(args.input_dir),
        source_registry_path=Path(args.source_registry),
        universe_path=Path(args.universe),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        discovery_cache_root=Path(args.discovery_cache_root)
        if args.discovery_cache_root is not None
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
                "NEW_OFFICIAL_SOURCES_FOUND": manifest["NEW_OFFICIAL_SOURCES_FOUND"],
                "NEW_EXACT_CAPABLE_SOURCES": manifest["NEW_EXACT_CAPABLE_SOURCES"],
                "NEW_EXACT_EVENTS": manifest["NEW_EXACT_EVENTS"],
                "SOURCE_DISCOVERY_CONCLUSION": manifest["SOURCE_DISCOVERY_CONCLUSION"],
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
    parser = argparse.ArgumentParser(description="Discover official EXACT event sources v5.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--input-dir",
        default="artifacts/exact-event-source-depth-expansion-v4",
    )
    parser.add_argument(
        "--source-registry",
        default="artifacts/exact-event-source-depth-expansion-v4/source-registry.jsonl",
    )
    parser.add_argument(
        "--universe",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-official-source-discovery-v5",
    )
    parser.add_argument("--discovery-cache-root", default=None)
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
