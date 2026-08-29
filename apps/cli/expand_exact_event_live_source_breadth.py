from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_live_source_breadth_expansion.application import (
    build_live_source_breadth_expansion_artifact,
)
from src.exact_event_live_source_breadth_expansion.domain import (
    ARTIFACT_VERSION,
    DEFAULT_ELIGIBILITY_MANIFEST_PATH,
    DEFAULT_INPUT_EVENTS_PATH,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_UNIVERSE_PATH,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_live_source_breadth_expansion_artifact(
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        universe_path=Path(args.universe),
        input_events_path=Path(args.input_events),
        eligibility_manifest_path=Path(args.eligibility_manifest),
        live_registry_path=Path(args.live_registry),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
        write_registry=not args.no_registry_update,
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "TARGET_UNIVERSE_SHA": manifest["TARGET_UNIVERSE_SHA"],
                "DISCOVERY_LIMITS_SHA": manifest["DISCOVERY_LIMITS_SHA"],
                "SOURCE_DISCOVERY_EVIDENCE_SHA": manifest["SOURCE_DISCOVERY_EVIDENCE_SHA"],
                "SOURCE_CANDIDATES_SHA": manifest["SOURCE_CANDIDATES_SHA"],
                "SOURCE_REGISTRY_SHA": manifest["SOURCE_REGISTRY_SHA"],
                "COLLECTION_RESULT_SHA": manifest["COLLECTION_RESULT_SHA"],
                "NEW_EXACT_LIVE_SOURCES": manifest["NEW_EXACT_LIVE_SOURCES"],
                "NEW_CANONICAL_EXACT_EVENTS": manifest["NEW_CANONICAL_EXACT_EVENTS"],
                "NEW_FUTURE_METADATA_ONLY_EVENTS": manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"],
                "REPLAY_ITEMS_NEW": manifest["REPLAY_ITEMS_NEW"],
                "FINAL_DECISION": manifest["FINAL_DECISION"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand live official EXACT source breadth with bounded discovery."
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--input-events", default=DEFAULT_INPUT_EVENTS_PATH)
    parser.add_argument("--eligibility-manifest", default=DEFAULT_ELIGIBILITY_MANIFEST_PATH)
    parser.add_argument("--live-registry", default=DEFAULT_LIVE_REGISTRY_PATH)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--no-registry-update", action="store_true")
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
