from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_live_official_collection.application import (
    build_live_official_collection_artifact,
)
from src.exact_event_live_official_collection.domain import DEFAULT_SOURCE_REGISTRY_PATH


def run(args: argparse.Namespace) -> int:
    manifest = build_live_official_collection_artifact(
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        input_events_path=Path(args.input_events),
        source_registry_path=Path(args.source_registry),
        audit_manifest_path=Path(args.audit_manifest) if args.audit_manifest else None,
        state_path=Path(args.state_file) if args.state_file else None,
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
        event_origin_filter=tuple(args.event_origin) if args.event_origin else None,
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "SOURCE_REGISTRY_SHA": manifest["SOURCE_REGISTRY_SHA"],
                "NETWORK_PROVENANCE_SHA": manifest["NETWORK_PROVENANCE_SHA"],
                "RAW_SNAPSHOT_SHA": manifest["RAW_SNAPSHOT_SHA"],
                "RAW_PUBLICATION_SNAPSHOT_SHA": manifest["RAW_PUBLICATION_SNAPSHOT_SHA"],
                "PUBLICATION_MATERIAL_PROVENANCE_SHA": manifest[
                    "PUBLICATION_MATERIAL_PROVENANCE_SHA"
                ],
                "COLLECTED_EVENT_METADATA_SHA": manifest["COLLECTED_EVENT_METADATA_SHA"],
                "DEDUPE_STATE_SHA": manifest["DEDUPE_STATE_SHA"],
                "LIVE_EXACT_SOURCES_ENABLED": manifest["LIVE_EXACT_SOURCES_ENABLED"],
                "LIVE_EXACT_SOURCES_ATTEMPTED": manifest["LIVE_EXACT_SOURCES_ATTEMPTED"],
                "LIVE_EXACT_SOURCES_SUCCESS": manifest["LIVE_EXACT_SOURCES_SUCCESS"],
                "ITEMS_FETCHED": manifest["ITEMS_FETCHED"],
                "ITEMS_WITH_PUBLICATION_MATERIAL": manifest["ITEMS_WITH_PUBLICATION_MATERIAL"],
                "ITEMS_WITHOUT_PUBLICATION_MATERIAL": manifest[
                    "ITEMS_WITHOUT_PUBLICATION_MATERIAL"
                ],
                "ITEMS_NEW": manifest["ITEMS_NEW"],
                "ITEMS_DUPLICATE": manifest["ITEMS_DUPLICATE"],
                "SNAPSHOTS_WRITTEN": manifest["SNAPSHOTS_WRITTEN"],
                "DUPLICATE_SNAPSHOTS": manifest["DUPLICATE_SNAPSHOTS"],
                "NEW_EXACT_EVENTS": manifest["NEW_EXACT_EVENTS"],
                "NEW_HISTORICAL_EXACT_EVENTS": manifest["NEW_HISTORICAL_EXACT_EVENTS"],
                "NEW_FUTURE_METADATA_ONLY_EVENTS": manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"],
                "FUTURE_EVENT_HOLDOUT_USED": manifest["FUTURE_EVENT_HOLDOUT_USED"],
                "FUTURE_EVENT_HOLDOUT_OBSERVED": manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire live official EXACT event metadata.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--source-registry",
        default=DEFAULT_SOURCE_REGISTRY_PATH,
        help="JSON registry of enabled live official EXACT sources.",
    )
    parser.add_argument(
        "--input-events",
        default="artifacts/exact-event-official-source-discovery-v5/events.jsonl",
        help="Existing exact event metadata used only for deterministic dedupe counts.",
    )
    parser.add_argument(
        "--audit-manifest",
        default="artifacts/exact-event-official-source-mechanism-audit-v1/manifest.json",
        help="Verified source-mechanism audit manifest.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional previous dedupe-state.json from an earlier one-shot run.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-live-official-collection-v1",
    )
    parser.add_argument("--created-at", default=None)
    parser.add_argument(
        "--event-origin",
        action="append",
        default=None,
        help="Optionally restrict collection to one event origin such as ISSUER_ORIGINATED.",
    )
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
