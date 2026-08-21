from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_source_diversity_v3.application import (
    build_source_diversity_v3_artifact,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_source_diversity_v3_artifact(
        warmup_root=Path(args.warmup_dir),
        v2_root=Path(args.v2_dir),
        universe_path=Path(args.universe),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
        moex_feed_path=Path(args.moex_feed) if args.moex_feed is not None else None,
        max_new_events=args.max_new_events,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build EXACT event source / issuer diversity v3 artifact."
    )
    parser.add_argument(
        "--warmup-dir",
        default="artifacts/exact-event-market-history-warmup-recovery-v1",
    )
    parser.add_argument("--v2-dir", default="artifacts/exact-event-market-dataset-v2")
    parser.add_argument(
        "--universe",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument("--output-dir", default="artifacts/exact-event-source-diversity-v3")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--moex-feed", default=None)
    parser.add_argument("--max-new-events", type=int, default=100)
    parser.add_argument("--created-at", default=None)
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
