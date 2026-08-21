from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_new_source_maturation.application import run_new_source_maturation


def run(args: argparse.Namespace) -> int:
    manifest = run_new_source_maturation(
        previous_root=Path(args.previous_dir),
        current_root=Path(args.current_dir),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mature PR35 new EXACT source events from local T-Invest market cache."
    )
    parser.add_argument(
        "--previous-dir",
        default="artifacts/exact-event-market-history-warmup-recovery-v1",
    )
    parser.add_argument("--current-dir", default="artifacts/exact-event-source-diversity-v3")
    parser.add_argument("--output-dir", default="artifacts/exact-event-new-source-maturation-v1")
    parser.add_argument("--base-main-sha", required=True)
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
