from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.moex_target_source_expansion_v4.application import (
    ARTIFACT_VERSION,
    DEFAULT_OUTPUT_ROOT,
    run_moex_target_source_expansion_v4,
)


def run(args: argparse.Namespace) -> int:
    created_at = _created_at(args.created_at) if args.created_at else None
    manifest = run_moex_target_source_expansion_v4(
        output_root=Path(args.output_root),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        network_check=not args.no_network,
        created_at=created_at,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=ARTIFACT_VERSION)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--created-at", default=None, help=argparse.SUPPRESS)
    return parser


def _created_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("created-at must include timezone")
    return parsed.astimezone(UTC)


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
