from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_security_tradability_eligibility.application import (
    run_security_tradability_eligibility,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_security_tradability_eligibility(
        diagnostic_root=Path(args.diagnostic_dir),
        maturation_root=Path(args.maturation_dir),
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
        description="Build exact-event security tradability eligibility artifact."
    )
    parser.add_argument(
        "--diagnostic-dir",
        default="artifacts/chep-security-history-diagnostics-v1",
    )
    parser.add_argument(
        "--maturation-dir",
        default="artifacts/chep-historical-exact-maturation-v1",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-security-tradability-eligibility-v1",
    )
    parser.add_argument("--base-main-sha", required=True)
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
