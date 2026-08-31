from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.timezone_verified_issuer_exact_source_discovery.application import (
    run_timezone_verified_issuer_exact_source_discovery,
)
from src.timezone_verified_issuer_exact_source_discovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_ISSUER_DIVERSITY_ROOT,
    DEFAULT_READINESS_AUDIT_ROOT,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=Path(args.readiness_dir),
        issuer_diversity_root=Path(args.issuer_diversity_dir),
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
                "DOMAINS_AUDITED": manifest["DOMAINS_AUDITED"],
                "SOURCES_AUDITED": manifest["SOURCES_AUDITED"],
                "STRICT_EXACT_HISTORICAL_READY": manifest["STRICT_EXACT_HISTORICAL_READY"],
                "VERIFIED_HISTORICAL_ITEMS": manifest["VERIFIED_HISTORICAL_ITEMS"],
                "FINAL_DECISION": manifest["FINAL_DECISION"],
                "TINVEST_REQUESTS": manifest["TINVEST_REQUESTS"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit timezone-verified issuer exact historical source candidates v2."
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--readiness-dir", default=DEFAULT_READINESS_AUDIT_ROOT)
    parser.add_argument("--issuer-diversity-dir", default=DEFAULT_ISSUER_DIVERSITY_ROOT)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
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
