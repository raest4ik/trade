from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_dataset_readiness_audit.application import (
    run_exact_dataset_readiness_audit,
)
from src.exact_dataset_readiness_audit.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_exact_dataset_readiness_audit(
        input_root=Path(args.input_dir),
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
                "CANONICAL_EXACT_EVENTS": manifest["CANONICAL_EXACT_EVENTS"],
                "FEATURE_READY_EVENTS": manifest["FEATURE_READY_EVENTS"],
                "ISSUER_ORIGINATED_FEATURE_READY": manifest["ISSUER_ORIGINATED_FEATURE_READY"],
                "EXCHANGE_ORIGINATED_FEATURE_READY": manifest["EXCHANGE_ORIGINATED_FEATURE_READY"],
                "UNKNOWN_RATE_TOTAL": manifest["UNKNOWN_RATE_TOTAL"],
                "MOEX_RISK_UNKNOWN_RATE": manifest["MOEX_RISK_UNKNOWN_RATE"],
                "READINESS_DECISION": manifest["READINESS_DECISION"],
                "RECOMMENDED_PRIMARY_COHORT": manifest["RECOMMENDED_PRIMARY_COHORT"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit strict-EXACT event+market dataset readiness and composition."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
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
