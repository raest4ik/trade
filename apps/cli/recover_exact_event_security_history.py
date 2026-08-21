from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_security_history_recovery.application import (
    run_security_history_recovery,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token


def run(args: argparse.Namespace) -> int:
    manifest = asyncio.run(_run_async(args))
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "RECOVERY_COHORT_SHA": manifest["RECOVERY_COHORT_SHA"],
                "RECOVERY_COHORT_TOTAL": manifest["RECOVERY_COHORT_TOTAL"],
                "RECOVERY_SUCCESS_COUNT": manifest["RECOVERY_SUCCESS_COUNT"],
                "RECOVERY_BLOCKED_COUNT": manifest["RECOVERY_BLOCKED_COUNT"],
                "CACHE_ACQUISITION_STATUS": manifest["CACHE_ACQUISITION_STATUS"],
                "CACHE_DEDUPE": manifest["CACHE_DEDUPE"],
                "LEAKAGE_CHECK": manifest["LEAKAGE_CHECK"],
                "MODEL_TRAINING_PERFORMED": False,
                "TEST_OUTCOME_USED": False,
                "FUTURE_EVENT_HOLDOUT_USED": False,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


async def _run_async(args: argparse.Namespace) -> dict[str, object]:
    client: TInvestReadOnlyClient | None = None
    if args.live_readonly:
        client = TInvestReadOnlyClient(
            token=load_readonly_token(),
            contour=TInvestContour.READONLY_PRODUCTION,
            max_retries=1,
        )
    try:
        return await run_security_history_recovery(
            pr36_root=Path(args.pr36_dir),
            diagnostics_root=Path(args.diagnostics_dir),
            output_root=Path(args.output_dir),
            base_main_sha=args.base_main_sha,
            git_sha=_git_sha(),
            client=client,
            created_at=(
                datetime.fromisoformat(args.created_at) if args.created_at is not None else None
            ),
            extra_cache_roots=tuple(Path(path) for path in args.extra_cache_dir),
        )
    finally:
        if client is not None:
            await client.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover bounded T-Invest security minute history for PR37 EXACT events."
    )
    parser.add_argument("--pr36-dir", default="artifacts/exact-event-new-source-maturation-v1")
    parser.add_argument(
        "--diagnostics-dir", default="artifacts/exact-event-security-history-diagnostics-v1"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/exact-event-security-history-recovery-v1"
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--extra-cache-dir", action="append", default=[])
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="Use existing TINVEST_READONLY_TOKEN for bounded read-only candle acquisition.",
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
