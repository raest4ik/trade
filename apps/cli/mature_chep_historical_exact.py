from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.chep_historical_exact_maturation.application import (
    run_chep_historical_exact_maturation,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token


def run(args: argparse.Namespace) -> int:
    manifest = asyncio.run(_run_async(args))
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "INPUT_COLLECTOR_ARTIFACT_SHA": manifest["INPUT_COLLECTOR_ARTIFACT_SHA"],
                "HISTORICAL_COHORT_SHA": manifest["HISTORICAL_COHORT_SHA"],
                "FUTURE_METADATA_COHORT_SHA": manifest["FUTURE_METADATA_COHORT_SHA"],
                "INSTRUMENT_IDENTITY_SHA": manifest["INSTRUMENT_IDENTITY_SHA"],
                "MARKET_ACQUISITION_PROVENANCE_SHA": manifest["MARKET_ACQUISITION_PROVENANCE_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "MATURATION_REPORT_SHA": manifest["MATURATION_REPORT_SHA"],
                "CHEP_HISTORICAL_EVENTS_TOTAL": manifest["CHEP_HISTORICAL_EVENTS_TOTAL"],
                "FUTURE_CHEP_EVENTS": manifest["FUTURE_CHEP_EVENTS"],
                "CHEP_REACTION_READY": manifest["CHEP_REACTION_READY"],
                "CHEP_FEATURE_READY": manifest["CHEP_FEATURE_READY"],
                "CHEP_1M_READY": manifest["CHEP_1M_READY"],
                "CHEP_5M_READY": manifest["CHEP_5M_READY"],
                "CHEP_15M_READY": manifest["CHEP_15M_READY"],
                "CHEP_30M_READY": manifest["CHEP_30M_READY"],
                "CHEP_60M_READY": manifest["CHEP_60M_READY"],
                "FUTURE_EVENT_HOLDOUT_USED": False,
                "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
                "FINAL_DECISION": manifest["FINAL_DECISION"],
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
        return await run_chep_historical_exact_maturation(
            collector_root=Path(args.collector_dir),
            base_dataset_root=Path(args.base_dataset_dir),
            output_root=Path(args.output_dir),
            base_main_sha=args.base_main_sha,
            git_sha=_git_sha(),
            client=client,
            created_at=(
                datetime.fromisoformat(args.created_at) if args.created_at is not None else None
            ),
            extra_cache_roots=tuple(Path(path) for path in args.extra_cache_dir),
            universe_root=Path(args.universe_dir),
        )
    finally:
        if client is not None:
            await client.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mature historical CHEP strict-EXACT live collector rows."
    )
    parser.add_argument(
        "--collector-dir", default="artifacts/exact-event-live-official-collection-v1"
    )
    parser.add_argument(
        "--base-dataset-dir", default="artifacts/exact-event-new-source-maturation-v1"
    )
    parser.add_argument("--output-dir", default="artifacts/chep-historical-exact-maturation-v1")
    parser.add_argument("--universe-dir", default="artifacts/tinvest-market-universe-raw-v1")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--extra-cache-dir", action="append", default=[])
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="Use existing TINVEST_READONLY_TOKEN for bounded CHEP minute candle acquisition.",
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
