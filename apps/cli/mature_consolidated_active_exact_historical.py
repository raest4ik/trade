from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.consolidated_active_exact_historical_maturation.application import (
    run_consolidated_active_exact_historical_maturation,
)
from src.consolidated_active_exact_historical_maturation.domain import (
    ARTIFACT_VERSION,
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_UNIVERSE_ROOT,
    DEFAULT_V1_ARTIFACT_ROOT,
    DEFAULT_V2_ARTIFACT_ROOT,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token


def run(args: argparse.Namespace) -> int:
    manifest = asyncio.run(_run_async(args))
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "INPUT_V1_ARTIFACT_SHA": manifest["INPUT_V1_ARTIFACT_SHA"],
                "INPUT_V2_ARTIFACT_SHA": manifest["INPUT_V2_ARTIFACT_SHA"],
                "MATURATION_COHORT_SHA": manifest["MATURATION_COHORT_SHA"],
                "FUTURE_EXCLUDED_COHORT_SHA": manifest["FUTURE_EXCLUDED_COHORT_SHA"],
                "INSTRUMENT_IDENTITY_SHA": manifest["INSTRUMENT_IDENTITY_SHA"],
                "MARKET_ACQUISITION_PROVENANCE_SHA": manifest["MARKET_ACQUISITION_PROVENANCE_SHA"],
                "MATURATION_RESULT_SHA": manifest["MATURATION_RESULT_SHA"],
                "OUTPUT_DATASET_SHA": manifest["OUTPUT_DATASET_SHA"],
                "V1_HISTORICAL_INPUT": manifest["V1_HISTORICAL_INPUT"],
                "V2_HISTORICAL_INPUT": manifest["V2_HISTORICAL_INPUT"],
                "DEDUPED_HISTORICAL_INPUT": manifest["DEDUPED_HISTORICAL_INPUT"],
                "MARKET_ELIGIBLE_INPUT": manifest["MARKET_ELIGIBLE_INPUT"],
                "NEW_REACTION_READY": manifest["NEW_REACTION_READY"],
                "NEW_FEATURE_READY": manifest["NEW_FEATURE_READY"],
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
        return await run_consolidated_active_exact_historical_maturation(
            v1_root=Path(args.v1_dir),
            v2_root=Path(args.v2_dir),
            base_dataset_root=Path(args.base_dataset_dir),
            output_root=Path(args.output_dir),
            base_main_sha=args.base_main_sha,
            git_sha=_git_sha(),
            live_registry_path=Path(args.live_registry),
            universe_root=Path(args.universe_dir),
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
        description="Mature historical active strict-EXACT breadth events from v1 and v2."
    )
    parser.add_argument("--v1-dir", default=DEFAULT_V1_ARTIFACT_ROOT)
    parser.add_argument("--v2-dir", default=DEFAULT_V2_ARTIFACT_ROOT)
    parser.add_argument("--base-dataset-dir", default=DEFAULT_BASE_DATASET_ROOT)
    parser.add_argument("--live-registry", default=DEFAULT_LIVE_REGISTRY_PATH)
    parser.add_argument("--universe-dir", default=DEFAULT_UNIVERSE_ROOT)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--extra-cache-dir", action="append", default=[])
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="Use existing TINVEST_READONLY_TOKEN for bounded minute candle acquisition.",
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
