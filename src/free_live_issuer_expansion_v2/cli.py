from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from src.free_live_issuer_expansion_v2.application import (
    provider_factory_from_tinvest_mapping,
    run_free_live_issuer_source_expansion_v2,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import READONLY_TOKEN_ENV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument(
        "--tinvest-mapping",
        type=Path,
        default=Path("artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"),
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    provider_factory = None
    token = os.getenv(READONLY_TOKEN_ENV, "").strip()
    if token:
        async with TInvestReadOnlyClient(
            token=token, contour=TInvestContour.READONLY_PRODUCTION, max_retries=1
        ) as tinvest:
            if args.tinvest_mapping.exists():
                provider_factory = provider_factory_from_tinvest_mapping(
                    mapping_path=args.tinvest_mapping,
                    client=tinvest,
                )
            manifest = await run_free_live_issuer_source_expansion_v2(
                output_root=args.output_root,
                base_main_sha=args.base_main_sha,
                git_sha=args.git_sha,
                provider_factory=provider_factory,
                network_check=not args.no_network,
            )
    else:
        manifest = await run_free_live_issuer_source_expansion_v2(
            output_root=args.output_root,
            base_main_sha=args.base_main_sha,
            git_sha=args.git_sha,
            provider_factory=provider_factory,
            network_check=not args.no_network,
        )
    print(manifest["ARTIFACT_SHA"])


if __name__ == "__main__":
    main()
