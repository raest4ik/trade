from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.chep_security_history_diagnostics.application import (
    run_chep_security_history_diagnostics,
)
from src.chep_security_history_diagnostics.moex import MoexIssClient
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token


def run(args: argparse.Namespace) -> int:
    manifest = asyncio.run(_run_async(args))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


async def _run_async(args: argparse.Namespace) -> dict[str, object]:
    client: TInvestReadOnlyClient | None = None
    moex_client: MoexIssClient | None = None
    if args.live_readonly:
        client = TInvestReadOnlyClient(
            token=load_readonly_token(),
            contour=TInvestContour.READONLY_PRODUCTION,
            max_retries=1,
        )
    if args.moex:
        moex_client = MoexIssClient()
    try:
        return await run_chep_security_history_diagnostics(
            input_root=Path(args.input_dir),
            output_root=Path(args.output_dir),
            base_main_sha=args.base_main_sha,
            git_sha=_git_sha(),
            client=client,
            moex_client=moex_client,
            created_at=(
                datetime.fromisoformat(args.created_at) if args.created_at is not None else None
            ),
        )
    finally:
        if client is not None:
            await client.aclose()
        if moex_client is not None:
            await moex_client.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose CHEP historical security data gaps without mutating canonical rows."
    )
    parser.add_argument(
        "--input-dir",
        default="artifacts/chep-historical-exact-maturation-v1",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/chep-security-history-diagnostics-v1",
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="Use existing TINVEST_READONLY_TOKEN for bounded read-only diagnostics.",
    )
    parser.add_argument(
        "--moex",
        action="store_true",
        help="Use official zero-cost MOEX ISS as diagnostic cross-check only.",
    )
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
