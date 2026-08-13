from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token
from src.tinvest_market_universe.application import expand_universe


async def run(args: argparse.Namespace) -> int:
    token = load_readonly_token()
    async with TInvestReadOnlyClient(
        token=token,
        contour=TInvestContour.READONLY_PRODUCTION,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        result = await expand_universe(
            client,
            raw_dir=Path(args.raw_dir),
            feature_dir=Path(args.output_dir),
            baseline_raw_dir=Path(args.baseline_raw_dir),
            date_from=date.fromisoformat(args.date_from),
            date_to=date.fromisoformat(args.date_to),
            git_sha=_git_sha(),
        )
    print(json.dumps(result.report, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or incrementally update the read-only T-Invest equity universe."
    )
    parser.add_argument("--from", dest="date_from", default="1970-01-01")
    parser.add_argument(
        "--to", dest="date_to", default=(date.today() - timedelta(days=1)).isoformat()
    )
    parser.add_argument("--raw-dir", default="artifacts/tinvest-market-universe-raw-v1")
    parser.add_argument("--output-dir", default="artifacts/tinvest-market-universe-features-v1")
    parser.add_argument("--baseline-raw-dir", default="artifacts/tinvest-market-raw-v1")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
