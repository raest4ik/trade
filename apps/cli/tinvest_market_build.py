from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from src.tinvest_market.application import acquire_history, validate_connectivity
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token, load_sandbox_token
from src.tinvest_market.domain import SECURITY_TICKERS, SplitConfig, build_dataset, temporal_split
from src.tinvest_market.policy import execution_safety
from src.tinvest_market.reporting import write_feature_artifacts


async def run(args: argparse.Namespace) -> int:
    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    tickers = tuple(item.strip().upper() for item in args.tickers.split(",") if item.strip())
    readonly_token = load_readonly_token()
    async with TInvestReadOnlyClient(
        token=readonly_token,
        contour=TInvestContour.READONLY_PRODUCTION,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        connectivity = await validate_connectivity(client)
        acquired = await acquire_history(
            client,
            raw_dir=Path(args.raw_dir),
            date_from=date_from,
            date_to=date_to,
            tickers=tickers,
            git_sha=_git_sha(),
        )
    sandbox = (
        await _sandbox_connectivity(args) if args.check_sandbox else {"status": "NOT_REQUESTED"}
    )
    result = build_dataset(acquired.security_bars, acquired.benchmark_bars)
    split = temporal_split(
        result.features,
        SplitConfig(purge_sessions=args.purge_sessions, embargo_sessions=args.embargo_sessions),
    )
    paths = write_feature_artifacts(
        Path(args.output_dir),
        result=result,
        split=split,
        acquisition_manifest=acquired.manifest,
        git_sha=_git_sha(),
        event_daily_feature_ready=_event_daily_feature_ready(Path(args.event_readiness)),
        moex_targets_path=Path(args.moex_targets) if args.moex_targets else None,
    )
    print(
        json.dumps(
            {
                "READONLY_AUTH": connectivity["auth"],
                "SANDBOX_AUTH": sandbox["status"],
                "raw_rows": result.raw_rows + result.benchmark_rows,
                "feature_ready": len(result.features),
                "ticker_count": len(result.ticker_distribution),
                "split_counts": split.counts(),
                "purged_rows": len(split.purged_row_ids),
                "embargoed_rows": len(split.embargoed_row_ids),
                "dataset_sha": result.dataset_sha,
                "execution_safety": execution_safety(),
                "model_trained": False,
                "backtest_executed": False,
                "artifacts": {key: str(path) for key, path in paths.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


async def _sandbox_connectivity(args: argparse.Namespace) -> dict[str, object]:
    token = load_sandbox_token()
    async with TInvestReadOnlyClient(
        token=token,
        contour=TInvestContour.SANDBOX_READONLY_CONNECTIVITY,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        result = await validate_connectivity(client)
    return {"status": "AUTH_OK", "endpoint_isolated": True, "orders_executed": False, **result}


def build_parser() -> argparse.ArgumentParser:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Build a private T-Invest read-only daily market dataset."
    )
    parser.add_argument("--from", dest="date_from", default="1970-01-01")
    parser.add_argument("--to", dest="date_to", default=yesterday.isoformat())
    parser.add_argument("--tickers", default=",".join(SECURITY_TICKERS))
    parser.add_argument("--raw-dir", default="artifacts/tinvest-market-raw-v1")
    parser.add_argument("--output-dir", default="artifacts/tinvest-market-baseline-features-v1")
    parser.add_argument("--moex-targets", default="artifacts/market-baseline-v1/targets.jsonl")
    parser.add_argument(
        "--event-readiness", default="artifacts/free-daily-historical-v1/readiness.json"
    )
    parser.add_argument("--purge-sessions", type=int, default=1)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--check-sandbox", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _event_daily_feature_ready(path: Path) -> int:
    if not path.exists():
        return 0
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    return int(payload.get("feature_ready", payload.get("daily_feature_ready", 0)))


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
