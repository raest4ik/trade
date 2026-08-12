from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from src.market_baseline.application import DEFAULT_TICKERS, acquire_market_history
from src.market_baseline.domain import SplitConfig, build_dataset, date_grouped_temporal_split
from src.market_baseline.reporting import write_market_baseline_artifacts
from src.market_data.infrastructure.moex_client import MoexIssClient
from src.shared.config.settings import get_settings


async def run(args: argparse.Namespace) -> int:
    date_from = date.fromisoformat(args.date_from)
    date_till = date.fromisoformat(args.date_to)
    tickers = tuple(item.strip().upper() for item in args.tickers.split(",") if item.strip())
    settings = get_settings()
    async with MoexIssClient(
        base_url=settings.moex_iss_base_url,
        timeout_seconds=settings.moex_http_timeout_seconds,
        max_retries=settings.moex_http_max_retries,
        max_pages=settings.moex_http_max_pages,
        user_agent=settings.moex_http_user_agent,
    ) as client:
        acquired = await acquire_market_history(
            client,
            tickers=tickers,
            date_from=date_from,
            date_till=date_till,
            max_concurrency=args.max_concurrency,
        )
    rejected = sum(
        int(item["rows_rejected"])
        for item in cast("dict[str, dict[str, Any]]", acquired.acquisition["series"]).values()
    )
    dataset = build_dataset(
        acquired.security_bars,
        acquired.benchmark_bars,
        provider_rejected_rows=rejected,
    )
    split = date_grouped_temporal_split(
        dataset.features,
        SplitConfig(
            purge_sessions=args.purge_sessions,
            embargo_sessions=args.embargo_sessions,
        ),
    )
    paths = write_market_baseline_artifacts(
        Path(args.output_dir),
        result=dataset,
        split=split,
        acquisition=acquired.acquisition,
        git_sha=_git_sha(),
        event_daily_feature_ready=_event_daily_feature_ready(Path(args.event_readiness)),
    )
    print(
        json.dumps(
            {
                "source_rows": dataset.source_row_count,
                "feature_ready": len(dataset.features),
                "ticker_count": len(dataset.source_ticker_distribution),
                "split_counts": split.counts(),
                "purged_rows": len(split.purged_row_ids),
                "embargoed_rows": len(split.embargoed_row_ids),
                "dataset_sha": dataset.dataset_sha256,
                "split_sha": split.split_sha256,
                "model_trained": False,
                "artifacts": {key: str(path) for key, path in paths.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Build the zero-cost leakage-safe daily MOEX market baseline dataset."
    )
    parser.add_argument("--from", dest="date_from", default="2000-01-01")
    parser.add_argument("--to", dest="date_to", default=yesterday.isoformat())
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--purge-sessions", type=int, default=1)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/market-baseline-v1")
    parser.add_argument(
        "--event-readiness",
        default="artifacts/free-daily-historical-v1/readiness.json",
    )
    return parser


def _event_daily_feature_ready(path: Path) -> int:
    if not path.exists():
        return 0
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    return int(payload.get("feature_ready", 0))


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
