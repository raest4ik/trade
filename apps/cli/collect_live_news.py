from __future__ import annotations

import argparse
import asyncio

from apps.cli.historical_news_common import (
    bounded_pages,
    ingest_from_client,
    parse_range_datetime,
)
from src.free_historical_data.domain import MAX_PILOT_ITEMS
from src.free_historical_data.registry import compliant_exact_audits
from src.historical_news.domain.enums import HistoricalNewsSourceKind
from src.historical_news.infrastructure.issuer_feed import IssuerFeedNewsSource


def _limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_PILOT_ITEMS:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {MAX_PILOT_ITEMS}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    sources = {audit.source_code: audit for audit in compliant_exact_audits()}
    parser = argparse.ArgumentParser(
        description="Incrementally collect accepted zero-cost issuer RSS without model inference."
    )
    parser.add_argument("--source-code", choices=sorted(sources), required=True)
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=_limit, default=100)
    parser.add_argument("--max-pages", type=bounded_pages, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--min-request-interval", type=float, default=0.5)
    parser.add_argument("--user-agent", default="trade-ai-zero-cost-live/1.0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--match-instruments", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> int:
    sources = {audit.source_code: audit for audit in compliant_exact_audits()}
    audit = sources[args.source_code]
    adapter = IssuerFeedNewsSource(
        feed_url=audit.source_url,
        source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
        content_storage_policy=audit.storage_policy,
        source_timezone=None,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        max_items=args.limit,
        user_agent=args.user_agent,
        min_request_interval_seconds=args.min_request_interval,
    )
    args.storage_policy = audit.storage_policy.value
    args.source_timezone = None
    parse_range_datetime(args.date_from, end_of_day=False)
    parse_range_datetime(args.date_to, end_of_day=True)
    try:
        return await ingest_from_client(
            args,
            client=adapter,
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            feed_url=audit.source_url,
        )
    finally:
        await adapter.aclose()


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
