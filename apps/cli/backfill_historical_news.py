from __future__ import annotations

import argparse
import asyncio

from apps.cli.historical_news_common import add_ingestion_arguments, ingest_from_client
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.infrastructure.issuer_feed import IssuerFeedNewsSource


async def run(args: argparse.Namespace) -> int:
    source_kind = HistoricalNewsSourceKind(args.source_kind)
    adapter = IssuerFeedNewsSource(
        feed_url=args.feed_url,
        source_kind=source_kind,
        content_storage_policy=ContentStoragePolicy(args.storage_policy),
        source_timezone=args.source_timezone,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        max_items=args.limit,
        user_agent=args.user_agent,
        min_request_interval_seconds=args.min_request_interval,
    )
    try:
        return await ingest_from_client(
            args,
            client=adapter,
            source_kind=source_kind,
            feed_url=args.feed_url,
        )
    finally:
        await adapter.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill a bounded issuer-owned HTTPS RSS or Atom feed."
    )
    parser.add_argument("--feed-url", required=True)
    parser.add_argument(
        "--source-kind",
        choices=[
            HistoricalNewsSourceKind.ISSUER_RSS.value,
            HistoricalNewsSourceKind.ISSUER_ATOM.value,
        ],
        required=True,
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--min-request-interval", type=float, default=0.25)
    parser.add_argument("--user-agent", default="trade-ai-historical-news/1.0")
    add_ingestion_arguments(parser)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
