from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from apps.cli.historical_news_common import bounded_limit
from src.historical_news.infrastructure.reporting import (
    corpus_stats,
    load_corpus_rows,
    write_stats,
)
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            stats = corpus_stats(await load_corpus_rows(session, limit=args.limit))
    finally:
        await engine.dispose()
    stats["live_source_status"] = (
        "CONFIGURED_BY_OPERATOR"
        if args.live_source_configured
        else "LIVE_HISTORICAL_SOURCE_NOT_CONFIGURED"
    )
    write_stats(Path(args.output), stats)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report historical-news corpus quality.")
    parser.add_argument(
        "--output",
        default="artifacts/historical-news-v1/data-quality-report.json",
    )
    parser.add_argument("--limit", type=bounded_limit, default=100_000)
    parser.add_argument(
        "--live-source-configured",
        action="store_true",
        help="Declare that this operator-run used a separately authorized live source.",
    )
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
