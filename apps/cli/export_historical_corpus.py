from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from apps.cli.historical_news_common import bounded_limit
from src.historical_news.infrastructure.reporting import load_corpus_rows, write_corpus
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            rows = await load_corpus_rows(session, limit=args.limit)
    finally:
        await engine.dispose()
    count = write_corpus(
        Path(args.output),
        rows,
        reaction_ready_only=args.reaction_ready,
        include_content=args.include_content,
    )
    print(f"schema_version=historical-news-corpus-v1 rows_written={count} output={args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the auditable historical-news corpus.")
    parser.add_argument("--output", default="artifacts/historical-news-v1/corpus.jsonl")
    parser.add_argument("--reaction-ready", action="store_true")
    parser.add_argument("--include-content", action="store_true")
    parser.add_argument("--limit", type=bounded_limit, default=100_000)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
