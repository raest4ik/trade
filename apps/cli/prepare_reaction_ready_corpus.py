from __future__ import annotations

import argparse
import asyncio
import json

from apps.cli.historical_news_common import bounded_limit, parse_range_datetime
from src.reaction_ready_corpus.application import (
    PrepareCorpusCommand,
    PrepareReactionReadyCorpus,
)
from src.reaction_ready_corpus.domain import REAL_SOURCE_CODES, UNIVERSE
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            result = await PrepareReactionReadyCorpus(session).execute(
                PrepareCorpusCommand(
                    date_from=parse_range_datetime(args.date_from, end_of_day=False),
                    date_to=parse_range_datetime(args.date_to, end_of_day=True),
                    source_codes=_csv(args.source_codes),
                    tickers=_csv(args.tickers),
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
            )
    finally:
        await engine.dispose()
    print(
        json.dumps(
            {
                "candidate_count": result.candidate_count,
                "analyzed_count": result.analyzed_count,
                "matched_count": result.matched_count,
                "ambiguous_count": result.ambiguous_count,
                "unmatched_count": result.unmatched_count,
                "dry_run": result.dry_run,
                "market_backfill_windows": [window.payload() for window in result.windows],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match and deterministically analyze REAL historical news, then plan MOEX windows."
        )
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=bounded_limit, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-codes", default=",".join(sorted(REAL_SOURCE_CODES)))
    parser.add_argument("--tickers", default=",".join(UNIVERSE))
    return parser


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
