from __future__ import annotations

import argparse
import asyncio
from uuid import UUID

from sqlalchemy import select

from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.exceptions import ReactionApplicationError
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    processed = 0
    failed = 0
    try:
        async with session_factory() as session:
            if args.news_id is not None:
                news_ids = [UUID(args.news_id)]
            else:
                result = await session.execute(
                    select(NewsItemRecord.id)
                    .where(
                        NewsItemRecord.publication_timestamp_quality
                        == PublicationTimestampQuality.EXACT.value
                    )
                    .order_by(NewsItemRecord.published_at, NewsItemRecord.id)
                    .limit(args.limit)
                )
                news_ids = list(result.scalars())
            if args.dry_run:
                print(f"eligible_news={len(news_ids)} dry_run=true")
                return 0
            calculator = CalculateNewsMarketReactions(
                news_repository=SqlAlchemyNewsRepository(session),
                instrument_repository=SqlAlchemyInstrumentRepository(session),
                market_data_repository=SqlAlchemyMarketDataRepository(session),
                reaction_repository=SqlAlchemyReactionRepository(session),
            )
            for news_id in news_ids:
                try:
                    result = await calculator.execute(news_id)
                except ReactionApplicationError as exc:
                    failed += 1
                    print(f"news_id={news_id} status=FAILED error={exc}")
                    continue
                processed += 1
                print(f"news_id={news_id} status=SUCCEEDED reaction_rows={len(result.reactions)}")
    except ValueError as exc:
        print(str(exc))
        return 1
    finally:
        await engine.dispose()
    print(f"processed={processed} failed={failed}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute benchmark-adjusted reactions for exact-timestamp news."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--news-id")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=_bounded_limit, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _bounded_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 1000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 1000")
    return parsed


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
