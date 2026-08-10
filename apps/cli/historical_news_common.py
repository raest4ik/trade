from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime, time

from src.historical_news.application.ports import HistoricalNewsSourceClient
from src.historical_news.application.use_cases import (
    IngestHistoricalNews,
    IngestHistoricalNewsCommand,
)
from src.historical_news.domain.entities import HistoricalNewsSource
from src.historical_news.domain.enums import (
    ContentStoragePolicy,
    HistoricalNewsSourceKind,
)
from src.historical_news.infrastructure.repositories import SqlAlchemyHistoricalNewsRepository
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def ingest_from_client(
    args: argparse.Namespace,
    *,
    client: HistoricalNewsSourceClient,
    source_kind: HistoricalNewsSourceKind,
    feed_url: str | None = None,
) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        async with create_session_factory(engine)() as session:
            source = HistoricalNewsSource.create(
                source_code=args.source_code,
                source_kind=source_kind,
                content_storage_policy=ContentStoragePolicy(args.storage_policy),
                source_timezone=args.source_timezone,
                feed_url=feed_url,
            )
            result = await IngestHistoricalNews(
                repository=SqlAlchemyHistoricalNewsRepository(session),
                news_repository=SqlAlchemyNewsRepository(session),
                instrument_repository=SqlAlchemyInstrumentRepository(session),
                source_client=client,
            ).execute(
                source=source,
                command=IngestHistoricalNewsCommand(
                    date_from=parse_range_datetime(args.date_from, end_of_day=False),
                    date_to=parse_range_datetime(args.date_to, end_of_day=True),
                    limit=args.limit,
                    max_pages=args.max_pages,
                    dry_run=args.dry_run,
                    match_instruments=args.match_instruments,
                ),
            )
    finally:
        await engine.dispose()
    print(json.dumps(asdict(result), default=str, sort_keys=True))
    return 0


def add_ingestion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-code", required=True)
    parser.add_argument(
        "--storage-policy", choices=[item.value for item in ContentStoragePolicy], required=True
    )
    parser.add_argument("--source-timezone")
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=bounded_limit, default=1000)
    parser.add_argument("--max-pages", type=bounded_pages, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--match-instruments", action="store_true")


def parse_range_datetime(value: str, *, end_of_day: bool) -> datetime:
    normalized = value.strip()
    try:
        if len(normalized) == 10:
            parsed_date = datetime.fromisoformat(normalized).date()
            return datetime.combine(
                parsed_date,
                time.max if end_of_day else time.min,
                tzinfo=UTC,
            )
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("datetime must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime range must include timezone")
    return parsed.astimezone(UTC)


def bounded_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100_000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100000")
    return parsed


def bounded_pages(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 1_000:
        raise argparse.ArgumentTypeError("max-pages must be between 1 and 1000")
    return parsed
