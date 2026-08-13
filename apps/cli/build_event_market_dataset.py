from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.event_market_dataset.application import (
    acquire_new_events,
    build_dataset,
    build_source_registry,
)
from src.event_market_dataset.domain import AcquiredEvent, EventSourceRegistryEntry
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.reaction_ready_corpus.domain import REAL_SOURCE_CODES
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory

DEFAULT_OUTPUT = Path("artifacts/event-market-predictive-dataset-v1")


async def run(args: argparse.Namespace) -> int:
    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    if date_from > date_to:
        raise ValueError("--date-from must not be after --date-to")
    mapping_path = Path(args.instrument_mapping)
    registry = build_source_registry(mapping_path, checked_on=date.today())
    output_dir = Path(args.output)
    acquired, source_errors = await acquire_new_events(
        registry,
        date_from=date_from,
        date_to=date_to,
        per_source_limit=args.per_source_limit,
        cache_dir=output_dir / "raw-cache",
        allow_partial_sources=args.allow_partial_sources,
    )
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            existing_events, existing_texts = await _load_existing_events(session, registry)
    finally:
        await engine.dispose()
    report = build_dataset(
        acquired_events=acquired,
        existing_events=existing_events,
        existing_texts=existing_texts,
        old_corpus_path=Path(args.old_corpus),
        old_manifest_path=Path(args.old_manifest),
        market_feature_path=Path(args.market_features),
        market_manifest_path=Path(args.market_manifest),
        raw_series_dir=Path(args.raw_series_dir),
        instrument_mapping_path=mapping_path,
        output_dir=output_dir,
        source_registry=registry,
        source_errors=source_errors,
        git_sha=_git_sha(),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


async def _load_existing_events(
    session: AsyncSession,
    registry: tuple[EventSourceRegistryEntry, ...],
) -> tuple[list[AcquiredEvent], dict[str, str]]:
    result = await session.execute(
        select(
            HistoricalNewsCandidateRecord,
            HistoricalNewsSourceRecord,
            NewsItemRecord,
        )
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .join(NewsItemRecord, NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id)
        .where(HistoricalNewsSourceRecord.source_code.in_(REAL_SOURCE_CODES))
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsCandidateRecord.source_item_id,
        )
    )
    registry_by_ticker = {item.ticker: item for item in registry}
    source_ticker = {
        "ROSNEFT_PRESS_RELEASES_RSS": "ROSN",
        "YANDEX_IR_PRESS_RELEASES_RSS": "YDEX",
    }
    events: list[AcquiredEvent] = []
    texts: dict[str, str] = {}
    for candidate, source, news in result.all():
        ticker = source_ticker[source.source_code]
        instrument = registry_by_ticker[ticker]
        created = AcquiredEvent.create(
            source_code=source.source_code,
            source_item_id=candidate.source_item_id,
            source_url=candidate.source_url,
            ticker=ticker,
            issuer_name=instrument.issuer_name,
            instrument_uid=instrument.instrument_uid,
            figi=instrument.figi,
            title=news.title,
            publication_date=news.published_at.date(),
            published_at=news.published_at,
            timestamp_quality=PublicationTimestampQuality.EXACT,
            storage_policy=candidate.content_storage_policy,
        )
        events.append(replace(created, event_id=news.id))
        texts[str(news.id)] = news.raw_content
    return events, texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe, event-driven predictive dataset without training."
    )
    parser.add_argument("--date-from", default="2022-01-01")
    parser.add_argument("--date-to", default=date.today().isoformat())
    parser.add_argument("--per-source-limit", type=int, default=2000)
    parser.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help="Continue after an explicit source error and record it in provenance.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--old-corpus", default="artifacts/reaction-ready-corpus-v3/corpus.jsonl")
    parser.add_argument(
        "--old-manifest", default="artifacts/reaction-ready-corpus-v3/manifest.json"
    )
    parser.add_argument(
        "--instrument-mapping",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument(
        "--market-features",
        default="artifacts/tinvest-market-universe-features-v1/features.jsonl",
    )
    parser.add_argument(
        "--market-manifest",
        default="artifacts/tinvest-market-universe-features-v1/dataset-manifest.json",
    )
    parser.add_argument(
        "--raw-series-dir", default="artifacts/tinvest-market-universe-raw-v1/series"
    )
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
