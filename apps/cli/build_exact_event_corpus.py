from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.exact_event_corpus.application import (
    build_exact_dataset,
    build_exact_source_registry,
)
from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_corpus.sources import (
    ExactAppStateProfile,
    acquire_exact_json_pages,
    load_exact_app_state,
)
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.news.infrastructure.models import NewsItemRecord
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient

DEFAULT_OUTPUT = Path("artifacts/exact-event-market-dataset-v1")


async def run(args: argparse.Namespace) -> int:
    token = os.environ.get("TINVEST_READONLY_TOKEN", "")
    if not token:
        raise ValueError("TINVEST_READONLY_TOKEN_REQUIRED")
    mapping_path = Path(args.instrument_mapping)
    registry = build_exact_source_registry(mapping_path, Path(args.previous_source_registry))
    by_ticker = {item.ticker: item for item in registry}
    events = await _load_existing_exact_events(registry)
    magnit_profile = ExactAppStateProfile(
        source_code="MAGNIT_OFFICIAL_JSON_EXACT",
        ticker="MGNT",
        issuer=by_ticker["MGNT"].issuer,
        instrument_uid=by_ticker["MGNT"].instrument_uid,
        base_url="https://www.magnit.com/",
        timestamp_field="date",
        title_field="name",
        url_field="link",
    )
    profiles = (
        (
            Path(args.nornickel_cache),
            ExactAppStateProfile(
                source_code="NORNICKEL_OFFICIAL_APP_STATE_EXACT",
                ticker="GMKN",
                issuer=by_ticker["GMKN"].issuer,
                instrument_uid=by_ticker["GMKN"].instrument_uid,
                base_url="https://nornickel.ru/",
                timestamp_field="activeFrom",
                title_field="name",
                url_field="detailPageUrl",
                identity_field="code",
                reject_source_local_midnight=True,
            ),
        ),
        (
            Path(args.magnit_cache),
            magnit_profile,
        ),
    )
    for path, profile in profiles:
        events.extend(load_exact_app_state(path, profile=profile))
    events.extend(
        await acquire_exact_json_pages(
            profile=magnit_profile,
            url_template="https://www.magnit.com/ru/api/news?page={page}",
            date_from=date.fromisoformat(args.date_from),
            date_to=date.fromisoformat(args.date_to),
            page_limit=args.source_page_limit,
            item_limit=args.source_item_limit,
            cache_dir=Path(args.output) / "raw-source-cache" / "MAGNIT_OFFICIAL_JSON",
        )
    )
    async with TInvestReadOnlyClient(
        token=token,
        contour=TInvestContour.READONLY_PRODUCTION,
    ) as client:
        report = await build_exact_dataset(
            events=events,
            registry=registry,
            client=client,
            output_dir=Path(args.output),
            candle_cache_dir=Path(args.output) / "raw-minute-cache",
            benchmark_instrument_uid=_instrument_uid(mapping_path, "IMOEX"),
            git_sha=_git_sha(),
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


async def _load_existing_exact_events(
    registry: tuple[Any, ...],
) -> list[ExactEvent]:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            return await _query_existing_exact_events(session, registry)
    finally:
        await engine.dispose()


async def _query_existing_exact_events(
    session: AsyncSession,
    registry: tuple[Any, ...],
) -> list[ExactEvent]:
    result = await session.execute(
        select(HistoricalNewsCandidateRecord, HistoricalNewsSourceRecord, NewsItemRecord)
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .join(NewsItemRecord, NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id)
        .where(
            HistoricalNewsSourceRecord.source_code.in_(
                ("ROSNEFT_PRESS_RELEASES_RSS", "YANDEX_IR_PRESS_RELEASES_RSS")
            )
        )
        .order_by(NewsItemRecord.published_at, NewsItemRecord.id)
    )
    by_ticker = {item.ticker: item for item in registry}
    source_ticker = {
        "ROSNEFT_PRESS_RELEASES_RSS": "ROSN",
        "YANDEX_IR_PRESS_RELEASES_RSS": "YDEX",
    }
    events: list[ExactEvent] = []
    for candidate, source, news in result.all():
        ticker = source_ticker[source.source_code]
        identity = by_ticker[ticker]
        created = ExactEvent.create(
            source_code=source.source_code,
            source_item_id=candidate.source_item_id,
            canonical_url=candidate.source_url,
            ticker=ticker,
            issuer=identity.issuer,
            instrument_uid=identity.instrument_uid,
            title=news.title,
            publication_timestamp_raw=candidate.original_timestamp_text,
            publication_timestamp_utc=news.published_at,
            timestamp_source_field="RSS item pubDate",
            storage_policy=candidate.content_storage_policy,
            event_id=news.id,
        )
        events.append(replace(created, title=news.raw_content))
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the exact-time event corpus without model training or evaluation."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--date-from", default="2025-01-01")
    parser.add_argument("--date-to", default=date.today().isoformat())
    parser.add_argument("--source-page-limit", type=int, default=50)
    parser.add_argument("--source-item-limit", type=int, default=400)
    parser.add_argument(
        "--instrument-mapping",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument(
        "--previous-source-registry",
        default="artifacts/event-market-predictive-dataset-v2/source-registry.jsonl",
    )
    parser.add_argument(
        "--nornickel-cache",
        default=(
            "artifacts/event-market-predictive-dataset-v2/raw-cache/"
            "NORNICKEL_NEWS_ARCHIVE_DATE_ONLY/1.html"
        ),
    )
    parser.add_argument(
        "--magnit-cache",
        default=(
            "artifacts/event-market-predictive-dataset-v2/raw-cache/"
            "MAGNIT_PRESS_RELEASE_ARCHIVE_DATE_ONLY/1.html"
        ),
    )
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _instrument_uid(mapping_path: Path, ticker: str) -> str:
    payload = cast("dict[str, Any]", json.loads(mapping_path.read_text(encoding="utf-8")))
    instruments = cast("list[dict[str, Any]]", payload["instruments"])
    matches = [str(item["instrument_uid"]) for item in instruments if item["ticker"] == ticker]
    if len(matches) != 1:
        raise ValueError(f"INSTRUMENT_IDENTITY_MISSING_OR_AMBIGUOUS:{ticker}")
    return matches[0]


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
