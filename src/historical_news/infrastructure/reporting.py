from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.news.domain.enums import PublicationTimestampQuality

CORPUS_SCHEMA_VERSION = "historical-news-corpus-v1"


async def load_corpus_rows(session: AsyncSession, *, limit: int = 100_000) -> list[dict[str, Any]]:
    result = await session.execute(
        select(HistoricalNewsCandidateRecord, HistoricalNewsSourceRecord)
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsCandidateRecord.source_item_id,
        )
        .limit(limit)
    )
    candidate_pairs = result.all()
    news_ids = [
        candidate.imported_news_id for candidate, _ in candidate_pairs if candidate.imported_news_id
    ]
    matches_by_news: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
    if news_ids:
        matches = await session.execute(
            select(NewsInstrumentMatchRecord, InstrumentRecord)
            .join(InstrumentRecord, InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id)
            .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
            .order_by(InstrumentRecord.ticker)
        )
        for match, instrument in matches.all():
            matches_by_news[match.news_id].append(
                {
                    "instrument_id": str(instrument.id),
                    "ticker": instrument.ticker,
                    "confidence": match.confidence,
                    "is_ambiguous": match.is_ambiguous,
                    "matcher_version": match.matcher_version,
                }
            )
    return [
        _corpus_row(candidate, source, matches_by_news.get(candidate.imported_news_id, []))
        for candidate, source in candidate_pairs
    ]


def _corpus_row(
    candidate: HistoricalNewsCandidateRecord,
    source: HistoricalNewsSourceRecord,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    reaction_ready = (
        candidate.imported_news_id is not None
        and candidate.publication_timestamp_quality == PublicationTimestampQuality.EXACT.value
        and bool(matches)
        and not any(bool(match["is_ambiguous"]) for match in matches)
    )
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "candidate_id": str(candidate.id),
        "news_id": None if candidate.imported_news_id is None else str(candidate.imported_news_id),
        "source_code": source.source_code,
        "source_kind": source.source_kind,
        "source_item_id": candidate.source_item_id,
        "source_url": candidate.source_url,
        "title": candidate.title,
        "published_at": _iso(candidate.source_published_at),
        "original_timestamp_text": candidate.original_timestamp_text,
        "source_timezone": candidate.source_timezone,
        "timestamp_quality": candidate.publication_timestamp_quality,
        "fetched_at": _iso(candidate.fetched_at),
        "ingestion_run_id": str(candidate.ingestion_run_id),
        "status": candidate.status,
        "storage_policy": candidate.content_storage_policy,
        "content_hash": candidate.content_hash,
        "exact_content_duplicate": candidate.exact_content_duplicate,
        "corrects_source_item_id": candidate.corrects_source_item_id,
        "supersedes_candidate_id": (
            None
            if candidate.supersedes_candidate_id is None
            else str(candidate.supersedes_candidate_id)
        ),
        "content": candidate.content,
        "instrument_matches": matches,
        "reaction_ready": reaction_ready,
    }


def corpus_stats(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    by_source = Counter(str(row["source_code"]) for row in materialized)
    by_quality = Counter(str(row["timestamp_quality"]) for row in materialized)
    by_status = Counter(str(row["status"]) for row in materialized)
    by_ticker: Counter[str] = Counter()
    by_year: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    matched = 0
    ambiguous = 0
    for row in materialized:
        matches_value = row["instrument_matches"]
        if isinstance(matches_value, list) and matches_value:
            matches = cast("list[dict[str, Any]]", matches_value)
            matched += 1
            for match in matches:
                ticker = match.get("ticker")
                if ticker:
                    by_ticker[str(ticker)] += 1
                if match.get("is_ambiguous"):
                    ambiguous += 1
                    break
        published_at = row.get("published_at")
        if isinstance(published_at, str) and len(published_at) >= 7:
            by_year[published_at[:4]] += 1
            by_month[published_at[:7]] += 1
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_candidates": len(materialized),
        "imported_count": sum(row.get("news_id") is not None for row in materialized),
        "matched_count": matched,
        "unmatched_count": len(materialized) - matched,
        "ambiguous_count": ambiguous,
        "reaction_ready_count": sum(bool(row["reaction_ready"]) for row in materialized),
        "by_source": dict(sorted(by_source.items())),
        "by_timestamp_quality": dict(sorted(by_quality.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_ticker": dict(sorted(by_ticker.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_month": dict(sorted(by_month.items())),
    }


def write_corpus(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    reaction_ready_only: bool,
    include_content: bool,
) -> int:
    selected = [row for row in rows if not reaction_ready_only or row["reaction_ready"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected:
            payload = dict(row)
            if not include_content:
                payload.pop("content", None)
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return len(selected)


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
