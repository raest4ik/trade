from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.fresh_real_corpus.domain import (
    APPROVED_SOURCE_TICKERS,
    FreshCorpusRecord,
    MatchStatus,
    SelectionPolicy,
)
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.application.use_cases import MatchNewsInstruments
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository


async def refresh_instrument_matches(session: AsyncSession, *, policy: SelectionPolicy) -> int:
    """Match every bounded source/date candidate before selection without model inference."""
    normalized = policy.normalized()
    result = await session.execute(
        select(NewsItemRecord.id)
        .join(
            HistoricalNewsCandidateRecord,
            HistoricalNewsCandidateRecord.imported_news_id == NewsItemRecord.id,
        )
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .where(
            HistoricalNewsSourceRecord.source_code.in_(normalized.source_codes),
            HistoricalNewsCandidateRecord.source_published_at >= normalized.date_from,
            HistoricalNewsCandidateRecord.source_published_at <= normalized.date_to,
            HistoricalNewsCandidateRecord.publication_timestamp_quality
            == PublicationTimestampQuality.EXACT.value,
        )
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsSourceRecord.source_code,
            HistoricalNewsCandidateRecord.source_item_id,
        )
        .limit(normalized.limit)
    )
    news_ids = list(result.scalars())
    matcher = MatchNewsInstruments(
        SqlAlchemyNewsRepository(session), SqlAlchemyInstrumentRepository(session)
    )
    for news_id in news_ids:
        await matcher.execute(news_id)
    return len(news_ids)


async def load_bounded_records(
    session: AsyncSession, *, policy: SelectionPolicy
) -> list[FreshCorpusRecord]:
    normalized = policy.normalized()
    result = await session.execute(
        select(HistoricalNewsCandidateRecord, HistoricalNewsSourceRecord, NewsItemRecord)
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .join(NewsItemRecord, NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id)
        .where(
            HistoricalNewsSourceRecord.source_code.in_(normalized.source_codes),
            HistoricalNewsCandidateRecord.source_published_at >= normalized.date_from,
            HistoricalNewsCandidateRecord.source_published_at <= normalized.date_to,
            HistoricalNewsCandidateRecord.publication_timestamp_quality
            == PublicationTimestampQuality.EXACT.value,
        )
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsSourceRecord.source_code,
            HistoricalNewsCandidateRecord.source_item_id,
        )
        .limit(normalized.limit)
    )
    rows = result.all()
    news_ids = [news.id for _, _, news in rows]
    matches_by_news = await _matches(session, news_ids)
    records: list[FreshCorpusRecord] = []
    for candidate, source, news in rows:
        if candidate.source_published_at is None:
            continue
        matches = matches_by_news.get(news.id, [])
        status, ticker = _match_summary(source.source_code, matches)
        record = FreshCorpusRecord(
            news_id=news.id,
            source_code=source.source_code,
            source_item_id=candidate.source_item_id,
            source_url=candidate.source_url,
            ticker=ticker,
            published_at=candidate.source_published_at,
            original_timestamp_text=candidate.original_timestamp_text,
            source_timezone=candidate.source_timezone,
            timestamp_quality=PublicationTimestampQuality(candidate.publication_timestamp_quality),
            title=candidate.title,
            annotation_text=news.raw_content,
            content_hash=hashlib.sha256(news.raw_content.encode()).hexdigest(),
            storage_policy=candidate.content_storage_policy,
            content_is_excerpt=candidate.content_is_excerpt,
            match_status=status,
        )
        record.validate()
        records.append(record)
    return records


async def _matches(
    session: AsyncSession, news_ids: list[UUID]
) -> dict[UUID, list[tuple[str, bool]]]:
    if not news_ids:
        return {}
    result = await session.execute(
        select(NewsInstrumentMatchRecord, InstrumentRecord)
        .join(InstrumentRecord, InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id)
        .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
        .order_by(NewsInstrumentMatchRecord.news_id, InstrumentRecord.ticker)
    )
    grouped: dict[UUID, list[tuple[str, bool]]] = defaultdict(list)
    for match, instrument in result.all():
        grouped[match.news_id].append((instrument.ticker, match.is_ambiguous))
    return grouped


def _match_summary(source_code: str, matches: list[tuple[str, bool]]) -> tuple[MatchStatus, str]:
    source_ticker = APPROVED_SOURCE_TICKERS[source_code]
    tickers = sorted({ticker for ticker, _ in matches})
    if not tickers:
        return MatchStatus.UNMATCHED, source_ticker
    if len(tickers) > 1 or any(ambiguous for _, ambiguous in matches):
        return MatchStatus.AMBIGUOUS, source_ticker
    return MatchStatus.MATCHED, tickers[0]


def bounded_range_payload(policy: SelectionPolicy) -> dict[str, object]:
    normalized = policy.normalized()
    return {
        "source_codes": list(normalized.source_codes),
        "from": _timestamp(normalized.date_from),
        "to": _timestamp(normalized.date_to),
        "limit": normalized.limit,
        "source_order": normalized.source_order,
        "uses_model_outputs": False,
        "uses_future_returns": False,
    }


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
