from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.daily_corpus.domain import (
    DailyCandidate,
    DailyExclusionReason,
    DailyFeatureRow,
    DailyReaction,
    SessionClose,
    build_daily_feature_row,
    build_daily_reaction,
    collapse_complete_session_closes,
    daily_eligibility,
)
from src.free_historical_data.domain import FreeSourceStatus
from src.free_historical_data.registry import free_source_audits
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.market_data.domain.entities import BenchmarkCandle, MarketCandle
from src.market_data.infrastructure.models import (
    BenchmarkCandleRecord,
    MarketBenchmarkRecord,
    MarketCandleRecord,
)
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord

SOURCE_DATE_PROVENANCE = "historical_news_candidates.source_published_at"
_FIXED_UTC_OFFSET = re.compile(r"^UTC(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")


@dataclass(frozen=True, slots=True)
class DailyCorpusBuildResult:
    candidates: list[DailyCandidate]
    eligible: list[DailyCandidate]
    reactions: list[DailyReaction]
    features: list[DailyFeatureRow]
    exclusions: dict[UUID, DailyExclusionReason]


async def build_daily_corpus(session: AsyncSession) -> DailyCorpusBuildResult:
    candidates = await _load_candidates(session)
    eligible: list[DailyCandidate] = []
    exclusions: dict[UUID, DailyExclusionReason] = {}
    for candidate in candidates:
        reason = daily_eligibility(candidate)
        if reason is None:
            eligible.append(candidate)
        else:
            exclusions[candidate.news_id] = reason
    security_closes, benchmark_closes = await _load_session_closes(session, eligible)
    reactions: list[DailyReaction] = []
    features: list[DailyFeatureRow] = []
    for candidate in eligible:
        assert candidate.instrument_id is not None
        reaction, reason = build_daily_reaction(
            candidate,
            security_closes=security_closes.get(candidate.instrument_id, []),
            benchmark_closes=benchmark_closes,
        )
        if reaction is None:
            assert reason is not None
            exclusions[candidate.news_id] = reason
            continue
        baseline_security = _session_by_date(
            security_closes[candidate.instrument_id], reaction.baseline_session_date
        )
        baseline_benchmark = _session_by_date(benchmark_closes, reaction.baseline_session_date)
        reactions.append(reaction)
        features.append(
            build_daily_feature_row(
                reaction,
                baseline_security=baseline_security,
                baseline_benchmark=baseline_benchmark,
            )
        )
    return DailyCorpusBuildResult(
        candidates=candidates,
        eligible=eligible,
        reactions=reactions,
        features=features,
        exclusions=exclusions,
    )


async def _load_candidates(session: AsyncSession) -> list[DailyCandidate]:
    rows = (
        await session.execute(
            select(HistoricalNewsCandidateRecord, HistoricalNewsSourceRecord, NewsItemRecord)
            .join(
                HistoricalNewsSourceRecord,
                HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
            )
            .join(
                NewsItemRecord,
                NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id,
            )
            .where(HistoricalNewsCandidateRecord.imported_news_id.is_not(None))
            .where(HistoricalNewsSourceRecord.source_code != "ML_FEATURE_SYNTHETIC_SMOKE")
            .order_by(
                HistoricalNewsSourceRecord.source_code,
                HistoricalNewsCandidateRecord.source_published_at,
                HistoricalNewsCandidateRecord.source_item_id,
            )
        )
    ).all()
    news_ids = [news.id for _, _, news in rows]
    match_rows: list[tuple[NewsInstrumentMatchRecord, InstrumentRecord]] = []
    if news_ids:
        match_result = await session.execute(
            select(NewsInstrumentMatchRecord, InstrumentRecord)
            .join(
                InstrumentRecord,
                InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id,
            )
            .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
        )
        match_rows = list(match_result.tuples())
    matches: defaultdict[UUID, list[tuple[NewsInstrumentMatchRecord, InstrumentRecord]]] = (
        defaultdict(list)
    )
    for match, instrument in match_rows:
        matches[match.news_id].append((match, instrument))
    compliant_sources = {
        audit.source_code
        for audit in free_source_audits()
        if audit.status
        in {
            FreeSourceStatus.COMPLIANT_EXACT,
            FreeSourceStatus.COMPLIANT_DATE_ONLY,
        }
    }
    candidates: list[DailyCandidate] = []
    for historical, source, news in rows:
        source_matches = matches[news.id]
        unambiguous = [item for item in source_matches if not item[0].is_ambiguous]
        selected = unambiguous[0][1] if len(unambiguous) == 1 else None
        source_date = source_calendar_date(
            historical.source_published_at,
            historical.source_timezone,
        )
        candidates.append(
            DailyCandidate(
                news_id=news.id,
                source_code=source.source_code,
                source_item_id=historical.source_item_id,
                source_url=historical.source_url,
                ticker=None if selected is None else selected.ticker,
                instrument_id=None if selected is None else selected.id,
                publication_date=source_date,
                timestamp_quality=PublicationTimestampQuality(
                    historical.publication_timestamp_quality
                ),
                publication_date_from_source=historical.source_published_at is not None,
                provenance="REAL",
                source_compliant=source.source_code in compliant_sources,
                duplicate=historical.exact_content_duplicate,
                match_count=len(source_matches),
                ambiguous_match=any(item[0].is_ambiguous for item in source_matches),
                text_length=len(news.raw_content),
            )
        )
    return candidates


async def _load_session_closes(
    session: AsyncSession,
    candidates: list[DailyCandidate],
) -> tuple[dict[UUID, list[SessionClose]], list[SessionClose]]:
    if not candidates:
        return {}, []
    instrument_ids = {
        candidate.instrument_id for candidate in candidates if candidate.instrument_id is not None
    }
    dates = [
        datetime.combine(candidate.publication_date, time(), UTC)
        for candidate in candidates
        if candidate.publication_date is not None
    ]
    from_at = min(dates) - timedelta(days=14)
    till_at = max(dates) + timedelta(days=15)
    candle_records = (
        await session.execute(
            select(MarketCandleRecord)
            .where(
                MarketCandleRecord.instrument_id.in_(instrument_ids),
                MarketCandleRecord.interval_minutes == 1,
                MarketCandleRecord.begin_at >= from_at,
                MarketCandleRecord.begin_at <= till_at,
            )
            .order_by(MarketCandleRecord.instrument_id, MarketCandleRecord.end_at)
        )
    ).scalars()
    by_instrument: defaultdict[UUID, list[MarketCandle]] = defaultdict(list)
    for record in candle_records:
        by_instrument[record.instrument_id].append(record.to_entity())
    benchmark = (
        (
            await session.execute(
                select(MarketBenchmarkRecord).where(MarketBenchmarkRecord.code == "IMOEX")
            )
        )
        .scalars()
        .first()
    )
    benchmark_entities: list[BenchmarkCandle] = []
    if benchmark is not None:
        records = (
            await session.execute(
                select(BenchmarkCandleRecord)
                .where(
                    BenchmarkCandleRecord.benchmark_id == benchmark.id,
                    BenchmarkCandleRecord.interval_minutes == 1,
                    BenchmarkCandleRecord.begin_at >= from_at,
                    BenchmarkCandleRecord.begin_at <= till_at,
                )
                .order_by(BenchmarkCandleRecord.end_at)
            )
        ).scalars()
        benchmark_entities = [record.to_entity() for record in records]
    security = {
        instrument_id: collapse_complete_session_closes(candles)
        for instrument_id, candles in by_instrument.items()
    }
    return security, collapse_complete_session_closes(benchmark_entities)


def source_calendar_date(published_at: datetime | None, source_timezone: str | None) -> date | None:
    if published_at is None:
        return None
    return published_at.astimezone(_source_timezone(source_timezone)).date()


def _source_timezone(value: str | None) -> timezone | ZoneInfo:
    if not value:
        return UTC
    fixed = _FIXED_UTC_OFFSET.fullmatch(value)
    if fixed is None:
        return ZoneInfo(value)
    offset = timedelta(
        hours=int(fixed.group("hours")),
        minutes=int(fixed.group("minutes")),
    )
    if fixed.group("sign") == "-":
        offset = -offset
    return timezone(offset)


def _session_by_date(closes: list[SessionClose], session_date: date) -> SessionClose:
    for close in closes:
        if close.session_date == session_date:
            return close
    raise ValueError("session close disappeared during daily build")
