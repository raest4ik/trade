from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.events.application.use_cases import AnalyzeNewsEvent
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.application.use_cases import MatchNewsInstruments
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reaction_ready_corpus.domain import (
    REAL_SOURCE_CODES,
    UNIVERSE,
    MarketBackfillWindow,
    MatchStatus,
    match_status,
    plan_market_windows,
)


@dataclass(frozen=True, slots=True)
class PrepareCorpusCommand:
    date_from: datetime
    date_to: datetime
    source_codes: tuple[str, ...]
    tickers: tuple[str, ...] = UNIVERSE
    limit: int = 100
    dry_run: bool = False

    def normalized(self) -> PrepareCorpusCommand:
        sources = tuple(
            sorted({item.strip().upper() for item in self.source_codes if item.strip()})
        )
        tickers = tuple(sorted({item.strip().upper() for item in self.tickers if item.strip()}))
        if not sources or any(item not in REAL_SOURCE_CODES for item in sources):
            raise ValueError("source_codes must be explicitly approved REAL sources")
        if any(item not in UNIVERSE for item in tickers):
            raise ValueError("tickers must belong to the configured universe")
        if self.date_to < self.date_from:
            raise ValueError("date_to must not be before date_from")
        if not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        return PrepareCorpusCommand(
            date_from=self.date_from,
            date_to=self.date_to,
            source_codes=sources,
            tickers=tickers,
            limit=self.limit,
            dry_run=self.dry_run,
        )


@dataclass(frozen=True, slots=True)
class PrepareCorpusResult:
    candidate_count: int
    analyzed_count: int
    matched_count: int
    ambiguous_count: int
    unmatched_count: int
    windows: tuple[MarketBackfillWindow, ...]
    dry_run: bool


class PrepareReactionReadyCorpus:
    """Run deterministic matching/analysis before bounded market-data backfill."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def execute(self, command: PrepareCorpusCommand) -> PrepareCorpusResult:
        normalized = command.normalized()
        rows = await self._session.execute(
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
                NewsItemRecord.published_at >= normalized.date_from,
                NewsItemRecord.published_at <= normalized.date_to,
            )
            .order_by(NewsItemRecord.published_at, NewsItemRecord.id)
            .limit(normalized.limit)
        )
        news_ids = list(rows.scalars())
        news_repository = SqlAlchemyNewsRepository(self._session)
        instrument_repository = SqlAlchemyInstrumentRepository(self._session)
        matcher = MatchNewsInstruments(news_repository, instrument_repository)
        analyzer = AnalyzeNewsEvent(
            news_repository=news_repository,
            event_repository=SqlAlchemyEventAnalysisRepository(self._session),
        )
        analyzed = 0
        if not normalized.dry_run:
            for news_id in news_ids:
                await matcher.execute(news_id)
                await analyzer.execute(news_id)
                analyzed += 1

        matches = await self._session.execute(
            select(NewsInstrumentMatchRecord, InstrumentRecord, NewsItemRecord.published_at)
            .join(InstrumentRecord, InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id)
            .join(NewsItemRecord, NewsItemRecord.id == NewsInstrumentMatchRecord.news_id)
            .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
            .order_by(NewsInstrumentMatchRecord.news_id, InstrumentRecord.ticker)
        )
        grouped: dict[UUID, list[tuple[NewsInstrumentMatchRecord, InstrumentRecord, datetime]]] = {}
        for match, instrument, published_at in matches.all():
            if instrument.ticker in normalized.tickers:
                grouped.setdefault(match.news_id, []).append((match, instrument, published_at))

        matched_count = 0
        ambiguous_count = 0
        publications: list[tuple[str, datetime]] = []
        for news_id in news_ids:
            items = grouped.get(news_id, [])
            status = match_status(len(items), any(item[0].is_ambiguous for item in items))
            if status == MatchStatus.MATCHED:
                matched_count += 1
                _, instrument, published_at = items[0]
                publications.append((instrument.ticker, published_at))
            elif status == MatchStatus.AMBIGUOUS:
                ambiguous_count += 1
        return PrepareCorpusResult(
            candidate_count=len(news_ids),
            analyzed_count=analyzed,
            matched_count=matched_count,
            ambiguous_count=ambiguous_count,
            unmatched_count=len(news_ids) - matched_count - ambiguous_count,
            windows=tuple(plan_market_windows(publications)),
            dry_run=normalized.dry_run,
        )
