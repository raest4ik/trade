from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

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
from src.ml_features.application.ports import CandidateInstrumentMatch, FeatureCandidate
from src.ml_features.domain.entities import FeatureDatasetConfig, MlFeatureDatasetRun
from src.ml_features.infrastructure.models import MlFeatureDatasetRunRecord
from src.news.infrastructure.models import NewsItemRecord


class MlFeatureStorageError(RuntimeError):
    """Raised when feature dataset persistence or reads fail."""


class SqlAlchemyMlFeatureRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_candidates(self, config: FeatureDatasetConfig) -> list[FeatureCandidate]:
        normalized = config.normalized()
        try:
            result = await self._session.execute(
                select(
                    NewsItemRecord,
                    HistoricalNewsCandidateRecord.source_item_id,
                    HistoricalNewsSourceRecord.source_code,
                )
                .join(
                    HistoricalNewsCandidateRecord,
                    HistoricalNewsCandidateRecord.imported_news_id == NewsItemRecord.id,
                )
                .join(
                    HistoricalNewsSourceRecord,
                    HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
                )
                .where(
                    NewsItemRecord.published_at >= normalized.date_from,
                    NewsItemRecord.published_at <= normalized.date_to,
                )
                .order_by(NewsItemRecord.published_at, NewsItemRecord.id)
            )
            news_rows = result.all()
            news_ids = [news.id for news, _, _ in news_rows]
            matches_by_news: dict[UUID, list[CandidateInstrumentMatch]] = defaultdict(list)
            if news_ids:
                matches = await self._session.execute(
                    select(NewsInstrumentMatchRecord, InstrumentRecord)
                    .join(
                        InstrumentRecord,
                        InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id,
                    )
                    .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
                    .order_by(
                        NewsInstrumentMatchRecord.news_id,
                        InstrumentRecord.ticker,
                        NewsInstrumentMatchRecord.instrument_id,
                    )
                )
                for match, instrument in matches.all():
                    if normalized.tickers and instrument.ticker not in normalized.tickers:
                        continue
                    matches_by_news[match.news_id].append(
                        CandidateInstrumentMatch(
                            match=match.to_entity(),
                            instrument=instrument.to_entity(),
                        )
                    )
        except SQLAlchemyError as exc:
            raise MlFeatureStorageError("could not list feature candidates") from exc
        candidates: list[FeatureCandidate] = []
        for news, source_item_id, source_code in news_rows:
            matches = matches_by_news.get(news.id, [])
            if normalized.tickers and not matches:
                continue
            candidates.append(
                FeatureCandidate(
                    news=news.to_entity(),
                    source_code=source_code,
                    source_item_id=source_item_id,
                    matches=matches,
                )
            )
            if len(candidates) >= normalized.limit:
                break
        return candidates

    async def list_security_candles_as_of(
        self,
        *,
        instrument_id: UUID,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[MarketCandle]:
        try:
            result = await self._session.execute(
                select(MarketCandleRecord)
                .where(
                    MarketCandleRecord.instrument_id == instrument_id,
                    MarketCandleRecord.interval_minutes == 1,
                    MarketCandleRecord.end_at >= as_of - timedelta(minutes=lookback_minutes),
                    MarketCandleRecord.end_at <= as_of,
                )
                .order_by(MarketCandleRecord.end_at)
            )
        except SQLAlchemyError as exc:
            raise MlFeatureStorageError("could not read point-in-time security candles") from exc
        return [record.to_entity() for record in result.scalars()]

    async def list_benchmark_candles_as_of(
        self,
        *,
        benchmark_code: str,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[BenchmarkCandle] | None:
        try:
            benchmark = await self._session.execute(
                select(MarketBenchmarkRecord).where(
                    MarketBenchmarkRecord.code == benchmark_code.strip().upper()
                )
            )
            benchmark_record = benchmark.scalar_one_or_none()
            if benchmark_record is None:
                return None
            result = await self._session.execute(
                select(BenchmarkCandleRecord)
                .where(
                    BenchmarkCandleRecord.benchmark_id == benchmark_record.id,
                    BenchmarkCandleRecord.interval_minutes == 1,
                    BenchmarkCandleRecord.end_at >= as_of - timedelta(minutes=lookback_minutes),
                    BenchmarkCandleRecord.end_at <= as_of,
                )
                .order_by(BenchmarkCandleRecord.end_at)
            )
        except SQLAlchemyError as exc:
            raise MlFeatureStorageError("could not read point-in-time benchmark candles") from exc
        return [record.to_entity() for record in result.scalars()]

    async def create_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun:
        record = MlFeatureDatasetRunRecord.from_entity(run)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MlFeatureStorageError("could not create feature dataset run") from exc
        return record.to_entity()

    async def finish_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun:
        try:
            result = await self._session.execute(
                select(MlFeatureDatasetRunRecord).where(MlFeatureDatasetRunRecord.id == run.id)
            )
            record = result.scalar_one()
            record.update_from_entity(run)
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MlFeatureStorageError("could not finish feature dataset run") from exc
        return record.to_entity()
