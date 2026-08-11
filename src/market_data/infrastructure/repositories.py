from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from src.market_data.application.exceptions import MarketDataStorageError
from src.market_data.domain.entities import (
    BenchmarkCandle,
    CandleBatchSaveResult,
    MarketBenchmark,
    MarketCandle,
    MarketDataImport,
)
from src.market_data.infrastructure.models import (
    BenchmarkCandleRecord,
    MarketBenchmarkRecord,
    MarketCandleRecord,
    MarketDataImportRecord,
)


class SqlAlchemyMarketDataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_benchmark(self, item: MarketBenchmark) -> MarketBenchmark:
        existing = await self.get_benchmark_by_code(item.code)
        if existing is not None:
            return existing
        record = MarketBenchmarkRecord.from_entity(item)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self.get_benchmark_by_code(item.code)
            if existing is not None:
                return existing
            raise MarketDataStorageError(
                "benchmark uniqueness conflict could not be resolved"
            ) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MarketDataStorageError("could not save market benchmark") from exc
        return record.to_entity()

    async def get_benchmark_by_code(self, code: str) -> MarketBenchmark | None:
        try:
            result = await self._session.execute(
                select(MarketBenchmarkRecord).where(
                    MarketBenchmarkRecord.code == code.strip().upper()
                )
            )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not read market benchmark") from exc
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def create_import(self, item: MarketDataImport) -> MarketDataImport:
        record = MarketDataImportRecord.from_entity(item)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MarketDataStorageError("could not create market data import") from exc
        return record.to_entity()

    async def finish_import(self, item: MarketDataImport) -> MarketDataImport:
        try:
            result = await self._session.execute(
                select(MarketDataImportRecord).where(MarketDataImportRecord.id == item.id)
            )
            record = result.scalar_one()
            record.update_from_entity(item)
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MarketDataStorageError("could not finish market data import") from exc
        return record.to_entity()

    async def get_import(self, import_id: UUID) -> MarketDataImport | None:
        try:
            result = await self._session.execute(
                select(MarketDataImportRecord).where(MarketDataImportRecord.id == import_id)
            )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not read market data import") from exc
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def save_candles(self, candles: list[MarketCandle]) -> CandleBatchSaveResult:
        if not candles:
            return CandleBatchSaveResult(inserted=0, existing=0)
        unique_candles = list(
            {
                (
                    candle.instrument_id,
                    candle.provider.value,
                    candle.board,
                    candle.interval_minutes,
                    candle.begin_at,
                ): candle
                for candle in candles
            }.values()
        )
        existing_keys = await self._existing_candle_keys(unique_candles)
        new_candles = [
            candle
            for candle in unique_candles
            if (
                candle.instrument_id,
                candle.provider.value,
                candle.board,
                candle.interval_minutes,
                candle.begin_at,
            )
            not in existing_keys
        ]
        if not new_candles:
            return CandleBatchSaveResult(inserted=0, existing=len(candles))
        self._session.add_all(MarketCandleRecord.from_entity(candle) for candle in new_candles)
        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            existing_after_conflict = await self._existing_candle_keys(candles)
            return CandleBatchSaveResult(
                inserted=0,
                existing=sum(
                    1
                    for candle in candles
                    if (
                        candle.instrument_id,
                        candle.provider.value,
                        candle.board,
                        candle.interval_minutes,
                        candle.begin_at,
                    )
                    in existing_after_conflict
                ),
            )
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MarketDataStorageError("could not save market candles") from exc
        return CandleBatchSaveResult(
            inserted=len(new_candles),
            existing=len(candles) - len(new_candles),
        )

    async def save_benchmark_candles(self, candles: list[BenchmarkCandle]) -> CandleBatchSaveResult:
        if not candles:
            return CandleBatchSaveResult(inserted=0, existing=0)
        unique_candles = list(
            {
                (
                    candle.benchmark_id,
                    candle.provider.value,
                    candle.interval_minutes,
                    candle.begin_at,
                ): candle
                for candle in candles
            }.values()
        )
        existing_keys = await self._existing_benchmark_candle_keys(unique_candles)
        new_candles = [
            candle
            for candle in unique_candles
            if (
                candle.benchmark_id,
                candle.provider.value,
                candle.interval_minutes,
                candle.begin_at,
            )
            not in existing_keys
        ]
        if not new_candles:
            return CandleBatchSaveResult(inserted=0, existing=len(candles))
        self._session.add_all(BenchmarkCandleRecord.from_entity(item) for item in new_candles)
        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            existing_after_conflict = await self._existing_benchmark_candle_keys(candles)
            return CandleBatchSaveResult(
                inserted=0,
                existing=sum(
                    1
                    for candle in candles
                    if (
                        candle.benchmark_id,
                        candle.provider.value,
                        candle.interval_minutes,
                        candle.begin_at,
                    )
                    in existing_after_conflict
                ),
            )
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise MarketDataStorageError("could not save benchmark candles") from exc
        return CandleBatchSaveResult(
            inserted=len(new_candles),
            existing=len(candles) - len(new_candles),
        )

    async def list_benchmark_candles(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        from_at: datetime,
        till_at: datetime,
        limit: int,
        offset: int,
    ) -> list[BenchmarkCandle]:
        try:
            result = await self._session.execute(
                select(BenchmarkCandleRecord)
                .where(
                    BenchmarkCandleRecord.benchmark_id == benchmark_id,
                    BenchmarkCandleRecord.interval_minutes == interval_minutes,
                    BenchmarkCandleRecord.begin_at >= from_at,
                    BenchmarkCandleRecord.begin_at <= till_at,
                )
                .order_by(BenchmarkCandleRecord.begin_at)
                .limit(limit)
                .offset(offset)
            )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not list benchmark candles") from exc
        return [record.to_entity() for record in result.scalars()]

    async def get_last_benchmark_candle_ending_at_or_before(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> BenchmarkCandle | None:
        return await self._one_benchmark_candle(
            select(BenchmarkCandleRecord)
            .where(
                BenchmarkCandleRecord.benchmark_id == benchmark_id,
                BenchmarkCandleRecord.interval_minutes == interval_minutes,
                BenchmarkCandleRecord.end_at <= at,
            )
            .order_by(BenchmarkCandleRecord.end_at.desc())
        )

    async def get_first_benchmark_candle_ending_at_or_after(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> BenchmarkCandle | None:
        return await self._one_benchmark_candle(
            select(BenchmarkCandleRecord)
            .where(
                BenchmarkCandleRecord.benchmark_id == benchmark_id,
                BenchmarkCandleRecord.interval_minutes == interval_minutes,
                BenchmarkCandleRecord.end_at >= at,
            )
            .order_by(BenchmarkCandleRecord.end_at)
        )

    async def list_candles(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        from_at: datetime,
        till_at: datetime,
        limit: int,
        offset: int,
    ) -> list[MarketCandle]:
        try:
            result = await self._session.execute(
                select(MarketCandleRecord)
                .where(
                    MarketCandleRecord.instrument_id == instrument_id,
                    MarketCandleRecord.interval_minutes == interval_minutes,
                    MarketCandleRecord.begin_at >= from_at,
                    MarketCandleRecord.begin_at <= till_at,
                )
                .order_by(MarketCandleRecord.begin_at)
                .limit(limit)
                .offset(offset)
            )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not list market candles") from exc
        return [record.to_entity() for record in result.scalars()]

    async def get_last_candle_ending_at_or_before(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None:
        return await self._one_candle(
            select(MarketCandleRecord)
            .where(
                MarketCandleRecord.instrument_id == instrument_id,
                MarketCandleRecord.interval_minutes == interval_minutes,
                MarketCandleRecord.end_at <= at,
            )
            .order_by(MarketCandleRecord.end_at.desc())
        )

    async def get_first_candle_beginning_at_or_after(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None:
        return await self._one_candle(
            select(MarketCandleRecord)
            .where(
                MarketCandleRecord.instrument_id == instrument_id,
                MarketCandleRecord.interval_minutes == interval_minutes,
                MarketCandleRecord.begin_at >= at,
            )
            .order_by(MarketCandleRecord.begin_at)
        )

    async def get_first_candle_ending_at_or_after(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None:
        return await self._one_candle(
            select(MarketCandleRecord)
            .where(
                MarketCandleRecord.instrument_id == instrument_id,
                MarketCandleRecord.interval_minutes == interval_minutes,
                MarketCandleRecord.end_at >= at,
            )
            .order_by(MarketCandleRecord.end_at)
        )

    async def _one_candle(self, query: Select[tuple[MarketCandleRecord]]) -> MarketCandle | None:
        try:
            result = await self._session.execute(query)
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not read market candle") from exc
        record = result.scalars().first()
        return None if record is None else record.to_entity()

    async def _one_benchmark_candle(
        self, query: Select[tuple[BenchmarkCandleRecord]]
    ) -> BenchmarkCandle | None:
        try:
            result = await self._session.execute(query)
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not read benchmark candle") from exc
        record = result.scalars().first()
        return None if record is None else record.to_entity()

    async def _existing_candle_keys(
        self,
        candles: list[MarketCandle],
    ) -> set[tuple[UUID, str, str, int, datetime]]:
        grouped: dict[tuple[UUID, str, str, int], list[datetime]] = defaultdict(list)
        for candle in candles:
            grouped[
                (
                    candle.instrument_id,
                    candle.provider.value,
                    candle.board,
                    candle.interval_minutes,
                )
            ].append(candle.begin_at)
        existing: set[tuple[UUID, str, str, int, datetime]] = set()
        try:
            for (instrument_id, provider, board, interval), timestamps in grouped.items():
                for chunk in _chunks(timestamps):
                    result = await self._session.execute(
                        select(
                            MarketCandleRecord.instrument_id,
                            MarketCandleRecord.provider,
                            MarketCandleRecord.board,
                            MarketCandleRecord.interval_minutes,
                            MarketCandleRecord.begin_at,
                        ).where(
                            MarketCandleRecord.instrument_id == instrument_id,
                            MarketCandleRecord.provider == provider,
                            MarketCandleRecord.board == board,
                            MarketCandleRecord.interval_minutes == interval,
                            MarketCandleRecord.begin_at.in_(chunk),
                        )
                    )
                    existing.update(
                        (
                            row.instrument_id,
                            row.provider,
                            row.board,
                            row.interval_minutes,
                            row.begin_at,
                        )
                        for row in result.all()
                    )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not check existing candles") from exc
        return existing

    async def _existing_benchmark_candle_keys(
        self,
        candles: list[BenchmarkCandle],
    ) -> set[tuple[UUID, str, int, datetime]]:
        grouped: dict[tuple[UUID, str, int], list[datetime]] = defaultdict(list)
        for candle in candles:
            grouped[
                (
                    candle.benchmark_id,
                    candle.provider.value,
                    candle.interval_minutes,
                )
            ].append(candle.begin_at)
        existing: set[tuple[UUID, str, int, datetime]] = set()
        try:
            for (benchmark_id, provider, interval), timestamps in grouped.items():
                for chunk in _chunks(timestamps):
                    result = await self._session.execute(
                        select(
                            BenchmarkCandleRecord.benchmark_id,
                            BenchmarkCandleRecord.provider,
                            BenchmarkCandleRecord.interval_minutes,
                            BenchmarkCandleRecord.begin_at,
                        ).where(
                            BenchmarkCandleRecord.benchmark_id == benchmark_id,
                            BenchmarkCandleRecord.provider == provider,
                            BenchmarkCandleRecord.interval_minutes == interval,
                            BenchmarkCandleRecord.begin_at.in_(chunk),
                        )
                    )
                    existing.update(
                        (
                            row.benchmark_id,
                            row.provider,
                            row.interval_minutes,
                            row.begin_at,
                        )
                        for row in result.all()
                    )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not check existing benchmark candles") from exc
        return existing


def _chunks(values: list[datetime], size: int = 500) -> list[list[datetime]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
