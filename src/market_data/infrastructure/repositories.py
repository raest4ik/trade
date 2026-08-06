from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from src.market_data.application.exceptions import MarketDataStorageError
from src.market_data.domain.entities import (
    CandleBatchSaveResult,
    MarketCandle,
    MarketDataImport,
)
from src.market_data.infrastructure.models import MarketCandleRecord, MarketDataImportRecord


class SqlAlchemyMarketDataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        existing_keys = await self._existing_candle_keys(candles)
        new_candles = [
            candle
            for candle in candles
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
        return CandleBatchSaveResult(inserted=len(new_candles), existing=len(existing_keys))

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

    async def get_first_candle_beginning_after(
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
                MarketCandleRecord.begin_at > at,
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

    async def _existing_candle_keys(
        self,
        candles: list[MarketCandle],
    ) -> set[tuple[UUID, str, str, int, datetime]]:
        clauses = [
            and_(
                MarketCandleRecord.instrument_id == candle.instrument_id,
                MarketCandleRecord.provider == candle.provider.value,
                MarketCandleRecord.board == candle.board,
                MarketCandleRecord.interval_minutes == candle.interval_minutes,
                MarketCandleRecord.begin_at == candle.begin_at,
            )
            for candle in candles
        ]
        try:
            result = await self._session.execute(
                select(
                    MarketCandleRecord.instrument_id,
                    MarketCandleRecord.provider,
                    MarketCandleRecord.board,
                    MarketCandleRecord.interval_minutes,
                    MarketCandleRecord.begin_at,
                ).where(or_(*clauses))
            )
        except SQLAlchemyError as exc:
            raise MarketDataStorageError("could not check existing candles") from exc
        return {
            (row.instrument_id, row.provider, row.board, row.interval_minutes, row.begin_at)
            for row in result.all()
        }
