from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from uuid import UUID

from src.instruments.application.ports import InstrumentRepository
from src.market_data.application.exceptions import (
    InstrumentMarketDataConflictError,
    InstrumentMarketDataNotFoundError,
    MarketDataProviderError,
    MarketDataStorageError,
    MarketDataValidationError,
)
from src.market_data.application.ports import MarketDataProvider, MarketDataRepository
from src.market_data.domain.entities import MarketCandle, MarketDataImport
from src.market_data.domain.enums import MarketDataImportStatus
from src.news.domain.time import ensure_aware_utc


@dataclass(frozen=True, slots=True)
class BackfillInstrumentCandlesCommand:
    instrument_id: UUID
    date_from: date
    date_till: date
    interval_minutes: int = 1


@dataclass(frozen=True, slots=True)
class BackfillInstrumentCandlesResult:
    import_record: MarketDataImport


class BackfillInstrumentCandles:
    def __init__(
        self,
        *,
        instrument_repository: InstrumentRepository,
        market_data_repository: MarketDataRepository,
        provider: MarketDataProvider,
    ) -> None:
        self._instrument_repository = instrument_repository
        self._market_data_repository = market_data_repository
        self._provider = provider

    async def execute(
        self,
        command: BackfillInstrumentCandlesCommand,
    ) -> BackfillInstrumentCandlesResult:
        _validate_backfill_range(command.date_from, command.date_till)
        if command.interval_minutes != 1:
            raise MarketDataValidationError("only interval_minutes=1 is supported")
        instrument = await self._instrument_repository.get_instrument(command.instrument_id)
        if instrument is None:
            raise InstrumentMarketDataNotFoundError("instrument not found")
        if not instrument.ticker or instrument.primary_board is None:
            raise InstrumentMarketDataConflictError("instrument lacks ticker or primary_board")
        started = await self._market_data_repository.create_import(
            MarketDataImport.start(
                instrument_id=instrument.id,
                ticker=instrument.ticker,
                board=instrument.primary_board,
                interval_minutes=command.interval_minutes,
                requested_from=command.date_from,
                requested_till=command.date_till,
            )
        )
        try:
            candles, pages, received, valid, rejected = await self._provider.fetch_candles(
                instrument_id=instrument.id,
                ticker=instrument.ticker,
                board=instrument.primary_board,
                date_from=command.date_from,
                date_till=command.date_till,
                interval_minutes=command.interval_minutes,
            )
            save_result = await self._market_data_repository.save_candles(candles)
            status = (
                MarketDataImportStatus.PARTIAL if rejected else MarketDataImportStatus.SUCCEEDED
            )
            finished = started.finish(
                status=status,
                pages_received=pages,
                rows_received=received,
                rows_valid=valid,
                rows_rejected=rejected,
                rows_inserted=save_result.inserted,
                rows_existing=save_result.existing,
            )
        except MarketDataProviderError:
            finished = started.finish(
                status=MarketDataImportStatus.FAILED,
                pages_received=0,
                rows_received=0,
                rows_valid=0,
                rows_rejected=0,
                rows_inserted=0,
                rows_existing=0,
                error_code="PROVIDER_ERROR",
            )
            await self._market_data_repository.finish_import(finished)
            raise
        except MarketDataStorageError:
            finished = started.finish(
                status=MarketDataImportStatus.FAILED,
                pages_received=0,
                rows_received=0,
                rows_valid=0,
                rows_rejected=0,
                rows_inserted=0,
                rows_existing=0,
                error_code="STORAGE_ERROR",
            )
            await self._market_data_repository.finish_import(finished)
            raise
        return BackfillInstrumentCandlesResult(
            import_record=await self._market_data_repository.finish_import(finished)
        )


class ListInstrumentCandles:
    def __init__(self, repository: MarketDataRepository) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        instrument_id: UUID,
        from_at: datetime,
        till_at: datetime,
        interval_minutes: int,
        limit: int,
        offset: int,
    ) -> list[MarketCandle]:
        if interval_minutes != 1:
            raise MarketDataValidationError("only interval_minutes=1 is supported")
        from_utc = ensure_aware_utc(from_at, "from")
        till_utc = ensure_aware_utc(till_at, "till")
        if till_utc < from_utc:
            raise MarketDataValidationError("till must not be before from")
        return await self._repository.list_candles(
            instrument_id=instrument_id,
            interval_minutes=interval_minutes,
            from_at=from_utc,
            till_at=till_utc,
            limit=limit,
            offset=offset,
        )


class GetMarketDataImport:
    def __init__(self, repository: MarketDataRepository) -> None:
        self._repository = repository

    async def execute(self, import_id: UUID) -> MarketDataImport | None:
        return await self._repository.get_import(import_id)


def utc_bounds_for_dates(date_from: date, date_till: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(date_from, time.min, tzinfo=UTC),
        datetime.combine(date_till, time.max, tzinfo=UTC),
    )


def _validate_backfill_range(date_from: date, date_till: date) -> None:
    if date_till < date_from:
        raise MarketDataValidationError("date_till must not be before date_from")
    if (date_till - date_from).days > 31:
        raise MarketDataValidationError("date range must not exceed 31 calendar days")
