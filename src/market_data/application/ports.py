from __future__ import annotations

from datetime import date, datetime
from typing import Protocol
from uuid import UUID

from src.market_data.domain.entities import (
    CandleBatchSaveResult,
    MarketCandle,
    MarketDataImport,
)


class MarketDataRepository(Protocol):
    async def create_import(self, item: MarketDataImport) -> MarketDataImport: ...

    async def finish_import(self, item: MarketDataImport) -> MarketDataImport: ...

    async def get_import(self, import_id: UUID) -> MarketDataImport | None: ...

    async def save_candles(self, candles: list[MarketCandle]) -> CandleBatchSaveResult: ...

    async def list_candles(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        from_at: datetime,
        till_at: datetime,
        limit: int,
        offset: int,
    ) -> list[MarketCandle]: ...

    async def get_last_candle_ending_at_or_before(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None: ...

    async def get_first_candle_beginning_at_or_after(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None: ...

    async def get_first_candle_ending_at_or_after(
        self,
        *,
        instrument_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> MarketCandle | None: ...


class MarketDataProvider(Protocol):
    async def fetch_candles(
        self,
        *,
        instrument_id: UUID,
        ticker: str,
        board: str,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> tuple[list[MarketCandle], int, int, int, int]: ...
