from __future__ import annotations

from datetime import date, datetime
from typing import Protocol
from uuid import UUID

from src.market_data.domain.entities import (
    BenchmarkCandle,
    CandleBatchSaveResult,
    MarketBenchmark,
    MarketCandle,
    MarketDataImport,
)


class MarketDataRepository(Protocol):
    async def save_benchmark(self, item: MarketBenchmark) -> MarketBenchmark: ...

    async def get_benchmark_by_code(self, code: str) -> MarketBenchmark | None: ...

    async def create_import(self, item: MarketDataImport) -> MarketDataImport: ...

    async def finish_import(self, item: MarketDataImport) -> MarketDataImport: ...

    async def get_import(self, import_id: UUID) -> MarketDataImport | None: ...

    async def save_candles(self, candles: list[MarketCandle]) -> CandleBatchSaveResult: ...

    async def save_benchmark_candles(
        self, candles: list[BenchmarkCandle]
    ) -> CandleBatchSaveResult: ...

    async def list_benchmark_candles(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        from_at: datetime,
        till_at: datetime,
        limit: int,
        offset: int,
    ) -> list[BenchmarkCandle]: ...

    async def get_last_benchmark_candle_ending_at_or_before(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> BenchmarkCandle | None: ...

    async def get_first_benchmark_candle_ending_at_or_after(
        self,
        *,
        benchmark_id: UUID,
        interval_minutes: int,
        at: datetime,
    ) -> BenchmarkCandle | None: ...

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


class BenchmarkMarketDataProvider(Protocol):
    async def fetch_benchmark_candles(
        self,
        *,
        benchmark: MarketBenchmark,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> tuple[list[BenchmarkCandle], int, int, int, int]: ...
