from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field

from src.market_data.domain.entities import MarketCandle, MarketDataImport
from src.market_data.domain.enums import MarketDataImportStatus


class BackfillCandlesRequest(BaseModel):
    date_from: date
    date_till: date
    interval_minutes: int = Field(default=1, ge=1, le=60)


class MarketDataImportResponse(BaseModel):
    id: UUID
    provider: str
    instrument_id: UUID
    ticker: str
    board: str
    interval_minutes: int
    requested_from: date
    requested_till: date
    source_timezone: str
    started_at: datetime
    finished_at: datetime | None
    status: MarketDataImportStatus
    pages_received: int
    rows_received: int
    rows_valid: int
    rows_rejected: int
    rows_inserted: int
    rows_existing: int
    error_code: str | None
    adapter_version: str

    @classmethod
    def from_entity(cls, item: MarketDataImport) -> MarketDataImportResponse:
        payload = asdict(item)
        payload["provider"] = item.provider.value
        return cls(**payload)


class BackfillCandlesResponse(BaseModel):
    import_id: UUID
    status: MarketDataImportStatus
    instrument_id: UUID
    ticker: str
    board: str
    interval_minutes: int
    pages_received: int
    rows_received: int
    rows_valid: int
    rows_rejected: int
    rows_inserted: int
    rows_existing: int
    started_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_import(cls, item: MarketDataImport) -> BackfillCandlesResponse:
        return cls(
            import_id=item.id,
            status=item.status,
            instrument_id=item.instrument_id,
            ticker=item.ticker,
            board=item.board,
            interval_minutes=item.interval_minutes,
            pages_received=item.pages_received,
            rows_received=item.rows_received,
            rows_valid=item.rows_valid,
            rows_rejected=item.rows_rejected,
            rows_inserted=item.rows_inserted,
            rows_existing=item.rows_existing,
            started_at=item.started_at,
            finished_at=item.finished_at,
        )


class MarketCandleResponse(BaseModel):
    id: UUID
    instrument_id: UUID
    provider: str
    engine: str
    market: str
    board: str
    ticker_snapshot: str
    interval_minutes: int
    begin_at: datetime
    end_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    value: Decimal
    fetched_at: datetime
    adapter_version: str

    @classmethod
    def from_entity(cls, candle: MarketCandle) -> MarketCandleResponse:
        payload = asdict(candle)
        payload["provider"] = candle.provider.value
        return cls(**payload)
