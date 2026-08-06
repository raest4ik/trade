from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.market_data.domain.enums import MarketDataImportStatus, MarketDataProvider
from src.market_data.domain.exceptions import MarketDataDomainError
from src.news.domain.time import ensure_aware_utc, utc_now

MOEX_ENGINE_STOCK = "stock"
MOEX_MARKET_SHARES = "shares"
MOEX_ADAPTER_VERSION = "moex-iss-v1-minute-candles"
MOEX_SOURCE_TIMEZONE = "Europe/Moscow"
SUPPORTED_INTERVAL_MINUTES = 1


@dataclass(frozen=True, slots=True)
class MarketCandle:
    id: UUID
    instrument_id: UUID
    provider: MarketDataProvider
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
    def create(
        cls,
        *,
        instrument_id: UUID,
        board: str,
        ticker_snapshot: str,
        interval_minutes: int,
        begin_at: datetime,
        end_at: datetime,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        value: Decimal,
        fetched_at: datetime | None = None,
        provider: MarketDataProvider = MarketDataProvider.MOEX_ISS,
        engine: str = MOEX_ENGINE_STOCK,
        market: str = MOEX_MARKET_SHARES,
        adapter_version: str = MOEX_ADAPTER_VERSION,
    ) -> MarketCandle:
        if interval_minutes != SUPPORTED_INTERVAL_MINUTES:
            raise MarketDataDomainError("only 1 minute candles are supported")
        begin_utc = ensure_aware_utc(begin_at, "begin_at")
        end_utc = ensure_aware_utc(end_at, "end_at")
        fetched_utc = ensure_aware_utc(fetched_at or utc_now(), "fetched_at")
        if end_utc < begin_utc:
            raise MarketDataDomainError("end_at must not be before begin_at")
        if high < low:
            raise MarketDataDomainError("high must not be lower than low")
        if open_price < low or open_price > high or close < low or close > high:
            raise MarketDataDomainError("open and close must be inside low-high range")
        if any(price <= Decimal("0") for price in (open_price, high, low, close)):
            raise MarketDataDomainError("OHLC prices must be positive")
        if volume < Decimal("0") or value < Decimal("0"):
            raise MarketDataDomainError("volume and value must not be negative")
        board_normalized = _normalize_market_code(board, "board")
        ticker_normalized = _normalize_market_code(ticker_snapshot, "ticker_snapshot")
        return cls(
            id=uuid4(),
            instrument_id=instrument_id,
            provider=provider,
            engine=engine,
            market=market,
            board=board_normalized,
            ticker_snapshot=ticker_normalized,
            interval_minutes=interval_minutes,
            begin_at=begin_utc,
            end_at=end_utc,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            value=value,
            fetched_at=fetched_utc,
            adapter_version=adapter_version,
        )


@dataclass(frozen=True, slots=True)
class MarketDataImport:
    id: UUID
    provider: MarketDataProvider
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
    def start(
        cls,
        *,
        instrument_id: UUID,
        ticker: str,
        board: str,
        interval_minutes: int,
        requested_from: date,
        requested_till: date,
    ) -> MarketDataImport:
        if requested_till < requested_from:
            raise MarketDataDomainError("requested_till must not be before requested_from")
        return cls(
            id=uuid4(),
            provider=MarketDataProvider.MOEX_ISS,
            instrument_id=instrument_id,
            ticker=_normalize_market_code(ticker, "ticker"),
            board=_normalize_market_code(board, "board"),
            interval_minutes=interval_minutes,
            requested_from=requested_from,
            requested_till=requested_till,
            source_timezone=MOEX_SOURCE_TIMEZONE,
            started_at=utc_now(),
            finished_at=None,
            status=MarketDataImportStatus.RUNNING,
            pages_received=0,
            rows_received=0,
            rows_valid=0,
            rows_rejected=0,
            rows_inserted=0,
            rows_existing=0,
            error_code=None,
            adapter_version=MOEX_ADAPTER_VERSION,
        )

    def finish(
        self,
        *,
        status: MarketDataImportStatus,
        pages_received: int,
        rows_received: int,
        rows_valid: int,
        rows_rejected: int,
        rows_inserted: int,
        rows_existing: int,
        error_code: str | None = None,
    ) -> MarketDataImport:
        return MarketDataImport(
            id=self.id,
            provider=self.provider,
            instrument_id=self.instrument_id,
            ticker=self.ticker,
            board=self.board,
            interval_minutes=self.interval_minutes,
            requested_from=self.requested_from,
            requested_till=self.requested_till,
            source_timezone=self.source_timezone,
            started_at=self.started_at,
            finished_at=utc_now(),
            status=status,
            pages_received=pages_received,
            rows_received=rows_received,
            rows_valid=rows_valid,
            rows_rejected=rows_rejected,
            rows_inserted=rows_inserted,
            rows_existing=rows_existing,
            error_code=error_code,
            adapter_version=self.adapter_version,
        )


@dataclass(frozen=True, slots=True)
class RejectedCandleRow:
    page: int
    row_index: int
    reason: str


@dataclass(frozen=True, slots=True)
class CandleBatchSaveResult:
    inserted: int
    existing: int


def _normalize_market_code(value: str, field_name: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise MarketDataDomainError(f"{field_name} must not be empty")
    return normalized
