from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.market_data.domain.entities import MarketCandle, MarketDataImport
from src.market_data.domain.enums import MarketDataImportStatus, MarketDataProvider
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class MarketCandleRecord(Base):
    __tablename__ = "market_candles"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "provider",
            "board",
            "interval_minutes",
            "begin_at",
            name="uq_market_candles_instrument_provider_board_interval_begin",
        ),
        Index(
            "ix_market_candles_instrument_interval_begin",
            "instrument_id",
            "interval_minutes",
            "begin_at",
        ),
        Index("ix_market_candles_provider_board_begin", "provider", "board", "begin_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    instrument_id: Mapped[UUID] = mapped_column(ForeignKey("instruments.id", ondelete="RESTRICT"))
    provider: Mapped[str] = mapped_column(String(32))
    engine: Mapped[str] = mapped_column(String(32))
    market: Mapped[str] = mapped_column(String(32))
    board: Mapped[str] = mapped_column(String(32))
    ticker_snapshot: Mapped[str] = mapped_column(String(32))
    interval_minutes: Mapped[int] = mapped_column(Integer)
    begin_at: Mapped[datetime] = mapped_column(UtcDateTime())
    end_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(24, 10))
    high: Mapped[Decimal] = mapped_column(Numeric(24, 10))
    low: Mapped[Decimal] = mapped_column(Numeric(24, 10))
    close: Mapped[Decimal] = mapped_column(Numeric(24, 10))
    volume: Mapped[Decimal] = mapped_column(Numeric(28, 10))
    value: Mapped[Decimal] = mapped_column(Numeric(28, 10))
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime())
    adapter_version: Mapped[str] = mapped_column(String(64))

    @classmethod
    def from_entity(cls, candle: MarketCandle) -> MarketCandleRecord:
        return cls(
            id=candle.id,
            instrument_id=candle.instrument_id,
            provider=candle.provider.value,
            engine=candle.engine,
            market=candle.market,
            board=candle.board,
            ticker_snapshot=candle.ticker_snapshot,
            interval_minutes=candle.interval_minutes,
            begin_at=candle.begin_at,
            end_at=candle.end_at,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            value=candle.value,
            fetched_at=candle.fetched_at,
            adapter_version=candle.adapter_version,
        )

    def to_entity(self) -> MarketCandle:
        return MarketCandle(
            id=self.id,
            instrument_id=self.instrument_id,
            provider=MarketDataProvider(self.provider),
            engine=self.engine,
            market=self.market,
            board=self.board,
            ticker_snapshot=self.ticker_snapshot,
            interval_minutes=self.interval_minutes,
            begin_at=self.begin_at,
            end_at=self.end_at,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            value=self.value,
            fetched_at=self.fetched_at,
            adapter_version=self.adapter_version,
        )


class MarketDataImportRecord(Base):
    __tablename__ = "market_data_imports"
    __table_args__ = (
        Index("ix_market_data_imports_instrument_started", "instrument_id", "started_at"),
        Index("ix_market_data_imports_provider_board_started", "provider", "board", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    instrument_id: Mapped[UUID] = mapped_column(ForeignKey("instruments.id", ondelete="RESTRICT"))
    ticker: Mapped[str] = mapped_column(String(32))
    board: Mapped[str] = mapped_column(String(32))
    interval_minutes: Mapped[int] = mapped_column(Integer)
    requested_from: Mapped[date] = mapped_column()
    requested_till: Mapped[date] = mapped_column()
    source_timezone: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    pages_received: Mapped[int] = mapped_column(Integer)
    rows_received: Mapped[int] = mapped_column(Integer)
    rows_valid: Mapped[int] = mapped_column(Integer)
    rows_rejected: Mapped[int] = mapped_column(Integer)
    rows_inserted: Mapped[int] = mapped_column(Integer)
    rows_existing: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adapter_version: Mapped[str] = mapped_column(String(64))

    @classmethod
    def from_entity(cls, item: MarketDataImport) -> MarketDataImportRecord:
        return cls(
            id=item.id,
            provider=item.provider.value,
            instrument_id=item.instrument_id,
            ticker=item.ticker,
            board=item.board,
            interval_minutes=item.interval_minutes,
            requested_from=item.requested_from,
            requested_till=item.requested_till,
            source_timezone=item.source_timezone,
            started_at=item.started_at,
            finished_at=item.finished_at,
            status=item.status.value,
            pages_received=item.pages_received,
            rows_received=item.rows_received,
            rows_valid=item.rows_valid,
            rows_rejected=item.rows_rejected,
            rows_inserted=item.rows_inserted,
            rows_existing=item.rows_existing,
            error_code=item.error_code,
            adapter_version=item.adapter_version,
        )

    def update_from_entity(self, item: MarketDataImport) -> None:
        self.finished_at = item.finished_at
        self.status = item.status.value
        self.pages_received = item.pages_received
        self.rows_received = item.rows_received
        self.rows_valid = item.rows_valid
        self.rows_rejected = item.rows_rejected
        self.rows_inserted = item.rows_inserted
        self.rows_existing = item.rows_existing
        self.error_code = item.error_code

    def to_entity(self) -> MarketDataImport:
        return MarketDataImport(
            id=self.id,
            provider=MarketDataProvider(self.provider),
            instrument_id=self.instrument_id,
            ticker=self.ticker,
            board=self.board,
            interval_minutes=self.interval_minutes,
            requested_from=self.requested_from,
            requested_till=self.requested_till,
            source_timezone=self.source_timezone,
            started_at=self.started_at,
            finished_at=self.finished_at,
            status=MarketDataImportStatus(self.status),
            pages_received=self.pages_received,
            rows_received=self.rows_received,
            rows_valid=self.rows_valid,
            rows_rejected=self.rows_rejected,
            rows_inserted=self.rows_inserted,
            rows_existing=self.rows_existing,
            error_code=self.error_code,
            adapter_version=self.adapter_version,
        )
