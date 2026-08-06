from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.instruments.domain.entities import Instrument, IssuerAlias, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class InstrumentRecord(Base):
    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("exchange", "ticker", name="uq_instruments_exchange_ticker"),
        UniqueConstraint("isin", name="uq_instruments_isin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    figi: Mapped[str | None] = mapped_column(String(32), nullable=True)
    isin: Mapped[str | None] = mapped_column(String(16), nullable=True)
    short_name: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(500))
    issuer_name: Mapped[str] = mapped_column(String(255), index=True)
    exchange: Mapped[str] = mapped_column(String(32), index=True)
    currency: Mapped[str] = mapped_column(String(8))
    instrument_type: Mapped[str] = mapped_column(String(32))
    primary_board: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime())

    @classmethod
    def from_entity(cls, instrument: Instrument) -> InstrumentRecord:
        return cls(
            id=instrument.id,
            ticker=instrument.ticker,
            figi=instrument.figi,
            isin=instrument.isin,
            short_name=instrument.short_name,
            full_name=instrument.full_name,
            issuer_name=instrument.issuer_name,
            exchange=instrument.exchange,
            currency=instrument.currency,
            instrument_type=instrument.instrument_type.value,
            primary_board=instrument.primary_board,
            is_active=instrument.is_active,
            created_at=instrument.created_at,
            updated_at=instrument.updated_at,
        )

    def to_entity(self) -> Instrument:
        return Instrument(
            id=self.id,
            ticker=self.ticker,
            figi=self.figi,
            isin=self.isin,
            short_name=self.short_name,
            full_name=self.full_name,
            issuer_name=self.issuer_name,
            exchange=self.exchange,
            currency=self.currency,
            instrument_type=InstrumentType(self.instrument_type),
            primary_board=self.primary_board,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class IssuerAliasRecord(Base):
    __tablename__ = "issuer_aliases"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "normalized_alias",
            name="uq_issuer_aliases_instrument_normalized_alias",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"),
        index=True,
    )
    alias: Mapped[str] = mapped_column(String(500))
    normalized_alias: Mapped[str] = mapped_column(String(500), index=True)
    alias_type: Mapped[str] = mapped_column(String(32), index=True)
    priority: Mapped[int] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())

    @classmethod
    def from_entity(cls, alias: IssuerAlias) -> IssuerAliasRecord:
        return cls(
            id=alias.id,
            instrument_id=alias.instrument_id,
            alias=alias.alias,
            normalized_alias=alias.normalized_alias,
            alias_type=alias.alias_type.value,
            priority=alias.priority,
            is_active=alias.is_active,
            created_at=alias.created_at,
        )

    def to_entity(self) -> IssuerAlias:
        return IssuerAlias(
            id=self.id,
            instrument_id=self.instrument_id,
            alias=self.alias,
            normalized_alias=self.normalized_alias,
            alias_type=AliasType(self.alias_type),
            priority=self.priority,
            is_active=self.is_active,
            created_at=self.created_at,
        )


class NewsInstrumentMatchRecord(Base):
    __tablename__ = "news_instrument_matches"
    __table_args__ = (
        UniqueConstraint(
            "news_id",
            "instrument_id",
            "matcher_version",
            name="uq_news_instrument_matches_news_instrument_version",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    news_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_items.id", ondelete="CASCADE"),
        index=True,
    )
    instrument_id: Mapped[UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"),
        index=True,
    )
    matched_alias: Mapped[str] = mapped_column(String(500))
    alias_type: Mapped[str] = mapped_column(String(32))
    match_type: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    start_position: Mapped[int] = mapped_column(Integer)
    end_position: Mapped[int] = mapped_column(Integer)
    is_ambiguous: Mapped[bool] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    matcher_version: Mapped[str] = mapped_column(String(64), index=True)

    @classmethod
    def from_entity(cls, match: NewsInstrumentMatch) -> NewsInstrumentMatchRecord:
        return cls(
            id=match.id,
            news_id=match.news_id,
            instrument_id=match.instrument_id,
            matched_alias=match.matched_alias,
            alias_type=match.alias_type.value,
            match_type=match.match_type.value,
            confidence=match.confidence,
            start_position=match.start_position,
            end_position=match.end_position,
            is_ambiguous=match.is_ambiguous,
            created_at=match.created_at,
            matcher_version=match.matcher_version,
        )

    def to_entity(self) -> NewsInstrumentMatch:
        return NewsInstrumentMatch(
            id=self.id,
            news_id=self.news_id,
            instrument_id=self.instrument_id,
            matched_alias=self.matched_alias,
            alias_type=AliasType(self.alias_type),
            match_type=MatchType(self.match_type),
            confidence=self.confidence,
            start_position=self.start_position,
            end_position=self.end_position,
            is_ambiguous=self.is_ambiguous,
            created_at=self.created_at,
            matcher_version=self.matcher_version,
        ).ensure_utc()
