from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.instruments.application.use_cases import CreateInstrumentCommand, CreateIssuerAliasCommand
from src.instruments.domain.entities import Instrument, IssuerAlias, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType


class InstrumentCreateRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    figi: str | None = Field(default=None, max_length=32)
    isin: str | None = Field(default=None, max_length=16)
    short_name: str = Field(min_length=1, max_length=255)
    full_name: str = Field(min_length=1, max_length=500)
    issuer_name: str = Field(min_length=1, max_length=255)
    exchange: str = Field(default="MOEX", min_length=1, max_length=32)
    currency: str = Field(default="RUB", min_length=1, max_length=8)
    instrument_type: InstrumentType
    is_active: bool = True

    def to_command(self) -> CreateInstrumentCommand:
        return CreateInstrumentCommand(
            ticker=self.ticker,
            figi=self.figi,
            isin=self.isin,
            short_name=self.short_name,
            full_name=self.full_name,
            issuer_name=self.issuer_name,
            exchange=self.exchange,
            currency=self.currency,
            instrument_type=self.instrument_type,
            is_active=self.is_active,
        )


class InstrumentResponse(BaseModel):
    id: UUID
    ticker: str
    figi: str | None
    isin: str | None
    short_name: str
    full_name: str
    issuer_name: str
    exchange: str
    currency: str
    instrument_type: InstrumentType
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, instrument: Instrument) -> InstrumentResponse:
        return cls(**asdict(instrument))


class IssuerAliasCreateRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=500)
    alias_type: AliasType
    priority: int = Field(default=100, ge=0, le=1000)
    is_active: bool = True

    def to_command(self, instrument_id: UUID) -> CreateIssuerAliasCommand:
        return CreateIssuerAliasCommand(
            instrument_id=instrument_id,
            alias=self.alias,
            alias_type=self.alias_type,
            priority=self.priority,
            is_active=self.is_active,
        )


class IssuerAliasResponse(BaseModel):
    id: UUID
    instrument_id: UUID
    alias: str
    normalized_alias: str
    alias_type: AliasType
    priority: int
    is_active: bool
    created_at: datetime

    @classmethod
    def from_entity(cls, alias: IssuerAlias) -> IssuerAliasResponse:
        return cls(**asdict(alias))


class NewsInstrumentMatchResponse(BaseModel):
    id: UUID
    news_id: UUID
    instrument_id: UUID
    matched_alias: str
    alias_type: AliasType
    match_type: MatchType
    confidence: float
    start_position: int
    end_position: int
    is_ambiguous: bool
    created_at: datetime
    matcher_version: str

    @classmethod
    def from_entity(cls, match: NewsInstrumentMatch) -> NewsInstrumentMatchResponse:
        return cls(**asdict(match))


class MatchNewsInstrumentsResponse(BaseModel):
    news_id: UUID
    matcher_version: str
    matches: list[NewsInstrumentMatchResponse]
