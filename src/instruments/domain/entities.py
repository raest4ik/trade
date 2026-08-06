from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.instruments.domain.exceptions import InstrumentDomainError
from src.instruments.domain.normalization import normalize_text
from src.news.domain.time import ensure_aware_utc, utc_now

MIN_ALIAS_LENGTH = 3
DEFAULT_MATCHER_VERSION = "deterministic-v1"


@dataclass(frozen=True, slots=True)
class Instrument:
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
    def create(
        cls,
        *,
        ticker: str,
        figi: str | None,
        isin: str | None,
        short_name: str,
        full_name: str,
        issuer_name: str,
        exchange: str,
        currency: str,
        instrument_type: InstrumentType,
        is_active: bool = True,
    ) -> Instrument:
        normalized_ticker = ticker.strip().upper()
        cls._require_text("ticker", normalized_ticker)
        cls._require_text("short_name", short_name)
        cls._require_text("full_name", full_name)
        cls._require_text("issuer_name", issuer_name)
        cls._require_text("exchange", exchange)
        cls._require_text("currency", currency)
        if exchange.strip().upper() != "MOEX":
            raise InstrumentDomainError("exchange must be MOEX for this MVP")
        now = utc_now()
        return cls(
            id=uuid4(),
            ticker=normalized_ticker,
            figi=_blank_to_none(figi),
            isin=_blank_to_none(isin),
            short_name=short_name,
            full_name=full_name,
            issuer_name=issuer_name,
            exchange=exchange.strip().upper(),
            currency=currency.strip().upper(),
            instrument_type=instrument_type,
            is_active=is_active,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _require_text(field_name: str, value: str) -> None:
        if not value.strip():
            raise InstrumentDomainError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class IssuerAlias:
    id: UUID
    instrument_id: UUID
    alias: str
    normalized_alias: str
    alias_type: AliasType
    priority: int
    is_active: bool
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        instrument_id: UUID,
        alias: str,
        alias_type: AliasType,
        priority: int = 100,
        is_active: bool = True,
    ) -> IssuerAlias:
        if not alias.strip():
            raise InstrumentDomainError("alias must not be empty")
        normalized_alias = normalize_text(alias)
        if alias_type != AliasType.TICKER and len(normalized_alias) < MIN_ALIAS_LENGTH:
            raise InstrumentDomainError(
                "non-ticker aliases shorter than 3 characters are not allowed"
            )
        return cls(
            id=uuid4(),
            instrument_id=instrument_id,
            alias=alias,
            normalized_alias=normalized_alias,
            alias_type=alias_type,
            priority=priority,
            is_active=is_active,
            created_at=utc_now(),
        )


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    instrument_id: UUID
    ticker: str
    issuer_name: str
    matched_alias: str
    normalized_alias: str
    alias_type: AliasType
    priority: int


@dataclass(frozen=True, slots=True)
class InstrumentMatch:
    instrument_id: UUID
    ticker: str
    issuer_name: str
    matched_alias: str
    alias_type: AliasType
    match_type: MatchType
    confidence: float
    start_position: int
    end_position: int
    is_ambiguous: bool


@dataclass(frozen=True, slots=True)
class NewsInstrumentMatch:
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
    def create(
        cls,
        *,
        news_id: UUID,
        match: InstrumentMatch,
        matcher_version: str = DEFAULT_MATCHER_VERSION,
    ) -> NewsInstrumentMatch:
        return cls(
            id=uuid4(),
            news_id=news_id,
            instrument_id=match.instrument_id,
            matched_alias=match.matched_alias,
            alias_type=match.alias_type,
            match_type=match.match_type,
            confidence=match.confidence,
            start_position=match.start_position,
            end_position=match.end_position,
            is_ambiguous=match.is_ambiguous,
            created_at=utc_now(),
            matcher_version=matcher_version,
        )

    def ensure_utc(self) -> NewsInstrumentMatch:
        return NewsInstrumentMatch(
            id=self.id,
            news_id=self.news_id,
            instrument_id=self.instrument_id,
            matched_alias=self.matched_alias,
            alias_type=self.alias_type,
            match_type=self.match_type,
            confidence=self.confidence,
            start_position=self.start_position,
            end_position=self.end_position,
            is_ambiguous=self.is_ambiguous,
            created_at=ensure_aware_utc(self.created_at, "created_at"),
            matcher_version=self.matcher_version,
        )


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
