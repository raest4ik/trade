from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from src.instruments.domain.entities import (
    Instrument,
    IssuerAlias,
    MatchCandidate,
    NewsInstrumentMatch,
)


@dataclass(frozen=True, slots=True)
class SaveInstrumentResult:
    instrument: Instrument
    created: bool


@dataclass(frozen=True, slots=True)
class SaveIssuerAliasResult:
    alias: IssuerAlias
    created: bool


class InstrumentRepository(Protocol):
    async def save_instrument(self, instrument: Instrument) -> SaveInstrumentResult: ...

    async def list_instruments(self, limit: int, offset: int) -> list[Instrument]: ...

    async def get_instrument(self, instrument_id: UUID) -> Instrument | None: ...

    async def save_alias(self, alias: IssuerAlias) -> SaveIssuerAliasResult: ...

    async def list_match_candidates(self) -> list[MatchCandidate]: ...

    async def replace_news_matches(
        self,
        news_id: UUID,
        matcher_version: str,
        matches: list[NewsInstrumentMatch],
    ) -> list[NewsInstrumentMatch]: ...

    async def get_news_matches(
        self,
        news_id: UUID,
        matcher_version: str | None = None,
    ) -> list[NewsInstrumentMatch]: ...
