from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.instruments.application.exceptions import (
    InstrumentNotFoundError,
    NewsForMatchingNotFoundError,
)
from src.instruments.application.ports import (
    InstrumentRepository,
    SaveInstrumentResult,
    SaveIssuerAliasResult,
)
from src.instruments.domain.entities import (
    DEFAULT_MATCHER_VERSION,
    Instrument,
    InstrumentMatch,
    IssuerAlias,
    NewsInstrumentMatch,
)
from src.instruments.domain.enums import AliasType, InstrumentType
from src.instruments.domain.matcher import InstrumentMatcher
from src.news.application.ports import NewsRepository


@dataclass(frozen=True, slots=True)
class CreateInstrumentCommand:
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
    primary_board: str | None = None


@dataclass(frozen=True, slots=True)
class CreateIssuerAliasCommand:
    instrument_id: UUID
    alias: str
    alias_type: AliasType
    priority: int
    is_active: bool


@dataclass(frozen=True, slots=True)
class MatchNewsInstrumentsResult:
    news_id: UUID
    matcher_version: str
    matches: list[NewsInstrumentMatch]


class CreateInstrument:
    def __init__(self, repository: InstrumentRepository) -> None:
        self._repository = repository

    async def execute(self, command: CreateInstrumentCommand) -> SaveInstrumentResult:
        instrument = Instrument.create(
            ticker=command.ticker,
            figi=command.figi,
            isin=command.isin,
            short_name=command.short_name,
            full_name=command.full_name,
            issuer_name=command.issuer_name,
            exchange=command.exchange,
            currency=command.currency,
            instrument_type=command.instrument_type,
            primary_board=command.primary_board,
            is_active=command.is_active,
        )
        return await self._repository.save_instrument(instrument)


class ListInstruments:
    def __init__(self, repository: InstrumentRepository) -> None:
        self._repository = repository

    async def execute(self, limit: int, offset: int) -> list[Instrument]:
        return await self._repository.list_instruments(limit=limit, offset=offset)


class CreateIssuerAlias:
    def __init__(self, repository: InstrumentRepository) -> None:
        self._repository = repository

    async def execute(self, command: CreateIssuerAliasCommand) -> SaveIssuerAliasResult:
        instrument = await self._repository.get_instrument(command.instrument_id)
        if instrument is None:
            raise InstrumentNotFoundError("instrument not found")
        alias = IssuerAlias.create(
            instrument_id=command.instrument_id,
            alias=command.alias,
            alias_type=command.alias_type,
            priority=command.priority,
            is_active=command.is_active,
        )
        return await self._repository.save_alias(alias)


class MatchNewsInstruments:
    def __init__(
        self,
        news_repository: NewsRepository,
        instrument_repository: InstrumentRepository,
        matcher: InstrumentMatcher | None = None,
        matcher_version: str = DEFAULT_MATCHER_VERSION,
    ) -> None:
        self._news_repository = news_repository
        self._instrument_repository = instrument_repository
        self._matcher = matcher or InstrumentMatcher()
        self._matcher_version = matcher_version

    async def execute(self, news_id: UUID) -> MatchNewsInstrumentsResult:
        news_item = await self._news_repository.get_by_id(news_id)
        if news_item is None:
            raise NewsForMatchingNotFoundError("news item not found")

        candidates = await self._instrument_repository.list_match_candidates()
        domain_matches: list[InstrumentMatch] = self._matcher.match(
            news_item.raw_content, candidates
        )
        saved_matches = await self._instrument_repository.replace_news_matches(
            news_id=news_id,
            matcher_version=self._matcher_version,
            matches=[
                NewsInstrumentMatch.create(
                    news_id=news_id,
                    match=match,
                    matcher_version=self._matcher_version,
                )
                for match in domain_matches
            ],
        )
        return MatchNewsInstrumentsResult(
            news_id=news_id,
            matcher_version=self._matcher_version,
            matches=saved_matches,
        )


class GetNewsInstrumentMatches:
    def __init__(self, repository: InstrumentRepository) -> None:
        self._repository = repository

    async def execute(self, news_id: UUID) -> MatchNewsInstrumentsResult:
        matches = await self._repository.get_news_matches(news_id)
        matcher_version = matches[0].matcher_version if matches else DEFAULT_MATCHER_VERSION
        return MatchNewsInstrumentsResult(
            news_id=news_id,
            matcher_version=matcher_version,
            matches=matches,
        )
