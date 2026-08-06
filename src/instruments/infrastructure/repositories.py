from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.instruments.application.exceptions import InstrumentStorageError
from src.instruments.application.ports import SaveInstrumentResult, SaveIssuerAliasResult
from src.instruments.domain.entities import (
    Instrument,
    IssuerAlias,
    MatchCandidate,
    NewsInstrumentMatch,
)
from src.instruments.domain.enums import AliasType
from src.instruments.infrastructure.models import (
    InstrumentRecord,
    IssuerAliasRecord,
    NewsInstrumentMatchRecord,
)


class SqlAlchemyInstrumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_instrument(self, instrument: Instrument) -> SaveInstrumentResult:
        record = InstrumentRecord.from_entity(instrument)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self._get_by_exchange_ticker(instrument.exchange, instrument.ticker)
            if existing is None:
                raise InstrumentStorageError("instrument uniqueness conflict") from exc
            return SaveInstrumentResult(instrument=existing, created=False)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise InstrumentStorageError("could not save instrument") from exc
        return SaveInstrumentResult(instrument=record.to_entity(), created=True)

    async def list_instruments(self, limit: int, offset: int) -> list[Instrument]:
        try:
            result = await self._session.execute(
                select(InstrumentRecord)
                .order_by(InstrumentRecord.exchange, InstrumentRecord.ticker)
                .limit(limit)
                .offset(offset)
            )
        except SQLAlchemyError as exc:
            raise InstrumentStorageError("could not list instruments") from exc
        return [record.to_entity() for record in result.scalars()]

    async def get_instrument(self, instrument_id: UUID) -> Instrument | None:
        try:
            result = await self._session.execute(
                select(InstrumentRecord).where(InstrumentRecord.id == instrument_id)
            )
        except SQLAlchemyError as exc:
            raise InstrumentStorageError("could not read instrument") from exc
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def save_alias(self, alias: IssuerAlias) -> SaveIssuerAliasResult:
        record = IssuerAliasRecord.from_entity(alias)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self._get_alias_by_unique_key(
                alias.instrument_id, alias.normalized_alias
            )
            if existing is None:
                raise InstrumentStorageError("issuer alias uniqueness conflict") from exc
            return SaveIssuerAliasResult(alias=existing, created=False)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise InstrumentStorageError("could not save issuer alias") from exc
        return SaveIssuerAliasResult(alias=record.to_entity(), created=True)

    async def list_match_candidates(self) -> list[MatchCandidate]:
        try:
            result = await self._session.execute(
                select(IssuerAliasRecord, InstrumentRecord)
                .join(InstrumentRecord, IssuerAliasRecord.instrument_id == InstrumentRecord.id)
                .where(IssuerAliasRecord.is_active.is_(True), InstrumentRecord.is_active.is_(True))
                .order_by(IssuerAliasRecord.priority, IssuerAliasRecord.normalized_alias)
            )
        except SQLAlchemyError as exc:
            raise InstrumentStorageError("could not list match candidates") from exc
        return [
            MatchCandidate(
                instrument_id=instrument.id,
                ticker=instrument.ticker,
                issuer_name=instrument.issuer_name,
                matched_alias=alias.alias,
                normalized_alias=alias.normalized_alias,
                alias_type=AliasType(alias.alias_type),
                priority=alias.priority,
            )
            for alias, instrument in result.all()
        ]

    async def replace_news_matches(
        self,
        news_id: UUID,
        matcher_version: str,
        matches: list[NewsInstrumentMatch],
    ) -> list[NewsInstrumentMatch]:
        try:
            await self._session.execute(
                delete(NewsInstrumentMatchRecord).where(
                    NewsInstrumentMatchRecord.news_id == news_id,
                    NewsInstrumentMatchRecord.matcher_version == matcher_version,
                )
            )
            self._session.add_all(
                NewsInstrumentMatchRecord.from_entity(match.ensure_utc()) for match in matches
            )
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise InstrumentStorageError("could not replace news instrument matches") from exc
        return await self.get_news_matches(news_id, matcher_version)

    async def get_news_matches(
        self,
        news_id: UUID,
        matcher_version: str | None = None,
    ) -> list[NewsInstrumentMatch]:
        query = select(NewsInstrumentMatchRecord).where(
            NewsInstrumentMatchRecord.news_id == news_id
        )
        if matcher_version is not None:
            query = query.where(NewsInstrumentMatchRecord.matcher_version == matcher_version)
        try:
            result = await self._session.execute(
                query.order_by(
                    NewsInstrumentMatchRecord.start_position,
                    NewsInstrumentMatchRecord.instrument_id,
                )
            )
        except SQLAlchemyError as exc:
            raise InstrumentStorageError("could not read news instrument matches") from exc
        return [record.to_entity() for record in result.scalars()]

    async def _get_by_exchange_ticker(self, exchange: str, ticker: str) -> Instrument | None:
        result = await self._session.execute(
            select(InstrumentRecord).where(
                InstrumentRecord.exchange == exchange,
                InstrumentRecord.ticker == ticker,
            )
        )
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def _get_alias_by_unique_key(
        self,
        instrument_id: UUID,
        normalized_alias: str,
    ) -> IssuerAlias | None:
        result = await self._session.execute(
            select(IssuerAliasRecord).where(
                IssuerAliasRecord.instrument_id == instrument_id,
                IssuerAliasRecord.normalized_alias == normalized_alias,
            )
        )
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()
