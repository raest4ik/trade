from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.historical_news.application.exceptions import HistoricalNewsStorageError
from src.historical_news.domain.entities import (
    HistoricalNewsCandidate,
    HistoricalNewsImportRun,
    HistoricalNewsSource,
)
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsImportRunRecord,
    HistoricalNewsSourceRecord,
)


class SqlAlchemyHistoricalNewsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_source(self, source: HistoricalNewsSource) -> HistoricalNewsSource:
        existing = await self.get_source_by_code(source.source_code)
        if existing is not None:
            if (
                existing.source_kind != source.source_kind
                or existing.content_storage_policy != source.content_storage_policy
                or existing.source_timezone != source.source_timezone
                or existing.feed_url != source.feed_url
            ):
                raise HistoricalNewsStorageError(
                    "source_code already exists with different immutable configuration"
                )
            return existing
        record = HistoricalNewsSourceRecord.from_entity(source)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self.get_source_by_code(source.source_code)
            if existing is None:
                raise HistoricalNewsStorageError("source uniqueness conflict") from exc
            return existing
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise HistoricalNewsStorageError("could not save historical source") from exc
        return record.to_entity()

    async def get_source_by_code(self, source_code: str) -> HistoricalNewsSource | None:
        try:
            result = await self._session.execute(
                select(HistoricalNewsSourceRecord).where(
                    HistoricalNewsSourceRecord.source_code == source_code.strip().upper()
                )
            )
        except SQLAlchemyError as exc:
            raise HistoricalNewsStorageError("could not read historical source") from exc
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def create_import_run(self, run: HistoricalNewsImportRun) -> HistoricalNewsImportRun:
        record = HistoricalNewsImportRunRecord.from_entity(run)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise HistoricalNewsStorageError("could not create historical import run") from exc
        return record.to_entity()

    async def finish_import_run(self, run: HistoricalNewsImportRun) -> HistoricalNewsImportRun:
        try:
            result = await self._session.execute(
                select(HistoricalNewsImportRunRecord).where(
                    HistoricalNewsImportRunRecord.id == run.id
                )
            )
            record = result.scalar_one()
            record.update_from_entity(run)
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise HistoricalNewsStorageError("could not finish historical import run") from exc
        return record.to_entity()

    async def get_candidate(
        self,
        *,
        source_id: UUID,
        source_item_id: str,
    ) -> HistoricalNewsCandidate | None:
        try:
            result = await self._session.execute(
                select(HistoricalNewsCandidateRecord).where(
                    HistoricalNewsCandidateRecord.source_id == source_id,
                    HistoricalNewsCandidateRecord.source_item_id == source_item_id,
                )
            )
        except SQLAlchemyError as exc:
            raise HistoricalNewsStorageError("could not read historical candidate") from exc
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def find_content_duplicate(
        self,
        *,
        content_hash: str,
        excluding_source_id: UUID,
    ) -> HistoricalNewsCandidate | None:
        try:
            result = await self._session.execute(
                select(HistoricalNewsCandidateRecord)
                .where(
                    HistoricalNewsCandidateRecord.content_hash == content_hash,
                    HistoricalNewsCandidateRecord.source_id != excluding_source_id,
                )
                .order_by(HistoricalNewsCandidateRecord.created_at)
            )
        except SQLAlchemyError as exc:
            raise HistoricalNewsStorageError("could not check content duplicate") from exc
        record = result.scalars().first()
        return None if record is None else record.to_entity()

    async def save_candidate(
        self, candidate: HistoricalNewsCandidate
    ) -> tuple[HistoricalNewsCandidate, bool]:
        existing = await self.get_candidate(
            source_id=candidate.source_id,
            source_item_id=candidate.source_item_id,
        )
        if existing is not None:
            return existing, False
        record = HistoricalNewsCandidateRecord.from_entity(candidate)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self.get_candidate(
                source_id=candidate.source_id,
                source_item_id=candidate.source_item_id,
            )
            if existing is None:
                raise HistoricalNewsStorageError("candidate uniqueness conflict") from exc
            return existing, False
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise HistoricalNewsStorageError("could not save historical candidate") from exc
        return record.to_entity(), True

    async def update_candidate(self, candidate: HistoricalNewsCandidate) -> HistoricalNewsCandidate:
        try:
            result = await self._session.execute(
                select(HistoricalNewsCandidateRecord).where(
                    HistoricalNewsCandidateRecord.id == candidate.id
                )
            )
            record = result.scalar_one()
            record.update_from_entity(candidate)
            await self._session.commit()
            await self._session.refresh(record)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise HistoricalNewsStorageError("could not update historical candidate") from exc
        return record.to_entity()
