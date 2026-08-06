from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.news.application.exceptions import NewsStorageError
from src.news.application.ports import SaveNewsItemResult
from src.news.domain.entities import NewsItem
from src.news.infrastructure.models import NewsItemRecord


class SqlAlchemyNewsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, item: NewsItem) -> SaveNewsItemResult:
        record = NewsItemRecord.from_entity(item)
        self._session.add(record)
        try:
            await self._session.commit()
            await self._session.refresh(record)
        except IntegrityError as exc:
            await self._session.rollback()
            existing = await self._get_by_unique_key(item)
            if existing is None:
                raise NewsStorageError("news uniqueness conflict could not be resolved") from exc
            return SaveNewsItemResult(item=existing, created=False)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise NewsStorageError("could not save news item") from exc

        return SaveNewsItemResult(item=record.to_entity(), created=True)

    async def get_by_id(self, news_id: UUID) -> NewsItem | None:
        try:
            result = await self._session.execute(
                select(NewsItemRecord).where(NewsItemRecord.id == news_id)
            )
        except SQLAlchemyError as exc:
            raise NewsStorageError("could not read news item") from exc

        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def _get_by_unique_key(self, item: NewsItem) -> NewsItem | None:
        result = await self._session.execute(
            select(NewsItemRecord).where(
                NewsItemRecord.source_id == item.source_id,
                NewsItemRecord.source_url == item.source_url,
                NewsItemRecord.raw_content_hash == item.raw_content_hash,
            )
        )
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()
