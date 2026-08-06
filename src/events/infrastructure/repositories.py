from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.events.application.exceptions import EventAnalysisStorageError
from src.events.domain.entities import NewsEventAnalysis
from src.events.infrastructure.models import NewsEventAnalysisRecord


class SqlAlchemyEventAnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_analysis(self, analysis: NewsEventAnalysis) -> NewsEventAnalysis:
        try:
            await self._session.execute(
                delete(NewsEventAnalysisRecord).where(
                    NewsEventAnalysisRecord.news_id == analysis.news_id,
                    NewsEventAnalysisRecord.analysis_version == analysis.analysis_version,
                )
            )
            self._session.add(NewsEventAnalysisRecord.from_entity(analysis))
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            saved = await self.get_by_news_id(
                news_id=analysis.news_id,
                analysis_version=analysis.analysis_version,
            )
            if saved is not None:
                return saved
            raise EventAnalysisStorageError("could not replace event analysis") from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise EventAnalysisStorageError("could not replace event analysis") from exc
        saved = await self.get_by_news_id(
            news_id=analysis.news_id,
            analysis_version=analysis.analysis_version,
        )
        if saved is None:
            raise EventAnalysisStorageError("event analysis save could not be resolved")
        return saved

    async def get_by_news_id(
        self,
        *,
        news_id: UUID,
        analysis_version: str | None = None,
    ) -> NewsEventAnalysis | None:
        query = select(NewsEventAnalysisRecord).where(NewsEventAnalysisRecord.news_id == news_id)
        if analysis_version is not None:
            query = query.where(NewsEventAnalysisRecord.analysis_version == analysis_version)
        try:
            result = await self._session.execute(
                query.options(
                    selectinload(NewsEventAnalysisRecord.events),
                    selectinload(NewsEventAnalysisRecord.financial_facts),
                ).order_by(NewsEventAnalysisRecord.analyzed_at.desc())
            )
        except SQLAlchemyError as exc:
            raise EventAnalysisStorageError("could not read event analysis") from exc
        record = result.scalars().first()
        return None if record is None else record.to_entity()
