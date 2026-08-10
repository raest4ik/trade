from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.reactions.application.exceptions import ReactionStorageError
from src.reactions.domain.entities import NewsMarketReaction
from src.reactions.infrastructure.models import (
    NewsMarketReactionRecord,
    ReactionBenchmarkAdjustmentRecord,
    ReactionPointRecord,
)


class SqlAlchemyReactionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_reactions(
        self,
        *,
        news_id: UUID,
        reaction_version: str,
        reactions: list[NewsMarketReaction],
    ) -> list[NewsMarketReaction]:
        try:
            reaction_ids = select(NewsMarketReactionRecord.id).where(
                NewsMarketReactionRecord.news_id == news_id,
                NewsMarketReactionRecord.reaction_version == reaction_version,
            )
            point_ids = select(ReactionPointRecord.id).where(
                ReactionPointRecord.reaction_id.in_(reaction_ids)
            )
            await self._session.execute(
                delete(ReactionBenchmarkAdjustmentRecord).where(
                    ReactionBenchmarkAdjustmentRecord.reaction_point_id.in_(point_ids)
                )
            )
            await self._session.execute(
                delete(ReactionPointRecord).where(ReactionPointRecord.reaction_id.in_(reaction_ids))
            )
            await self._session.execute(
                delete(NewsMarketReactionRecord).where(
                    NewsMarketReactionRecord.news_id == news_id,
                    NewsMarketReactionRecord.reaction_version == reaction_version,
                )
            )
            self._session.add_all(NewsMarketReactionRecord.from_entity(item) for item in reactions)
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise ReactionStorageError("could not replace news market reactions") from exc
        return await self.get_news_reactions(news_id=news_id, reaction_version=reaction_version)

    async def get_news_reactions(
        self,
        *,
        news_id: UUID,
        reaction_version: str | None = None,
    ) -> list[NewsMarketReaction]:
        query = select(NewsMarketReactionRecord).where(NewsMarketReactionRecord.news_id == news_id)
        if reaction_version is not None:
            query = query.where(NewsMarketReactionRecord.reaction_version == reaction_version)
        try:
            result = await self._session.execute(
                query.options(
                    selectinload(NewsMarketReactionRecord.points).selectinload(
                        ReactionPointRecord.benchmark_adjustment
                    )
                ).order_by(NewsMarketReactionRecord.instrument_id)
            )
        except SQLAlchemyError as exc:
            raise ReactionStorageError("could not read news market reactions") from exc
        return [record.to_entity() for record in result.scalars()]
