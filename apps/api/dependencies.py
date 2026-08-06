from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session


def get_news_repository(
    session: AsyncSession = Depends(get_session),
) -> SqlAlchemyNewsRepository:
    return SqlAlchemyNewsRepository(session)


def get_instrument_repository(
    session: AsyncSession = Depends(get_session),
) -> SqlAlchemyInstrumentRepository:
    return SqlAlchemyInstrumentRepository(session)
