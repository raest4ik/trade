from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.infrastructure.moex_client import MoexIssClient
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository


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


def get_market_data_repository(
    session: AsyncSession = Depends(get_session),
) -> SqlAlchemyMarketDataRepository:
    return SqlAlchemyMarketDataRepository(session)


def get_reaction_repository(
    session: AsyncSession = Depends(get_session),
) -> SqlAlchemyReactionRepository:
    return SqlAlchemyReactionRepository(session)


async def get_moex_client(request: Request) -> AsyncIterator[MoexIssClient]:
    settings = request.app.state.settings
    async with MoexIssClient(
        base_url=settings.moex_iss_base_url,
        timeout_seconds=settings.moex_http_timeout_seconds,
        max_retries=settings.moex_http_max_retries,
        max_pages=settings.moex_http_max_pages,
        user_agent=settings.moex_http_user_agent,
    ) as client:
        yield client
