from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import (
    get_instrument_repository,
    get_market_data_repository,
    get_news_repository,
    get_reaction_repository,
)
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.exceptions import (
    ReactionMissingInstrumentMatchesError,
    ReactionNewsNotFoundError,
    ReactionStorageError,
    ReactionTimestampIneligibleError,
)
from src.reactions.application.use_cases import (
    CalculateNewsMarketReactions,
    GetNewsMarketReactions,
)
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.reactions.presentation.schemas import NewsReactionsResponse

router = APIRouter(tags=["reactions"])


@router.post(
    "/news/{news_id}/calculate-reactions",
    response_model=NewsReactionsResponse,
)
async def calculate_news_market_reactions(
    news_id: UUID,
    news_repository: SqlAlchemyNewsRepository = Depends(get_news_repository),
    instrument_repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
    market_data_repository: SqlAlchemyMarketDataRepository = Depends(get_market_data_repository),
    reaction_repository: SqlAlchemyReactionRepository = Depends(get_reaction_repository),
) -> NewsReactionsResponse:
    try:
        result = await CalculateNewsMarketReactions(
            news_repository=news_repository,
            instrument_repository=instrument_repository,
            market_data_repository=market_data_repository,
            reaction_repository=reaction_repository,
        ).execute(news_id)
    except ReactionNewsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="news item not found"
        ) from exc
    except ReactionMissingInstrumentMatchesError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="instrument matching was not run",
        ) from exc
    except ReactionTimestampIneligibleError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="trusted exact publication timestamp is required",
        ) from exc
    except ReactionStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="reaction storage is unavailable",
        ) from exc
    return NewsReactionsResponse.from_reactions(
        news_id=result.news_id,
        reaction_version=result.reaction_version,
        reactions=result.reactions,
    )


@router.get("/news/{news_id}/reactions", response_model=NewsReactionsResponse)
async def get_news_market_reactions(
    news_id: UUID,
    reaction_repository: SqlAlchemyReactionRepository = Depends(get_reaction_repository),
) -> NewsReactionsResponse:
    try:
        result = await GetNewsMarketReactions(reaction_repository).execute(news_id)
    except ReactionStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="reaction storage is unavailable",
        ) from exc
    return NewsReactionsResponse.from_reactions(
        news_id=result.news_id,
        reaction_version=result.reaction_version,
        reactions=result.reactions,
    )
