from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from apps.api.dependencies import get_event_analysis_repository, get_news_repository
from src.events.application.exceptions import (
    EventAnalysisNewsNotFoundError,
    EventAnalysisStorageError,
)
from src.events.application.use_cases import AnalyzeNewsEvent, GetNewsEventAnalysis
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.events.presentation.schemas import NewsEventAnalysisResponse
from src.news.application.exceptions import NewsStorageError
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository

router = APIRouter(tags=["events"])


@router.post(
    "/news/{news_id}/analyze-event",
    response_model=NewsEventAnalysisResponse,
)
async def analyze_news_event(
    news_id: UUID,
    debug: bool = Query(default=False),
    news_repository: SqlAlchemyNewsRepository = Depends(get_news_repository),
    event_repository: SqlAlchemyEventAnalysisRepository = Depends(get_event_analysis_repository),
) -> NewsEventAnalysisResponse:
    try:
        result = await AnalyzeNewsEvent(
            news_repository=news_repository,
            event_repository=event_repository,
        ).execute(news_id)
    except EventAnalysisNewsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="news item not found",
        ) from exc
    except (EventAnalysisStorageError, NewsStorageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="event analysis storage is unavailable",
        ) from exc
    return NewsEventAnalysisResponse.from_entity(result.analysis, include_debug=debug)


@router.get(
    "/news/{news_id}/event-analysis",
    response_model=NewsEventAnalysisResponse,
)
async def get_news_event_analysis(
    news_id: UUID,
    debug: bool = Query(default=False),
    repository: SqlAlchemyEventAnalysisRepository = Depends(get_event_analysis_repository),
) -> NewsEventAnalysisResponse:
    try:
        analysis = await GetNewsEventAnalysis(repository).execute(news_id)
    except EventAnalysisStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="event analysis storage is unavailable",
        ) from exc
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="event analysis not found",
        )
    return NewsEventAnalysisResponse.from_entity(analysis, include_debug=debug)
