from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from apps.api.dependencies import get_news_repository
from src.news.application.exceptions import NewsStorageError
from src.news.application.use_cases import CreateNewsItem, GetNewsItem
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.news.presentation.schemas import NewsCreateRequest, NewsResponse

router = APIRouter(tags=["news"])
logger = logging.getLogger(__name__)


@router.post(
    "/news",
    response_model=NewsResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_news(
    payload: NewsCreateRequest,
    response: Response,
    repository: SqlAlchemyNewsRepository = Depends(get_news_repository),
) -> NewsResponse:
    use_case = CreateNewsItem(repository)
    try:
        result = await use_case.execute(payload.to_command())
    except NewsStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="news storage is unavailable",
        ) from exc

    response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    logger.info(
        "news_saved",
        extra={"news_id": str(result.item.id), "deduplicated": not result.created},
    )
    return NewsResponse.from_entity(result.item)


@router.get("/news/{news_id}", response_model=NewsResponse)
async def get_news(
    news_id: UUID,
    repository: SqlAlchemyNewsRepository = Depends(get_news_repository),
) -> NewsResponse:
    use_case = GetNewsItem(repository)
    try:
        item = await use_case.execute(news_id)
    except NewsStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="news storage is unavailable",
        ) from exc

    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="news item not found")
    return NewsResponse.from_entity(item)
