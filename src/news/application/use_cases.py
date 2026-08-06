from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.news.application.ports import NewsRepository, SaveNewsItemResult
from src.news.domain.entities import NewsItem


@dataclass(frozen=True, slots=True)
class CreateNewsItemCommand:
    source_id: str
    source_name: str
    source_url: str
    title: str
    raw_content: str
    language: str
    published_at: datetime
    received_at: datetime | None


class CreateNewsItem:
    def __init__(self, repository: NewsRepository) -> None:
        self._repository = repository

    async def execute(self, command: CreateNewsItemCommand) -> SaveNewsItemResult:
        item = NewsItem.create(
            source_id=command.source_id,
            source_name=command.source_name,
            source_url=command.source_url,
            title=command.title,
            raw_content=command.raw_content,
            language=command.language,
            published_at=command.published_at,
            received_at=command.received_at,
        )
        return await self._repository.save(item)


class GetNewsItem:
    def __init__(self, repository: NewsRepository) -> None:
        self._repository = repository

    async def execute(self, news_id: UUID) -> NewsItem | None:
        return await self._repository.get_by_id(news_id)
