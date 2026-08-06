from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from src.news.domain.entities import NewsItem


@dataclass(frozen=True, slots=True)
class SaveNewsItemResult:
    item: NewsItem
    created: bool


class NewsRepository(Protocol):
    async def save(self, item: NewsItem) -> SaveNewsItemResult: ...

    async def get_by_id(self, news_id: UUID) -> NewsItem | None: ...
