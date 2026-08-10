from __future__ import annotations

from typing import Literal

from pydantic import AnyUrl, BaseModel, Field, field_validator

from src.historical_news.domain.entities import HistoricalSourceItem
from src.historical_news.domain.enums import ContentStoragePolicy
from src.news.domain.time import utc_now

HISTORICAL_SOURCE_SCHEMA_VERSION = "historical-news-source-v1"


class HistoricalNewsSourceItemV1(BaseModel):
    schema_version: Literal["historical-news-source-v1"] = HISTORICAL_SOURCE_SCHEMA_VERSION
    source_item_id: str = Field(min_length=1, max_length=512)
    source_url: AnyUrl = Field(max_length=2048)
    title: str = Field(min_length=1, max_length=1000)
    published_at: str = Field(min_length=1, max_length=128)
    source_timezone: str | None = Field(default=None, max_length=128)
    content: str | None = None
    content_storage_policy: ContentStoragePolicy
    content_is_excerpt: bool = False
    corrects_source_item_id: str | None = Field(default=None, max_length=512)

    @field_validator("source_item_id", "title", "published_at")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    def to_source_item(self) -> HistoricalSourceItem:
        return HistoricalSourceItem(
            source_item_id=self.source_item_id,
            source_url=str(self.source_url),
            title=self.title,
            published_at_text=self.published_at,
            source_timezone=self.source_timezone,
            content=self.content,
            content_storage_policy=self.content_storage_policy,
            content_is_excerpt=self.content_is_excerpt,
            original_timestamp_text=self.published_at,
            corrects_source_item_id=self.corrects_source_item_id,
            fetched_at=utc_now(),
        )
