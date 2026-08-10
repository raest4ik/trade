from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from src.news.application.use_cases import CreateNewsItemCommand
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.news.domain.time import ensure_aware_utc


class NewsCreateRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=255)
    source_name: str = Field(min_length=1, max_length=255)
    source_url: AnyUrl
    title: str = Field(min_length=1, max_length=500)
    raw_content: str = Field(min_length=1)
    language: str = Field(min_length=1, max_length=16)
    published_at: datetime
    received_at: datetime | None = None
    publication_timestamp_quality: PublicationTimestampQuality = PublicationTimestampQuality.UNKNOWN

    @field_validator("source_id", "source_name", "title", "language")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("raw_content")
    @classmethod
    def reject_blank_raw_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_content must not be empty")
        return value

    @field_validator("published_at", "received_at")
    @classmethod
    def require_timezone(cls, value: datetime | None, info: ValidationInfo) -> datetime | None:
        if value is None:
            return None
        return ensure_aware_utc(value, info.field_name or "datetime")

    def to_command(self) -> CreateNewsItemCommand:
        return CreateNewsItemCommand(
            source_id=self.source_id,
            source_name=self.source_name,
            source_url=str(self.source_url),
            title=self.title,
            raw_content=self.raw_content,
            language=self.language,
            published_at=self.published_at,
            received_at=self.received_at,
            publication_timestamp_quality=self.publication_timestamp_quality,
        )


class NewsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_id: str
    source_name: str
    source_url: str
    title: str
    raw_content: str
    raw_content_hash: str
    language: str
    published_at: datetime
    publication_timestamp_quality: PublicationTimestampQuality
    received_at: datetime
    created_at: datetime

    @classmethod
    def from_entity(cls, item: NewsItem) -> NewsResponse:
        return cls(
            id=item.id,
            source_id=item.source_id,
            source_name=item.source_name,
            source_url=item.source_url,
            title=item.title,
            raw_content=item.raw_content,
            raw_content_hash=item.raw_content_hash,
            language=item.language,
            published_at=item.published_at,
            publication_timestamp_quality=item.publication_timestamp_quality,
            received_at=item.received_at,
            created_at=item.created_at,
        )
