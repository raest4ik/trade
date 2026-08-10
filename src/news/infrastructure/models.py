from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class NewsItemRecord(Base):
    __tablename__ = "news_items"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "source_url",
            "raw_content_hash",
            name="uq_news_items_source_url_hash",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(String(255), index=True)
    source_name: Mapped[str] = mapped_column(String(255))
    source_url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(String(500))
    raw_content: Mapped[str] = mapped_column(Text)
    raw_content_hash: Mapped[str] = mapped_column(String(64), index=True)
    language: Mapped[str] = mapped_column(String(16))
    published_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    publication_timestamp_quality: Mapped[str] = mapped_column(
        String(16), default=PublicationTimestampQuality.UNKNOWN.value, index=True
    )
    received_at: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())

    @classmethod
    def from_entity(cls, item: NewsItem) -> NewsItemRecord:
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
            publication_timestamp_quality=item.publication_timestamp_quality.value,
            received_at=item.received_at,
            created_at=item.created_at,
        )

    def to_entity(self) -> NewsItem:
        return NewsItem(
            id=self.id,
            source_id=self.source_id,
            source_name=self.source_name,
            source_url=self.source_url,
            title=self.title,
            raw_content=self.raw_content,
            raw_content_hash=self.raw_content_hash,
            language=self.language,
            published_at=self.published_at,
            publication_timestamp_quality=PublicationTimestampQuality(
                self.publication_timestamp_quality
            ),
            received_at=self.received_at,
            created_at=self.created_at,
        )
