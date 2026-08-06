from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from src.news.domain.exceptions import DomainError
from src.news.domain.hash import calculate_raw_content_hash
from src.news.domain.time import ensure_aware_utc, utc_now


@dataclass(frozen=True, slots=True)
class NewsItem:
    id: UUID
    source_id: str
    source_name: str
    source_url: str
    title: str
    raw_content: str
    raw_content_hash: str
    language: str
    published_at: datetime
    received_at: datetime
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_name: str,
        source_url: str,
        title: str,
        raw_content: str,
        language: str,
        published_at: datetime,
        received_at: datetime | None = None,
    ) -> NewsItem:
        cls._validate_required_text("source_id", source_id)
        cls._validate_required_text("source_name", source_name)
        cls._validate_required_text("source_url", source_url)
        cls._validate_required_text("title", title)
        if not raw_content.strip():
            raise DomainError("raw_content must not be empty")
        cls._validate_required_text("language", language)

        normalized_published_at = ensure_aware_utc(published_at, "published_at")
        normalized_received_at = ensure_aware_utc(received_at or utc_now(), "received_at")
        created_at = utc_now()

        return cls(
            id=uuid4(),
            source_id=source_id,
            source_name=source_name,
            source_url=source_url,
            title=title,
            raw_content=raw_content,
            raw_content_hash=calculate_raw_content_hash(source_id, raw_content),
            language=language,
            published_at=normalized_published_at,
            received_at=normalized_received_at,
            created_at=created_at,
        )

    @staticmethod
    def _validate_required_text(field_name: str, value: str) -> None:
        if not value.strip():
            raise DomainError(f"{field_name} must not be empty")
