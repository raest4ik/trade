from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID, uuid4

from src.historical_news.domain.enums import (
    ContentStoragePolicy,
    HistoricalNewsCandidateStatus,
    HistoricalNewsImportStatus,
    HistoricalNewsSourceKind,
)
from src.news.domain.enums import PublicationTimestampQuality
from src.news.domain.time import ensure_aware_utc, utc_now


class HistoricalNewsDomainError(ValueError):
    """Raised when historical news domain data is invalid."""


@dataclass(frozen=True, slots=True)
class HistoricalNewsSource:
    id: UUID
    source_code: str
    source_kind: HistoricalNewsSourceKind
    content_storage_policy: ContentStoragePolicy
    source_timezone: str | None
    feed_url: str | None
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        source_code: str,
        source_kind: HistoricalNewsSourceKind,
        content_storage_policy: ContentStoragePolicy,
        source_timezone: str | None = None,
        feed_url: str | None = None,
    ) -> HistoricalNewsSource:
        normalized_code = source_code.strip().upper()
        if not normalized_code:
            raise HistoricalNewsDomainError("source_code must not be empty")
        return cls(
            id=uuid4(),
            source_code=normalized_code,
            source_kind=source_kind,
            content_storage_policy=content_storage_policy,
            source_timezone=None if source_timezone is None else source_timezone.strip() or None,
            feed_url=None if feed_url is None else feed_url.strip() or None,
            created_at=utc_now(),
        )


@dataclass(frozen=True, slots=True)
class HistoricalSourceItem:
    source_item_id: str
    source_url: str
    title: str
    published_at_text: str
    source_timezone: str | None
    content: str | None
    content_storage_policy: ContentStoragePolicy
    content_is_excerpt: bool
    original_timestamp_text: str
    corrects_source_item_id: str | None
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class HistoricalNewsPage:
    items: list[HistoricalSourceItem]
    next_cursor: str | None
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False


@dataclass(frozen=True, slots=True)
class HistoricalNewsCandidate:
    id: UUID
    source_id: UUID
    ingestion_run_id: UUID
    source_item_id: str
    source_url: str
    title: str
    source_published_at: datetime | None
    source_timezone: str | None
    publication_timestamp_quality: PublicationTimestampQuality
    original_timestamp_text: str
    fetched_at: datetime
    content: str | None
    content_hash: str | None
    content_storage_policy: ContentStoragePolicy
    content_is_excerpt: bool
    exact_content_duplicate: bool
    corrects_source_item_id: str | None
    supersedes_candidate_id: UUID | None
    status: HistoricalNewsCandidateStatus
    rejection_reason: str | None
    imported_news_id: UUID | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        source_id: UUID,
        ingestion_run_id: UUID,
        source_item_id: str,
        source_url: str,
        title: str,
        source_published_at: datetime | None,
        source_timezone: str | None,
        publication_timestamp_quality: PublicationTimestampQuality,
        original_timestamp_text: str,
        fetched_at: datetime,
        content: str | None,
        content_hash: str | None,
        content_storage_policy: ContentStoragePolicy,
        content_is_excerpt: bool,
        exact_content_duplicate: bool,
        corrects_source_item_id: str | None,
        supersedes_candidate_id: UUID | None,
        status: HistoricalNewsCandidateStatus,
        rejection_reason: str | None = None,
    ) -> HistoricalNewsCandidate:
        now = utc_now()
        return cls(
            id=uuid4(),
            source_id=source_id,
            ingestion_run_id=ingestion_run_id,
            source_item_id=source_item_id.strip(),
            source_url=source_url.strip(),
            title=title.strip(),
            source_published_at=None
            if source_published_at is None
            else ensure_aware_utc(source_published_at, "source_published_at"),
            source_timezone=None if source_timezone is None else source_timezone.strip() or None,
            publication_timestamp_quality=publication_timestamp_quality,
            original_timestamp_text=original_timestamp_text,
            fetched_at=ensure_aware_utc(fetched_at, "fetched_at"),
            content=content,
            content_hash=content_hash,
            content_storage_policy=content_storage_policy,
            content_is_excerpt=content_is_excerpt,
            exact_content_duplicate=exact_content_duplicate,
            corrects_source_item_id=corrects_source_item_id,
            supersedes_candidate_id=supersedes_candidate_id,
            status=status,
            rejection_reason=rejection_reason,
            imported_news_id=None,
            created_at=now,
            updated_at=now,
        )

    def mark_imported(
        self,
        news_id: UUID,
        *,
        duplicate: bool,
    ) -> HistoricalNewsCandidate:
        return replace(
            self,
            status=(
                HistoricalNewsCandidateStatus.DUPLICATE
                if duplicate
                else HistoricalNewsCandidateStatus.IMPORTED
            ),
            imported_news_id=news_id,
            updated_at=utc_now(),
        )


@dataclass(frozen=True, slots=True)
class HistoricalNewsImportRun:
    id: UUID
    source_id: UUID
    date_from: datetime
    date_to: datetime
    started_at: datetime
    finished_at: datetime | None
    status: HistoricalNewsImportStatus
    discovered_count: int
    validated_count: int
    imported_count: int
    duplicate_count: int
    rejected_count: int
    metadata_only_count: int
    error: str | None

    @classmethod
    def start(
        cls,
        *,
        source_id: UUID,
        date_from: datetime,
        date_to: datetime,
    ) -> HistoricalNewsImportRun:
        from_utc = ensure_aware_utc(date_from, "date_from")
        to_utc = ensure_aware_utc(date_to, "date_to")
        if to_utc < from_utc:
            raise HistoricalNewsDomainError("date_to must not be before date_from")
        return cls(
            id=uuid4(),
            source_id=source_id,
            date_from=from_utc,
            date_to=to_utc,
            started_at=utc_now(),
            finished_at=None,
            status=HistoricalNewsImportStatus.RUNNING,
            discovered_count=0,
            validated_count=0,
            imported_count=0,
            duplicate_count=0,
            rejected_count=0,
            metadata_only_count=0,
            error=None,
        )

    def finish(
        self,
        *,
        status: HistoricalNewsImportStatus,
        discovered_count: int,
        validated_count: int,
        imported_count: int,
        duplicate_count: int,
        rejected_count: int,
        metadata_only_count: int,
        error: str | None = None,
    ) -> HistoricalNewsImportRun:
        return replace(
            self,
            finished_at=utc_now(),
            status=status,
            discovered_count=discovered_count,
            validated_count=validated_count,
            imported_count=imported_count,
            duplicate_count=duplicate_count,
            rejected_count=rejected_count,
            metadata_only_count=metadata_only_count,
            error=error,
        )
