from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.historical_news.domain.entities import (
    HistoricalNewsCandidate,
    HistoricalNewsImportRun,
    HistoricalNewsSource,
)
from src.historical_news.domain.enums import (
    ContentStoragePolicy,
    HistoricalNewsCandidateStatus,
    HistoricalNewsImportStatus,
    HistoricalNewsSourceKind,
)
from src.news.domain.enums import PublicationTimestampQuality
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class HistoricalNewsSourceRecord(Base):
    __tablename__ = "historical_news_sources"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_code: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_kind: Mapped[str] = mapped_column(String(32))
    content_storage_policy: Mapped[str] = mapped_column(String(32))
    source_timezone: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feed_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())

    @classmethod
    def from_entity(cls, item: HistoricalNewsSource) -> HistoricalNewsSourceRecord:
        return cls(
            id=item.id,
            source_code=item.source_code,
            source_kind=item.source_kind.value,
            content_storage_policy=item.content_storage_policy.value,
            source_timezone=item.source_timezone,
            feed_url=item.feed_url,
            created_at=item.created_at,
        )

    def to_entity(self) -> HistoricalNewsSource:
        return HistoricalNewsSource(
            id=self.id,
            source_code=self.source_code,
            source_kind=HistoricalNewsSourceKind(self.source_kind),
            content_storage_policy=ContentStoragePolicy(self.content_storage_policy),
            source_timezone=self.source_timezone,
            feed_url=self.feed_url,
            created_at=self.created_at,
        )


class HistoricalNewsImportRunRecord(Base):
    __tablename__ = "historical_news_import_runs"
    __table_args__ = (
        Index("ix_historical_news_import_runs_source_started", "source_id", "started_at"),
        Index("ix_historical_news_import_runs_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("historical_news_sources.id", ondelete="RESTRICT")
    )
    date_from: Mapped[datetime] = mapped_column(UtcDateTime())
    date_to: Mapped[datetime] = mapped_column(UtcDateTime())
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    discovered_count: Mapped[int] = mapped_column(Integer)
    validated_count: Mapped[int] = mapped_column(Integer)
    imported_count: Mapped[int] = mapped_column(Integer)
    duplicate_count: Mapped[int] = mapped_column(Integer)
    rejected_count: Mapped[int] = mapped_column(Integer)
    metadata_only_count: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    @classmethod
    def from_entity(cls, item: HistoricalNewsImportRun) -> HistoricalNewsImportRunRecord:
        return cls(
            id=item.id,
            source_id=item.source_id,
            date_from=item.date_from,
            date_to=item.date_to,
            started_at=item.started_at,
            finished_at=item.finished_at,
            status=item.status.value,
            discovered_count=item.discovered_count,
            validated_count=item.validated_count,
            imported_count=item.imported_count,
            duplicate_count=item.duplicate_count,
            rejected_count=item.rejected_count,
            metadata_only_count=item.metadata_only_count,
            error=item.error,
        )

    def update_from_entity(self, item: HistoricalNewsImportRun) -> None:
        self.finished_at = item.finished_at
        self.status = item.status.value
        self.discovered_count = item.discovered_count
        self.validated_count = item.validated_count
        self.imported_count = item.imported_count
        self.duplicate_count = item.duplicate_count
        self.rejected_count = item.rejected_count
        self.metadata_only_count = item.metadata_only_count
        self.error = item.error

    def to_entity(self) -> HistoricalNewsImportRun:
        return HistoricalNewsImportRun(
            id=self.id,
            source_id=self.source_id,
            date_from=self.date_from,
            date_to=self.date_to,
            started_at=self.started_at,
            finished_at=self.finished_at,
            status=HistoricalNewsImportStatus(self.status),
            discovered_count=self.discovered_count,
            validated_count=self.validated_count,
            imported_count=self.imported_count,
            duplicate_count=self.duplicate_count,
            rejected_count=self.rejected_count,
            metadata_only_count=self.metadata_only_count,
            error=self.error,
        )


class HistoricalNewsCandidateRecord(Base):
    __tablename__ = "historical_news_candidates"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "source_item_id",
            name="uq_historical_news_candidates_source_item",
        ),
        Index(
            "ix_historical_news_candidates_source_status",
            "source_id",
            "status",
        ),
        Index(
            "ix_historical_news_candidates_quality_status",
            "publication_timestamp_quality",
            "status",
        ),
        Index("ix_historical_news_candidates_content_hash", "content_hash"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("historical_news_sources.id", ondelete="RESTRICT")
    )
    ingestion_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("historical_news_import_runs.id", ondelete="RESTRICT")
    )
    source_item_id: Mapped[str] = mapped_column(String(512))
    source_url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(String(1000))
    source_published_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    source_timezone: Mapped[str | None] = mapped_column(String(128), nullable=True)
    publication_timestamp_quality: Mapped[str] = mapped_column(String(16))
    original_timestamp_text: Mapped[str] = mapped_column(String(256))
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime())
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_storage_policy: Mapped[str] = mapped_column(String(32))
    content_is_excerpt: Mapped[bool] = mapped_column(Boolean)
    exact_content_duplicate: Mapped[bool] = mapped_column(Boolean)
    corrects_source_item_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    supersedes_candidate_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("historical_news_candidates.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32))
    rejection_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    imported_news_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("news_items.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime())

    @classmethod
    def from_entity(cls, item: HistoricalNewsCandidate) -> HistoricalNewsCandidateRecord:
        return cls(
            id=item.id,
            source_id=item.source_id,
            ingestion_run_id=item.ingestion_run_id,
            source_item_id=item.source_item_id,
            source_url=item.source_url,
            title=item.title,
            source_published_at=item.source_published_at,
            source_timezone=item.source_timezone,
            publication_timestamp_quality=item.publication_timestamp_quality.value,
            original_timestamp_text=item.original_timestamp_text,
            fetched_at=item.fetched_at,
            content=item.content,
            content_hash=item.content_hash,
            content_storage_policy=item.content_storage_policy.value,
            content_is_excerpt=item.content_is_excerpt,
            exact_content_duplicate=item.exact_content_duplicate,
            corrects_source_item_id=item.corrects_source_item_id,
            supersedes_candidate_id=item.supersedes_candidate_id,
            status=item.status.value,
            rejection_reason=item.rejection_reason,
            imported_news_id=item.imported_news_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    def update_from_entity(self, item: HistoricalNewsCandidate) -> None:
        self.status = item.status.value
        self.rejection_reason = item.rejection_reason
        self.imported_news_id = item.imported_news_id
        self.updated_at = item.updated_at

    def to_entity(self) -> HistoricalNewsCandidate:
        return HistoricalNewsCandidate(
            id=self.id,
            source_id=self.source_id,
            ingestion_run_id=self.ingestion_run_id,
            source_item_id=self.source_item_id,
            source_url=self.source_url,
            title=self.title,
            source_published_at=self.source_published_at,
            source_timezone=self.source_timezone,
            publication_timestamp_quality=PublicationTimestampQuality(
                self.publication_timestamp_quality
            ),
            original_timestamp_text=self.original_timestamp_text,
            fetched_at=self.fetched_at,
            content=self.content,
            content_hash=self.content_hash,
            content_storage_policy=ContentStoragePolicy(self.content_storage_policy),
            content_is_excerpt=self.content_is_excerpt,
            exact_content_duplicate=self.exact_content_duplicate,
            corrects_source_item_id=self.corrects_source_item_id,
            supersedes_candidate_id=self.supersedes_candidate_id,
            status=HistoricalNewsCandidateStatus(self.status),
            rejection_reason=self.rejection_reason,
            imported_news_id=self.imported_news_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
