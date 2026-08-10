from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.ml_features.domain.entities import MlFeatureDatasetRun
from src.ml_features.domain.enums import FeatureDatasetRunStatus
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class MlFeatureDatasetRunRecord(Base):
    __tablename__ = "ml_feature_dataset_runs"
    __table_args__ = (
        Index("ix_ml_feature_dataset_runs_started_at", "started_at"),
        Index("ix_ml_feature_dataset_runs_status", "status"),
        Index("ix_ml_feature_dataset_runs_config_hash", "config_hash"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(64))
    date_from: Mapped[datetime] = mapped_column(UtcDateTime())
    date_to: Mapped[datetime] = mapped_column(UtcDateTime())
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    candidate_count: Mapped[int] = mapped_column(Integer)
    eligible_count: Mapped[int] = mapped_column(Integer)
    built_count: Mapped[int] = mapped_column(Integer)
    excluded_count: Mapped[int] = mapped_column(Integer)
    failed_count: Mapped[int] = mapped_column(Integer)
    config_hash: Mapped[str] = mapped_column(String(64))
    git_sha: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    @classmethod
    def from_entity(cls, item: MlFeatureDatasetRun) -> MlFeatureDatasetRunRecord:
        return cls(
            id=item.id,
            dataset_version=item.dataset_version,
            feature_version=item.feature_version,
            date_from=item.date_from,
            date_to=item.date_to,
            started_at=item.started_at,
            finished_at=item.finished_at,
            candidate_count=item.candidate_count,
            eligible_count=item.eligible_count,
            built_count=item.built_count,
            excluded_count=item.excluded_count,
            failed_count=item.failed_count,
            config_hash=item.config_hash,
            git_sha=item.git_sha,
            status=item.status.value,
            error=item.error,
        )

    def update_from_entity(self, item: MlFeatureDatasetRun) -> None:
        self.finished_at = item.finished_at
        self.candidate_count = item.candidate_count
        self.eligible_count = item.eligible_count
        self.built_count = item.built_count
        self.excluded_count = item.excluded_count
        self.failed_count = item.failed_count
        self.status = item.status.value
        self.error = item.error

    def to_entity(self) -> MlFeatureDatasetRun:
        return MlFeatureDatasetRun(
            id=self.id,
            dataset_version=self.dataset_version,
            feature_version=self.feature_version,
            date_from=self.date_from,
            date_to=self.date_to,
            started_at=self.started_at,
            finished_at=self.finished_at,
            candidate_count=self.candidate_count,
            eligible_count=self.eligible_count,
            built_count=self.built_count,
            excluded_count=self.excluded_count,
            failed_count=self.failed_count,
            config_hash=self.config_hash,
            git_sha=self.git_sha,
            status=FeatureDatasetRunStatus(self.status),
            error=self.error,
        )
