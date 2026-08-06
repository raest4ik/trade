from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.evaluation.domain.entities import (
    EvaluationDataset,
    EvaluationRun,
    GoldEvent,
    GoldFinancialFact,
)
from src.evaluation.domain.enums import DatasetSplit, EvaluationRunStatus
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class EvaluationDatasetRecord(Base):
    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("source_file_hash", name="uq_evaluation_datasets_source_file_hash"),
        Index("ix_evaluation_datasets_name", "name"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    schema_version: Mapped[str] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_file_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    imported_at: Mapped[datetime] = mapped_column(UtcDateTime())
    example_count: Mapped[int] = mapped_column(Integer)
    reviewed_count: Mapped[int] = mapped_column(Integer)
    train_count: Mapped[int] = mapped_column(Integer)
    validation_count: Mapped[int] = mapped_column(Integer)
    test_count: Mapped[int] = mapped_column(Integer)
    split_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    train_until: Mapped[date | None] = mapped_column(nullable=True)
    validation_until: Mapped[date | None] = mapped_column(nullable=True)

    examples: Mapped[list[EvaluationExampleRecord]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    runs: Mapped[list[EvaluationRunRecord]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @classmethod
    def from_entity(cls, dataset: EvaluationDataset) -> EvaluationDatasetRecord:
        return cls(
            id=dataset.id,
            name=dataset.name,
            schema_version=dataset.schema_version,
            description=dataset.description,
            source_file_hash=dataset.source_file_hash,
            created_at=dataset.created_at,
            imported_at=dataset.imported_at,
            example_count=dataset.example_count,
            reviewed_count=dataset.reviewed_count,
            train_count=dataset.train_count,
            validation_count=dataset.validation_count,
            test_count=dataset.test_count,
            split_strategy=dataset.split_strategy,
            train_until=dataset.train_until,
            validation_until=dataset.validation_until,
        )

    def to_entity(self) -> EvaluationDataset:
        return EvaluationDataset(
            id=self.id,
            name=self.name,
            schema_version=self.schema_version,
            description=self.description,
            source_file_hash=self.source_file_hash,
            created_at=self.created_at,
            imported_at=self.imported_at,
            example_count=self.example_count,
            reviewed_count=self.reviewed_count,
            train_count=self.train_count,
            validation_count=self.validation_count,
            test_count=self.test_count,
            split_strategy=self.split_strategy,
            train_until=self.train_until,
            validation_until=self.validation_until,
        )


class EvaluationExampleRecord(Base):
    __tablename__ = "evaluation_examples"
    __table_args__ = (
        UniqueConstraint("dataset_id", "news_id", name="uq_evaluation_examples_dataset_news"),
        Index("ix_evaluation_examples_dataset_split", "dataset_id", "split"),
        Index("ix_evaluation_examples_news_id", "news_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE")
    )
    news_id: Mapped[UUID] = mapped_column(ForeignKey("news_items.id", ondelete="RESTRICT"))
    published_at: Mapped[datetime] = mapped_column(UtcDateTime())
    raw_content_hash: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(32))
    review_status: Mapped[str] = mapped_column(String(32))
    annotator: Mapped[str] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_events: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    predicted_financial_facts: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime())

    dataset: Mapped[EvaluationDatasetRecord] = relationship(back_populates="examples")
    gold_events: Mapped[list[GoldEventRecord]] = relationship(
        back_populates="example",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    gold_financial_facts: Mapped[list[GoldFinancialFactRecord]] = relationship(
        back_populates="example",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class GoldEventRecord(Base):
    __tablename__ = "gold_events"
    __table_args__ = (Index("ix_gold_events_example_type", "example_id", "event_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    example_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_examples.id", ondelete="CASCADE")
    )
    event_type: Mapped[str] = mapped_column(String(64))
    evidence_text: Mapped[str] = mapped_column(String(1000))
    start_position: Mapped[int] = mapped_column(Integer)
    end_position: Mapped[int] = mapped_column(Integer)
    is_primary: Mapped[bool] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    example: Mapped[EvaluationExampleRecord] = relationship(back_populates="gold_events")

    @classmethod
    def from_entity(cls, example_id: UUID, event: GoldEvent) -> GoldEventRecord:
        return cls(
            example_id=example_id,
            event_type=event.event_type.value,
            evidence_text=event.evidence_text,
            start_position=event.start_position,
            end_position=event.end_position,
            is_primary=event.is_primary,
            notes=event.notes,
        )

    def to_entity(self) -> GoldEvent:
        return GoldEvent(
            event_type=EventType(self.event_type),
            evidence_text=self.evidence_text,
            start_position=self.start_position,
            end_position=self.end_position,
            is_primary=self.is_primary,
            notes=self.notes,
        )


class GoldFinancialFactRecord(Base):
    __tablename__ = "gold_financial_facts"
    __table_args__ = (Index("ix_gold_financial_facts_example_metric", "example_id", "metric"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    example_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_examples.id", ondelete="CASCADE")
    )
    metric: Mapped[str] = mapped_column(String(64))
    raw_value: Mapped[Decimal] = mapped_column(Numeric(28, 10))
    normalized_value: Mapped[Decimal] = mapped_column(Numeric(38, 10))
    unit: Mapped[str] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(16))
    scale: Mapped[str] = mapped_column(String(32))
    period_type: Mapped[str] = mapped_column(String(32))
    period_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_period: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fact_role: Mapped[str] = mapped_column(String(32))
    comparison_type: Mapped[str] = mapped_column(String(32))
    change_direction: Mapped[str] = mapped_column(String(32))
    change_value: Mapped[Decimal | None] = mapped_column(Numeric(28, 10), nullable=True)
    change_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evidence_text: Mapped[str] = mapped_column(String(1000))
    start_position: Mapped[int] = mapped_column(Integer)
    end_position: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    example: Mapped[EvaluationExampleRecord] = relationship(back_populates="gold_financial_facts")

    @classmethod
    def from_entity(cls, example_id: UUID, fact: GoldFinancialFact) -> GoldFinancialFactRecord:
        return cls(
            example_id=example_id,
            metric=fact.metric.value,
            raw_value=fact.raw_value,
            normalized_value=fact.normalized_value,
            unit=fact.unit.value,
            currency=fact.currency.value,
            scale=fact.scale.value,
            period_type=fact.period_type.value,
            period_year=fact.period_year,
            period_quarter=fact.period_quarter,
            period_month=fact.period_month,
            raw_period=fact.raw_period,
            fact_role=fact.fact_role.value,
            comparison_type=fact.comparison_type.value,
            change_direction=fact.change_direction.value,
            change_value=fact.change_value,
            change_unit=None if fact.change_unit is None else fact.change_unit.value,
            evidence_text=fact.evidence_text,
            start_position=fact.start_position,
            end_position=fact.end_position,
            notes=fact.notes,
        )

    def to_entity(self) -> GoldFinancialFact:
        return GoldFinancialFact(
            metric=FinancialMetric(self.metric),
            raw_value=self.raw_value,
            normalized_value=self.normalized_value,
            unit=FactUnit(self.unit),
            currency=Currency(self.currency),
            scale=ValueScale(self.scale),
            period_type=PeriodType(self.period_type),
            period_year=self.period_year,
            period_quarter=self.period_quarter,
            period_month=self.period_month,
            raw_period=self.raw_period,
            fact_role=FactRole(self.fact_role),
            comparison_type=ComparisonType(self.comparison_type),
            change_direction=ChangeDirection(self.change_direction),
            change_value=self.change_value,
            change_unit=None if self.change_unit is None else FactUnit(self.change_unit),
            evidence_text=self.evidence_text,
            start_position=self.start_position,
            end_position=self.end_position,
            notes=self.notes,
        )


class EvaluationRunRecord(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_dataset_started", "dataset_id", "started_at"),
        Index("ix_evaluation_runs_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE")
    )
    split: Mapped[str] = mapped_column(String(32))
    analysis_version: Mapped[str] = mapped_column(String(64))
    extractor_version: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    example_count: Mapped[int] = mapped_column(Integer)
    metrics_json: Mapped[dict[str, object]] = mapped_column(JSON)
    error_count: Mapped[int] = mapped_column(Integer)
    git_commit_sha: Mapped[str] = mapped_column(String(64))
    config_json: Mapped[dict[str, object]] = mapped_column(JSON)

    dataset: Mapped[EvaluationDatasetRecord] = relationship(back_populates="runs")

    @classmethod
    def from_entity(cls, run: EvaluationRun) -> EvaluationRunRecord:
        return cls(
            id=run.id,
            dataset_id=run.dataset_id,
            split=run.split.value,
            analysis_version=run.analysis_version,
            extractor_version=run.extractor_version,
            started_at=run.started_at,
            finished_at=run.finished_at,
            status=run.status.value,
            example_count=run.example_count,
            metrics_json=run.metrics_json,
            error_count=run.error_count,
            git_commit_sha=run.git_commit_sha,
            config_json=run.config_json,
        )

    def to_entity(self) -> EvaluationRun:
        return EvaluationRun(
            id=self.id,
            dataset_id=self.dataset_id,
            split=DatasetSplit(self.split),
            analysis_version=self.analysis_version,
            extractor_version=self.extractor_version,
            started_at=self.started_at,
            finished_at=self.finished_at,
            status=EvaluationRunStatus(self.status),
            example_count=self.example_count,
            metrics_json=self.metrics_json,
            error_count=self.error_count,
            git_commit_sha=self.git_commit_sha,
            config_json=self.config_json,
        )
