from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.evaluation.domain.enums import DatasetSplit, EvaluationRunStatus, ReviewStatus
from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
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
from src.news.domain.time import utc_now

GOLD_SCHEMA_VERSION = "event-gold-v1"


@dataclass(frozen=True, slots=True)
class GoldEvent:
    event_type: EventType
    evidence_text: str
    start_position: int
    end_position: int
    is_primary: bool
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class GoldFinancialFact:
    metric: FinancialMetric
    raw_value: Decimal
    normalized_value: Decimal
    unit: FactUnit
    currency: Currency
    scale: ValueScale
    period_type: PeriodType
    period_year: int | None
    period_quarter: int | None
    period_month: int | None
    raw_period: str | None
    fact_role: FactRole
    comparison_type: ComparisonType
    change_direction: ChangeDirection
    change_value: Decimal | None
    change_unit: FactUnit | None
    evidence_text: str
    start_position: int
    end_position: int
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationExample:
    schema_version: str
    news_id: UUID
    published_at: datetime
    raw_content_hash: str
    split: DatasetSplit
    review_status: ReviewStatus
    annotator: str
    notes: str | None
    predicted_events: list[dict[str, object]]
    predicted_financial_facts: list[dict[str, object]]
    gold_events: list[GoldEvent]
    gold_financial_facts: list[GoldFinancialFact]
    raw_content: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    id: UUID
    name: str
    schema_version: str
    description: str | None
    source_file_hash: str
    created_at: datetime
    imported_at: datetime
    example_count: int
    reviewed_count: int
    train_count: int
    validation_count: int
    test_count: int
    split_strategy: str | None
    train_until: date | None
    validation_until: date | None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        source_file_hash: str,
        example_count: int,
        reviewed_count: int,
        train_count: int,
        validation_count: int,
        test_count: int,
        description: str | None = None,
    ) -> EvaluationDataset:
        now = utc_now()
        return cls(
            id=uuid4(),
            name=name,
            schema_version=GOLD_SCHEMA_VERSION,
            description=description,
            source_file_hash=source_file_hash,
            created_at=now,
            imported_at=now,
            example_count=example_count,
            reviewed_count=reviewed_count,
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            split_strategy=None,
            train_until=None,
            validation_until=None,
        )


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    id: UUID
    dataset_id: UUID
    split: DatasetSplit
    analysis_version: str
    extractor_version: str
    started_at: datetime
    finished_at: datetime | None
    status: EvaluationRunStatus
    example_count: int
    metrics_json: dict[str, object]
    error_count: int
    git_commit_sha: str
    config_json: dict[str, object]

    @classmethod
    def running(
        cls,
        *,
        dataset_id: UUID,
        split: DatasetSplit,
        git_commit_sha: str,
        config_json: dict[str, object],
        analysis_version: str = EVENT_ANALYSIS_VERSION,
        extractor_version: str = FINANCIAL_FACTS_VERSION,
    ) -> EvaluationRun:
        return cls(
            id=uuid4(),
            dataset_id=dataset_id,
            split=split,
            analysis_version=analysis_version,
            extractor_version=extractor_version,
            started_at=utc_now(),
            finished_at=None,
            status=EvaluationRunStatus.RUNNING,
            example_count=0,
            metrics_json={},
            error_count=0,
            git_commit_sha=git_commit_sha,
            config_json=config_json,
        )
