from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventAnalysisStatus,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)
from src.news.domain.time import utc_now

EVENT_ANALYSIS_VERSION = "event-rules-v1"
FINANCIAL_FACTS_VERSION = "financial-facts-v1"


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    id: UUID
    analysis_id: UUID
    event_type: EventType
    confidence: Decimal
    rule_id: str
    matched_rule: str
    evidence_text: str
    start_position: int
    end_position: int


@dataclass(frozen=True, slots=True)
class ExtractedFinancialFact:
    id: UUID
    analysis_id: UUID
    metric: FinancialMetric
    raw_value: Decimal
    normalized_value: Decimal
    unit: FactUnit
    currency: Currency
    scale: ValueScale
    period_type: PeriodType
    year: int | None
    quarter: int | None
    month: int | None
    date_from: date | None
    date_to: date | None
    raw_period: str | None
    comparison_type: ComparisonType
    fact_role: FactRole
    change_direction: ChangeDirection
    change_value: Decimal | None
    change_unit: FactUnit | None
    confidence: Decimal
    rule_id: str
    evidence_text: str
    start_position: int
    end_position: int
    extractor_version: str
    matched_rule: str


@dataclass(frozen=True, slots=True)
class NewsEventAnalysis:
    id: UUID
    news_id: UUID
    analysis_version: str
    status: EventAnalysisStatus
    primary_event_type: EventType
    created_at: datetime
    analyzed_at: datetime
    events: list[DetectedEvent]
    financial_facts: list[ExtractedFinancialFact]

    @classmethod
    def create(
        cls,
        *,
        news_id: UUID,
        status: EventAnalysisStatus,
        primary_event_type: EventType,
        events: list[DetectedEvent],
        financial_facts: list[ExtractedFinancialFact],
        analysis_version: str = EVENT_ANALYSIS_VERSION,
    ) -> NewsEventAnalysis:
        analysis_id = uuid4()
        now = utc_now()
        return cls(
            id=analysis_id,
            news_id=news_id,
            analysis_version=analysis_version,
            status=status,
            primary_event_type=primary_event_type,
            created_at=now,
            analyzed_at=now,
            events=[
                DetectedEvent(
                    id=event.id,
                    analysis_id=analysis_id,
                    event_type=event.event_type,
                    confidence=event.confidence,
                    rule_id=event.rule_id,
                    matched_rule=event.matched_rule,
                    evidence_text=event.evidence_text,
                    start_position=event.start_position,
                    end_position=event.end_position,
                )
                for event in events
            ],
            financial_facts=[
                ExtractedFinancialFact(
                    id=fact.id,
                    analysis_id=analysis_id,
                    metric=fact.metric,
                    raw_value=fact.raw_value,
                    normalized_value=fact.normalized_value,
                    unit=fact.unit,
                    currency=fact.currency,
                    scale=fact.scale,
                    period_type=fact.period_type,
                    year=fact.year,
                    quarter=fact.quarter,
                    month=fact.month,
                    date_from=fact.date_from,
                    date_to=fact.date_to,
                    raw_period=fact.raw_period,
                    comparison_type=fact.comparison_type,
                    fact_role=fact.fact_role,
                    change_direction=fact.change_direction,
                    change_value=fact.change_value,
                    change_unit=fact.change_unit,
                    confidence=fact.confidence,
                    rule_id=fact.rule_id,
                    evidence_text=fact.evidence_text,
                    start_position=fact.start_position,
                    end_position=fact.end_position,
                    extractor_version=fact.extractor_version,
                    matched_rule=fact.matched_rule,
                )
                for fact in financial_facts
            ],
        )
