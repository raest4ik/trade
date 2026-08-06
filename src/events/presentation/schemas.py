from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact, NewsEventAnalysis
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


class DetectedEventResponse(BaseModel):
    id: UUID
    analysis_id: UUID
    event_type: EventType
    confidence: Decimal
    matched_rule: str
    evidence_text: str
    start_position: int
    end_position: int

    @classmethod
    def from_entity(cls, event: DetectedEvent) -> DetectedEventResponse:
        return cls(
            id=event.id,
            analysis_id=event.analysis_id,
            event_type=event.event_type,
            confidence=event.confidence,
            matched_rule=event.matched_rule,
            evidence_text=event.evidence_text,
            start_position=event.start_position,
            end_position=event.end_position,
        )


class ExtractedFinancialFactResponse(BaseModel):
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
    evidence_text: str
    start_position: int
    end_position: int
    extractor_version: str
    matched_rule: str

    @classmethod
    def from_entity(cls, fact: ExtractedFinancialFact) -> ExtractedFinancialFactResponse:
        return cls(
            id=fact.id,
            analysis_id=fact.analysis_id,
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
            evidence_text=fact.evidence_text,
            start_position=fact.start_position,
            end_position=fact.end_position,
            extractor_version=fact.extractor_version,
            matched_rule=fact.matched_rule,
        )


class NewsEventAnalysisResponse(BaseModel):
    id: UUID
    news_id: UUID
    analysis_version: str
    status: EventAnalysisStatus
    primary_event_type: EventType
    created_at: datetime
    analyzed_at: datetime
    events: list[DetectedEventResponse]
    financial_facts: list[ExtractedFinancialFactResponse]
    warnings: list[str]
    debug: dict[str, object] | None = None

    @classmethod
    def from_entity(
        cls,
        analysis: NewsEventAnalysis,
        *,
        include_debug: bool = False,
    ) -> NewsEventAnalysisResponse:
        warnings = _analysis_warnings(analysis)
        return cls(
            id=analysis.id,
            news_id=analysis.news_id,
            analysis_version=analysis.analysis_version,
            status=analysis.status,
            primary_event_type=analysis.primary_event_type,
            created_at=analysis.created_at,
            analyzed_at=analysis.analyzed_at,
            events=[DetectedEventResponse.from_entity(event) for event in analysis.events],
            financial_facts=[
                ExtractedFinancialFactResponse.from_entity(fact)
                for fact in analysis.financial_facts
            ],
            warnings=warnings,
            debug=_debug_payload(analysis) if include_debug else None,
        )


def _analysis_warnings(analysis: NewsEventAnalysis) -> list[str]:
    warnings: list[str] = []
    if not analysis.events:
        warnings.append("no deterministic corporate event rule matched")
    if not analysis.financial_facts:
        warnings.append("no financial facts were extracted")
    if any(fact.period_type == PeriodType.UNKNOWN for fact in analysis.financial_facts):
        warnings.append("some financial facts have no detected reporting period")
    if any(fact.metric == FinancialMetric.OTHER for fact in analysis.financial_facts):
        warnings.append("some numeric facts have no nearby supported metric")
    if any(fact.confidence < Decimal("0.75") for fact in analysis.financial_facts):
        warnings.append("some extracted facts have low deterministic confidence")
    return warnings


def _debug_payload(analysis: NewsEventAnalysis) -> dict[str, object]:
    return {
        "rules_version": analysis.analysis_version,
        "event_rule_ids": [event.matched_rule for event in analysis.events],
        "fact_rule_ids": [fact.matched_rule for fact in analysis.financial_facts],
        "fact_count": len(analysis.financial_facts),
        "event_count": len(analysis.events),
    }
