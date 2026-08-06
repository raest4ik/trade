from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class NewsEventAnalysisRecord(Base):
    __tablename__ = "news_event_analyses"
    __table_args__ = (
        UniqueConstraint("news_id", "analysis_version", name="uq_news_event_analyses_news_version"),
        Index("ix_news_event_analyses_news_version", "news_id", "analysis_version"),
        Index("ix_news_event_analyses_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    news_id: Mapped[UUID] = mapped_column(ForeignKey("news_items.id", ondelete="CASCADE"))
    analysis_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    primary_event_type: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    analyzed_at: Mapped[datetime] = mapped_column(UtcDateTime())

    events: Mapped[list[DetectedEventRecord]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    financial_facts: Mapped[list[ExtractedFinancialFactRecord]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @classmethod
    def from_entity(cls, analysis: NewsEventAnalysis) -> NewsEventAnalysisRecord:
        record = cls(
            id=analysis.id,
            news_id=analysis.news_id,
            analysis_version=analysis.analysis_version,
            status=analysis.status.value,
            primary_event_type=analysis.primary_event_type.value,
            created_at=analysis.created_at,
            analyzed_at=analysis.analyzed_at,
        )
        record.events = [DetectedEventRecord.from_entity(event) for event in analysis.events]
        record.financial_facts = [
            ExtractedFinancialFactRecord.from_entity(fact) for fact in analysis.financial_facts
        ]
        return record

    def to_entity(self) -> NewsEventAnalysis:
        return NewsEventAnalysis(
            id=self.id,
            news_id=self.news_id,
            analysis_version=self.analysis_version,
            status=EventAnalysisStatus(self.status),
            primary_event_type=EventType(self.primary_event_type),
            created_at=self.created_at,
            analyzed_at=self.analyzed_at,
            events=[
                event.to_entity()
                for event in sorted(self.events, key=lambda item: item.start_position)
            ],
            financial_facts=[
                fact.to_entity()
                for fact in sorted(self.financial_facts, key=lambda item: item.start_position)
            ],
        )


class DetectedEventRecord(Base):
    __tablename__ = "detected_events"
    __table_args__ = (
        Index("ix_detected_events_analysis_type", "analysis_id", "event_type"),
        Index("ix_detected_events_type", "event_type"),
        UniqueConstraint(
            "analysis_id",
            "event_type",
            "rule_id",
            "start_position",
            "end_position",
            name="uq_detected_events_exact_span",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_event_analyses.id", ondelete="CASCADE")
    )
    event_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    rule_id: Mapped[str] = mapped_column(String(128))
    matched_rule: Mapped[str] = mapped_column(String(128))
    evidence_text: Mapped[str] = mapped_column(String(1000))
    start_position: Mapped[int] = mapped_column(Integer)
    end_position: Mapped[int] = mapped_column(Integer)

    analysis: Mapped[NewsEventAnalysisRecord] = relationship(back_populates="events")

    @classmethod
    def from_entity(cls, event: DetectedEvent) -> DetectedEventRecord:
        return cls(
            id=event.id,
            analysis_id=event.analysis_id,
            event_type=event.event_type.value,
            confidence=event.confidence,
            rule_id=event.rule_id,
            matched_rule=event.matched_rule,
            evidence_text=event.evidence_text,
            start_position=event.start_position,
            end_position=event.end_position,
        )

    def to_entity(self) -> DetectedEvent:
        return DetectedEvent(
            id=self.id,
            analysis_id=self.analysis_id,
            event_type=EventType(self.event_type),
            confidence=self.confidence,
            rule_id=self.rule_id,
            matched_rule=self.matched_rule,
            evidence_text=self.evidence_text,
            start_position=self.start_position,
            end_position=self.end_position,
        )


class ExtractedFinancialFactRecord(Base):
    __tablename__ = "extracted_financial_facts"
    __table_args__ = (
        Index("ix_extracted_financial_facts_analysis_metric", "analysis_id", "metric"),
        Index("ix_extracted_financial_facts_metric", "metric"),
        UniqueConstraint(
            "analysis_id",
            "metric",
            "rule_id",
            "start_position",
            "end_position",
            name="uq_extracted_financial_facts_exact_span",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_event_analyses.id", ondelete="CASCADE")
    )
    metric: Mapped[str] = mapped_column(String(64))
    raw_value: Mapped[Decimal] = mapped_column(Numeric(28, 10))
    normalized_value: Mapped[Decimal] = mapped_column(Numeric(38, 10))
    unit: Mapped[str] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(16))
    scale: Mapped[str] = mapped_column(String(32))
    period_type: Mapped[str] = mapped_column(String(32))
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarter: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    date_from: Mapped[date | None] = mapped_column(nullable=True)
    date_to: Mapped[date | None] = mapped_column(nullable=True)
    raw_period: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comparison_type: Mapped[str] = mapped_column(String(32))
    fact_role: Mapped[str] = mapped_column(String(32))
    change_direction: Mapped[str] = mapped_column(String(32))
    change_value: Mapped[Decimal | None] = mapped_column(Numeric(28, 10), nullable=True)
    change_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    rule_id: Mapped[str] = mapped_column(String(128))
    evidence_text: Mapped[str] = mapped_column(String(1000))
    start_position: Mapped[int] = mapped_column(Integer)
    end_position: Mapped[int] = mapped_column(Integer)
    extractor_version: Mapped[str] = mapped_column(String(64))
    matched_rule: Mapped[str] = mapped_column(String(128))

    analysis: Mapped[NewsEventAnalysisRecord] = relationship(back_populates="financial_facts")

    @classmethod
    def from_entity(cls, fact: ExtractedFinancialFact) -> ExtractedFinancialFactRecord:
        return cls(
            id=fact.id,
            analysis_id=fact.analysis_id,
            metric=fact.metric.value,
            raw_value=fact.raw_value,
            normalized_value=fact.normalized_value,
            unit=fact.unit.value,
            currency=fact.currency.value,
            scale=fact.scale.value,
            period_type=fact.period_type.value,
            year=fact.year,
            quarter=fact.quarter,
            month=fact.month,
            date_from=fact.date_from,
            date_to=fact.date_to,
            raw_period=fact.raw_period,
            comparison_type=fact.comparison_type.value,
            fact_role=fact.fact_role.value,
            change_direction=fact.change_direction.value,
            change_value=fact.change_value,
            change_unit=None if fact.change_unit is None else fact.change_unit.value,
            confidence=fact.confidence,
            rule_id=fact.rule_id,
            evidence_text=fact.evidence_text,
            start_position=fact.start_position,
            end_position=fact.end_position,
            extractor_version=fact.extractor_version,
            matched_rule=fact.matched_rule,
        )

    def to_entity(self) -> ExtractedFinancialFact:
        return ExtractedFinancialFact(
            id=self.id,
            analysis_id=self.analysis_id,
            metric=FinancialMetric(self.metric),
            raw_value=self.raw_value,
            normalized_value=self.normalized_value,
            unit=FactUnit(self.unit),
            currency=Currency(self.currency),
            scale=ValueScale(self.scale),
            period_type=PeriodType(self.period_type),
            year=self.year,
            quarter=self.quarter,
            month=self.month,
            date_from=self.date_from,
            date_to=self.date_to,
            raw_period=self.raw_period,
            comparison_type=ComparisonType(self.comparison_type),
            fact_role=FactRole(self.fact_role),
            change_direction=ChangeDirection(self.change_direction),
            change_value=self.change_value,
            change_unit=None if self.change_unit is None else FactUnit(self.change_unit),
            confidence=self.confidence,
            rule_id=self.rule_id,
            evidence_text=self.evidence_text,
            start_position=self.start_position,
            end_position=self.end_position,
            extractor_version=self.extractor_version,
            matched_rule=self.matched_rule,
        )
