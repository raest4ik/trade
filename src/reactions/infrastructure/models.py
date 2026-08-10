from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.reactions.domain.entities import (
    NewsMarketReaction,
    ReactionBenchmarkAdjustment,
    ReactionPoint,
)
from src.reactions.domain.enums import (
    BenchmarkAdjustmentStatus,
    ReactionPointStatus,
    ReactionStatus,
)
from src.shared.database.base import Base
from src.shared.database.types import UtcDateTime


class NewsMarketReactionRecord(Base):
    __tablename__ = "news_market_reactions"
    __table_args__ = (
        UniqueConstraint(
            "news_id",
            "instrument_id",
            "reaction_version",
            name="uq_news_market_reactions_news_instrument_version",
        ),
        Index("ix_news_market_reactions_news_version", "news_id", "reaction_version"),
        Index("ix_news_market_reactions_instrument_created", "instrument_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    news_id: Mapped[UUID] = mapped_column(ForeignKey("news_items.id", ondelete="CASCADE"))
    instrument_id: Mapped[UUID] = mapped_column(ForeignKey("instruments.id", ondelete="RESTRICT"))
    reaction_version: Mapped[str] = mapped_column(String(64))
    published_at: Mapped[datetime] = mapped_column(UtcDateTime())
    received_at: Mapped[datetime] = mapped_column(UtcDateTime())
    effective_event_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    baseline_observed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    baseline_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    publication_to_receipt_ms: Mapped[int] = mapped_column(BigInteger)
    publication_to_effective_event_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    is_ambiguous_instrument: Mapped[bool] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())

    points: Mapped[list[ReactionPointRecord]] = relationship(
        back_populates="reaction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @classmethod
    def from_entity(cls, item: NewsMarketReaction) -> NewsMarketReactionRecord:
        record = cls(
            id=item.id,
            news_id=item.news_id,
            instrument_id=item.instrument_id,
            reaction_version=item.reaction_version,
            published_at=item.published_at,
            received_at=item.received_at,
            effective_event_at=item.effective_event_at,
            baseline_observed_at=item.baseline_observed_at,
            baseline_price=item.baseline_price,
            publication_to_receipt_ms=item.publication_to_receipt_ms,
            publication_to_effective_event_ms=item.publication_to_effective_event_ms,
            status=item.status.value,
            is_ambiguous_instrument=item.is_ambiguous_instrument,
            created_at=item.created_at,
        )
        record.points = [ReactionPointRecord.from_entity(point) for point in item.points]
        return record

    def to_entity(self) -> NewsMarketReaction:
        return NewsMarketReaction(
            id=self.id,
            news_id=self.news_id,
            instrument_id=self.instrument_id,
            reaction_version=self.reaction_version,
            published_at=self.published_at,
            received_at=self.received_at,
            effective_event_at=self.effective_event_at,
            baseline_observed_at=self.baseline_observed_at,
            baseline_price=self.baseline_price,
            publication_to_receipt_ms=self.publication_to_receipt_ms,
            publication_to_effective_event_ms=self.publication_to_effective_event_ms,
            status=ReactionStatus(self.status),
            is_ambiguous_instrument=self.is_ambiguous_instrument,
            created_at=self.created_at,
            points=[
                point.to_entity()
                for point in sorted(self.points, key=lambda item: item.horizon_minutes)
            ],
        )


class ReactionPointRecord(Base):
    __tablename__ = "reaction_points"
    __table_args__ = (
        UniqueConstraint(
            "reaction_id", "horizon_minutes", name="uq_reaction_points_reaction_horizon"
        ),
        Index("ix_reaction_points_reaction_horizon", "reaction_id", "horizon_minutes"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    reaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_market_reactions.id", ondelete="CASCADE")
    )
    horizon_minutes: Mapped[int] = mapped_column(Integer)
    target_at: Mapped[datetime] = mapped_column(UtcDateTime())
    observed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    simple_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    log_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    status: Mapped[str] = mapped_column(String(32))

    reaction: Mapped[NewsMarketReactionRecord] = relationship(back_populates="points")
    benchmark_adjustment: Mapped[ReactionBenchmarkAdjustmentRecord | None] = relationship(
        back_populates="reaction_point",
        cascade="all, delete-orphan",
        lazy="selectin",
        uselist=False,
    )

    @classmethod
    def from_entity(cls, item: ReactionPoint) -> ReactionPointRecord:
        record = cls(
            id=item.id,
            reaction_id=item.reaction_id,
            horizon_minutes=item.horizon_minutes,
            target_at=item.target_at,
            observed_at=item.observed_at,
            price=item.price,
            simple_return=item.simple_return,
            log_return=item.log_return,
            status=item.status.value,
        )
        if item.benchmark_adjustment is not None:
            record.benchmark_adjustment = ReactionBenchmarkAdjustmentRecord.from_entity(
                item.benchmark_adjustment
            )
        return record

    def to_entity(self) -> ReactionPoint:
        return ReactionPoint(
            id=self.id,
            reaction_id=self.reaction_id,
            horizon_minutes=self.horizon_minutes,
            target_at=self.target_at,
            observed_at=self.observed_at,
            price=self.price,
            simple_return=self.simple_return,
            log_return=self.log_return,
            status=ReactionPointStatus(self.status),
            benchmark_adjustment=None
            if self.benchmark_adjustment is None
            else self.benchmark_adjustment.to_entity(),
        )


class ReactionBenchmarkAdjustmentRecord(Base):
    __tablename__ = "reaction_benchmark_adjustments"
    __table_args__ = (
        UniqueConstraint(
            "reaction_point_id",
            "benchmark_id",
            name="uq_reaction_benchmark_adjustments_point_benchmark",
        ),
        Index(
            "ix_reaction_benchmark_adjustments_benchmark_status",
            "benchmark_id",
            "status",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    reaction_point_id: Mapped[UUID] = mapped_column(
        ForeignKey("reaction_points.id", ondelete="CASCADE")
    )
    benchmark_id: Mapped[UUID] = mapped_column(
        ForeignKey("market_benchmarks.id", ondelete="RESTRICT")
    )
    benchmark_code: Mapped[str] = mapped_column(String(32))
    baseline_value: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    target_value: Mapped[Decimal | None] = mapped_column(Numeric(24, 10), nullable=True)
    baseline_observed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    target_observed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    simple_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    log_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    abnormal_simple_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    abnormal_log_return: Mapped[Decimal | None] = mapped_column(Numeric(28, 18), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    missing_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    reaction_point: Mapped[ReactionPointRecord] = relationship(
        back_populates="benchmark_adjustment"
    )

    @classmethod
    def from_entity(cls, item: ReactionBenchmarkAdjustment) -> ReactionBenchmarkAdjustmentRecord:
        return cls(
            id=item.id,
            reaction_point_id=item.reaction_point_id,
            benchmark_id=item.benchmark_id,
            benchmark_code=item.benchmark_code,
            baseline_value=item.baseline_value,
            target_value=item.target_value,
            baseline_observed_at=item.baseline_observed_at,
            target_observed_at=item.target_observed_at,
            simple_return=item.simple_return,
            log_return=item.log_return,
            abnormal_simple_return=item.abnormal_simple_return,
            abnormal_log_return=item.abnormal_log_return,
            status=item.status.value,
            missing_reason=item.missing_reason,
        )

    def to_entity(self) -> ReactionBenchmarkAdjustment:
        return ReactionBenchmarkAdjustment(
            id=self.id,
            reaction_point_id=self.reaction_point_id,
            benchmark_id=self.benchmark_id,
            benchmark_code=self.benchmark_code,
            baseline_value=self.baseline_value,
            target_value=self.target_value,
            baseline_observed_at=self.baseline_observed_at,
            target_observed_at=self.target_observed_at,
            simple_return=self.simple_return,
            log_return=self.log_return,
            abnormal_simple_return=self.abnormal_simple_return,
            abnormal_log_return=self.abnormal_log_return,
            status=BenchmarkAdjustmentStatus(self.status),
            missing_reason=self.missing_reason,
        )
