from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.reactions.domain.entities import NewsMarketReaction, ReactionPoint
from src.reactions.domain.enums import ReactionPointStatus, ReactionStatus
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
    publication_to_receipt_ms: Mapped[int] = mapped_column(Integer)
    publication_to_effective_event_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
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

    @classmethod
    def from_entity(cls, item: ReactionPoint) -> ReactionPointRecord:
        return cls(
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
        )
