from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from src.reactions.domain.entities import NewsMarketReaction, ReactionPoint
from src.reactions.domain.enums import ReactionPointStatus, ReactionStatus


class ReactionPointResponse(BaseModel):
    id: UUID
    reaction_id: UUID
    horizon_minutes: int
    target_at: datetime
    observed_at: datetime | None
    price: Decimal | None
    simple_return: Decimal | None
    log_return: Decimal | None
    status: ReactionPointStatus

    @classmethod
    def from_entity(cls, point: ReactionPoint) -> ReactionPointResponse:
        return cls(
            id=point.id,
            reaction_id=point.reaction_id,
            horizon_minutes=point.horizon_minutes,
            target_at=point.target_at,
            observed_at=point.observed_at,
            price=point.price,
            simple_return=point.simple_return,
            log_return=point.log_return,
            status=point.status,
        )


class NewsMarketReactionResponse(BaseModel):
    id: UUID
    news_id: UUID
    instrument_id: UUID
    reaction_version: str
    published_at: datetime
    received_at: datetime
    effective_event_at: datetime | None
    baseline_observed_at: datetime | None
    baseline_price: Decimal | None
    publication_to_receipt_ms: int
    publication_to_effective_event_ms: int | None
    status: ReactionStatus
    is_ambiguous_instrument: bool
    created_at: datetime
    points: list[ReactionPointResponse]

    @classmethod
    def from_entity(cls, item: NewsMarketReaction) -> NewsMarketReactionResponse:
        return cls(
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
            status=item.status,
            is_ambiguous_instrument=item.is_ambiguous_instrument,
            created_at=item.created_at,
            points=[ReactionPointResponse.from_entity(point) for point in item.points],
        )


class NewsReactionsResponse(BaseModel):
    news_id: UUID
    reaction_version: str
    instruments: list[NewsMarketReactionResponse]
    baseline: list[dict[str, object]]
    horizons: list[dict[str, object]]
    data_quality: dict[str, object]
    ambiguity: dict[str, object]

    @classmethod
    def from_reactions(
        cls,
        *,
        news_id: UUID,
        reaction_version: str,
        reactions: list[NewsMarketReaction],
    ) -> NewsReactionsResponse:
        return cls(
            news_id=news_id,
            reaction_version=reaction_version,
            instruments=[NewsMarketReactionResponse.from_entity(item) for item in reactions],
            baseline=[
                {
                    "instrument_id": item.instrument_id,
                    "observed_at": item.baseline_observed_at,
                    "price": item.baseline_price,
                    "status": item.status.value,
                }
                for item in reactions
            ],
            horizons=[
                {
                    "instrument_id": item.instrument_id,
                    "horizon_minutes": point.horizon_minutes,
                    "status": point.status.value,
                    "target_at": point.target_at,
                    "observed_at": point.observed_at,
                }
                for item in reactions
                for point in item.points
            ],
            data_quality={
                "complete": sum(1 for item in reactions if item.status == ReactionStatus.COMPLETE),
                "partial": sum(1 for item in reactions if item.status == ReactionStatus.PARTIAL),
                "insufficient_data": sum(
                    1 for item in reactions if item.status == ReactionStatus.INSUFFICIENT_DATA
                ),
                "outside_session": sum(
                    1 for item in reactions if item.status == ReactionStatus.OUTSIDE_SESSION
                ),
            },
            ambiguity={
                "has_ambiguous_instruments": any(
                    item.is_ambiguous_instrument for item in reactions
                ),
                "ambiguous_count": sum(1 for item in reactions if item.is_ambiguous_instrument),
            },
        )
