from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.news.domain.time import ensure_aware_utc, utc_now
from src.reactions.domain.enums import ReactionPointStatus, ReactionStatus

REACTION_VERSION = "reaction-v1-minute-candles"
DEFAULT_REACTION_HORIZONS_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True, slots=True)
class ReactionPoint:
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
    def create(
        cls,
        *,
        reaction_id: UUID,
        horizon_minutes: int,
        target_at: datetime,
        observed_at: datetime | None,
        price: Decimal | None,
        simple_return: Decimal | None,
        log_return: Decimal | None,
        status: ReactionPointStatus,
    ) -> ReactionPoint:
        return cls(
            id=uuid4(),
            reaction_id=reaction_id,
            horizon_minutes=horizon_minutes,
            target_at=ensure_aware_utc(target_at, "target_at"),
            observed_at=None
            if observed_at is None
            else ensure_aware_utc(observed_at, "observed_at"),
            price=price,
            simple_return=simple_return,
            log_return=log_return,
            status=status,
        )


@dataclass(frozen=True, slots=True)
class NewsMarketReaction:
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
    points: list[ReactionPoint]

    @classmethod
    def create(
        cls,
        *,
        news_id: UUID,
        instrument_id: UUID,
        published_at: datetime,
        received_at: datetime,
        effective_event_at: datetime | None,
        baseline_observed_at: datetime | None,
        baseline_price: Decimal | None,
        status: ReactionStatus,
        is_ambiguous_instrument: bool,
        points: list[ReactionPoint],
        reaction_version: str = REACTION_VERSION,
    ) -> NewsMarketReaction:
        published_utc = ensure_aware_utc(published_at, "published_at")
        received_utc = ensure_aware_utc(received_at, "received_at")
        effective_utc = (
            None
            if effective_event_at is None
            else ensure_aware_utc(effective_event_at, "effective_event_at")
        )
        baseline_utc = (
            None
            if baseline_observed_at is None
            else ensure_aware_utc(baseline_observed_at, "baseline_observed_at")
        )
        reaction_id = uuid4()
        return cls(
            id=reaction_id,
            news_id=news_id,
            instrument_id=instrument_id,
            reaction_version=reaction_version,
            published_at=published_utc,
            received_at=received_utc,
            effective_event_at=effective_utc,
            baseline_observed_at=baseline_utc,
            baseline_price=baseline_price,
            publication_to_receipt_ms=_millis(received_utc, published_utc),
            publication_to_effective_event_ms=None
            if effective_utc is None
            else _millis(effective_utc, published_utc),
            status=status,
            is_ambiguous_instrument=is_ambiguous_instrument,
            created_at=utc_now(),
            points=[
                ReactionPoint(
                    id=point.id,
                    reaction_id=reaction_id,
                    horizon_minutes=point.horizon_minutes,
                    target_at=point.target_at,
                    observed_at=point.observed_at,
                    price=point.price,
                    simple_return=point.simple_return,
                    log_return=point.log_return,
                    status=point.status,
                )
                for point in points
            ],
        )


def _millis(later: datetime, earlier: datetime) -> int:
    return int((later - earlier).total_seconds() * 1000)
