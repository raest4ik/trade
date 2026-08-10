from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.news.domain.time import ensure_aware_utc, utc_now
from src.reactions.domain.enums import (
    BenchmarkAdjustmentStatus,
    ReactionPointStatus,
    ReactionStatus,
)

REACTION_VERSION = "reaction-v2-benchmark-adjusted"
DEFAULT_REACTION_HORIZONS_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True, slots=True)
class ReactionBenchmarkAdjustment:
    id: UUID
    reaction_point_id: UUID
    benchmark_id: UUID
    benchmark_code: str
    baseline_value: Decimal | None
    target_value: Decimal | None
    baseline_observed_at: datetime | None
    target_observed_at: datetime | None
    simple_return: Decimal | None
    log_return: Decimal | None
    abnormal_simple_return: Decimal | None
    abnormal_log_return: Decimal | None
    status: BenchmarkAdjustmentStatus
    missing_reason: str | None

    @classmethod
    def create(
        cls,
        *,
        reaction_point_id: UUID,
        benchmark_id: UUID,
        benchmark_code: str,
        baseline_value: Decimal | None,
        target_value: Decimal | None,
        baseline_observed_at: datetime | None,
        target_observed_at: datetime | None,
        simple_return: Decimal | None,
        log_return: Decimal | None,
        abnormal_simple_return: Decimal | None,
        abnormal_log_return: Decimal | None,
        status: BenchmarkAdjustmentStatus,
        missing_reason: str | None = None,
    ) -> ReactionBenchmarkAdjustment:
        return cls(
            id=uuid4(),
            reaction_point_id=reaction_point_id,
            benchmark_id=benchmark_id,
            benchmark_code=benchmark_code.strip().upper(),
            baseline_value=baseline_value,
            target_value=target_value,
            baseline_observed_at=None
            if baseline_observed_at is None
            else ensure_aware_utc(baseline_observed_at, "benchmark_baseline_observed_at"),
            target_observed_at=None
            if target_observed_at is None
            else ensure_aware_utc(target_observed_at, "benchmark_target_observed_at"),
            simple_return=simple_return,
            log_return=log_return,
            abnormal_simple_return=abnormal_simple_return,
            abnormal_log_return=abnormal_log_return,
            status=status,
            missing_reason=missing_reason,
        )


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
    benchmark_adjustment: ReactionBenchmarkAdjustment | None = None

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
        benchmark_adjustment: ReactionBenchmarkAdjustment | None = None,
    ) -> ReactionPoint:
        point_id = uuid4()
        adjustment = benchmark_adjustment
        if adjustment is not None and adjustment.reaction_point_id != point_id:
            adjustment = ReactionBenchmarkAdjustment(
                id=adjustment.id,
                reaction_point_id=point_id,
                benchmark_id=adjustment.benchmark_id,
                benchmark_code=adjustment.benchmark_code,
                baseline_value=adjustment.baseline_value,
                target_value=adjustment.target_value,
                baseline_observed_at=adjustment.baseline_observed_at,
                target_observed_at=adjustment.target_observed_at,
                simple_return=adjustment.simple_return,
                log_return=adjustment.log_return,
                abnormal_simple_return=adjustment.abnormal_simple_return,
                abnormal_log_return=adjustment.abnormal_log_return,
                status=adjustment.status,
                missing_reason=adjustment.missing_reason,
            )
        return cls(
            id=point_id,
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
            benchmark_adjustment=adjustment,
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
                    benchmark_adjustment=None
                    if point.benchmark_adjustment is None
                    else ReactionBenchmarkAdjustment(
                        id=point.benchmark_adjustment.id,
                        reaction_point_id=point.id,
                        benchmark_id=point.benchmark_adjustment.benchmark_id,
                        benchmark_code=point.benchmark_adjustment.benchmark_code,
                        baseline_value=point.benchmark_adjustment.baseline_value,
                        target_value=point.benchmark_adjustment.target_value,
                        baseline_observed_at=point.benchmark_adjustment.baseline_observed_at,
                        target_observed_at=point.benchmark_adjustment.target_observed_at,
                        simple_return=point.benchmark_adjustment.simple_return,
                        log_return=point.benchmark_adjustment.log_return,
                        abnormal_simple_return=point.benchmark_adjustment.abnormal_simple_return,
                        abnormal_log_return=point.benchmark_adjustment.abnormal_log_return,
                        status=point.benchmark_adjustment.status,
                        missing_reason=point.benchmark_adjustment.missing_reason,
                    ),
                )
                for point in points
            ],
        )


def _millis(later: datetime, earlier: datetime) -> int:
    return int((later - earlier).total_seconds() * 1000)
