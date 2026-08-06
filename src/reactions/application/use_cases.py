from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from uuid import UUID

from src.instruments.application.ports import InstrumentRepository
from src.market_data.application.ports import MarketDataRepository
from src.news.application.ports import NewsRepository
from src.reactions.application.exceptions import (
    ReactionMissingInstrumentMatchesError,
    ReactionNewsNotFoundError,
)
from src.reactions.application.ports import ReactionRepository
from src.reactions.domain.entities import (
    DEFAULT_REACTION_HORIZONS_MINUTES,
    REACTION_VERSION,
    NewsMarketReaction,
    ReactionPoint,
)
from src.reactions.domain.enums import ReactionPointStatus, ReactionStatus

OUTSIDE_SESSION_GAP = timedelta(minutes=90)


@dataclass(frozen=True, slots=True)
class CalculateNewsMarketReactionsResult:
    news_id: UUID
    reaction_version: str
    reactions: list[NewsMarketReaction]


class CalculateNewsMarketReactions:
    def __init__(
        self,
        *,
        news_repository: NewsRepository,
        instrument_repository: InstrumentRepository,
        market_data_repository: MarketDataRepository,
        reaction_repository: ReactionRepository,
        horizons_minutes: tuple[int, ...] = DEFAULT_REACTION_HORIZONS_MINUTES,
        reaction_version: str = REACTION_VERSION,
    ) -> None:
        self._news_repository = news_repository
        self._instrument_repository = instrument_repository
        self._market_data_repository = market_data_repository
        self._reaction_repository = reaction_repository
        self._horizons_minutes = horizons_minutes
        self._reaction_version = reaction_version

    async def execute(self, news_id: UUID) -> CalculateNewsMarketReactionsResult:
        news_item = await self._news_repository.get_by_id(news_id)
        if news_item is None:
            raise ReactionNewsNotFoundError("news item not found")
        matches = await self._instrument_repository.get_news_matches(news_id)
        if not matches:
            raise ReactionMissingInstrumentMatchesError("instrument matching was not run")
        reactions: list[NewsMarketReaction] = []
        for match in matches:
            baseline = await self._market_data_repository.get_last_candle_ending_at_or_before(
                instrument_id=match.instrument_id,
                interval_minutes=1,
                at=news_item.published_at,
            )
            effective = await self._market_data_repository.get_first_candle_beginning_after(
                instrument_id=match.instrument_id,
                interval_minutes=1,
                at=news_item.published_at,
            )
            if baseline is None or effective is None or baseline.close <= Decimal("0"):
                reactions.append(
                    NewsMarketReaction.create(
                        news_id=news_id,
                        instrument_id=match.instrument_id,
                        published_at=news_item.published_at,
                        received_at=news_item.received_at,
                        effective_event_at=None if effective is None else effective.begin_at,
                        baseline_observed_at=None if baseline is None else baseline.end_at,
                        baseline_price=None if baseline is None else baseline.close,
                        status=ReactionStatus.INSUFFICIENT_DATA,
                        is_ambiguous_instrument=match.is_ambiguous,
                        points=_missing_points(None, self._horizons_minutes),
                        reaction_version=self._reaction_version,
                    )
                )
                continue
            points: list[ReactionPoint] = []
            for horizon in self._horizons_minutes:
                target_at = effective.begin_at + timedelta(minutes=horizon)
                target = await self._market_data_repository.get_first_candle_ending_at_or_after(
                    instrument_id=match.instrument_id,
                    interval_minutes=1,
                    at=target_at,
                )
                if target is None:
                    points.append(
                        ReactionPoint.create(
                            reaction_id=UUID(int=0),
                            horizon_minutes=horizon,
                            target_at=target_at,
                            observed_at=None,
                            price=None,
                            simple_return=None,
                            log_return=None,
                            status=ReactionPointStatus.MISSING_CANDLE,
                        )
                    )
                    continue
                simple_return = target.close / baseline.close - Decimal("1")
                with localcontext() as context:
                    context.prec = 28
                    log_return = (target.close / baseline.close).ln()
                points.append(
                    ReactionPoint.create(
                        reaction_id=UUID(int=0),
                        horizon_minutes=horizon,
                        target_at=target_at,
                        observed_at=target.end_at,
                        price=target.close,
                        simple_return=simple_return,
                        log_return=log_return,
                        status=ReactionPointStatus.AVAILABLE,
                    )
                )
            status = _reaction_status(
                ambiguous=match.is_ambiguous,
                outside_session=effective.begin_at - news_item.published_at > OUTSIDE_SESSION_GAP,
                points=points,
            )
            reactions.append(
                NewsMarketReaction.create(
                    news_id=news_id,
                    instrument_id=match.instrument_id,
                    published_at=news_item.published_at,
                    received_at=news_item.received_at,
                    effective_event_at=effective.begin_at,
                    baseline_observed_at=baseline.end_at,
                    baseline_price=baseline.close,
                    status=status,
                    is_ambiguous_instrument=match.is_ambiguous,
                    points=points,
                    reaction_version=self._reaction_version,
                )
            )
        saved = await self._reaction_repository.replace_reactions(
            news_id=news_id,
            reaction_version=self._reaction_version,
            reactions=reactions,
        )
        return CalculateNewsMarketReactionsResult(
            news_id=news_id,
            reaction_version=self._reaction_version,
            reactions=saved,
        )


class GetNewsMarketReactions:
    def __init__(self, repository: ReactionRepository) -> None:
        self._repository = repository

    async def execute(self, news_id: UUID) -> CalculateNewsMarketReactionsResult:
        reactions = await self._repository.get_news_reactions(news_id=news_id)
        version = reactions[0].reaction_version if reactions else REACTION_VERSION
        return CalculateNewsMarketReactionsResult(
            news_id=news_id,
            reaction_version=version,
            reactions=reactions,
        )


def _missing_points(
    effective_event_at: datetime | None,
    horizons_minutes: tuple[int, ...],
) -> list[ReactionPoint]:
    from datetime import UTC, datetime

    target_seed = datetime.now(UTC) if effective_event_at is None else effective_event_at
    return [
        ReactionPoint.create(
            reaction_id=UUID(int=0),
            horizon_minutes=horizon,
            target_at=target_seed,
            observed_at=None,
            price=None,
            simple_return=None,
            log_return=None,
            status=ReactionPointStatus.OUTSIDE_AVAILABLE_RANGE,
        )
        for horizon in horizons_minutes
    ]


def _reaction_status(
    *,
    ambiguous: bool,
    outside_session: bool,
    points: list[ReactionPoint],
) -> ReactionStatus:
    if ambiguous:
        return ReactionStatus.AMBIGUOUS_INSTRUMENT
    if outside_session:
        return ReactionStatus.OUTSIDE_SESSION
    if all(point.status == ReactionPointStatus.AVAILABLE for point in points):
        return ReactionStatus.COMPLETE
    if any(point.status == ReactionPointStatus.AVAILABLE for point in points):
        return ReactionStatus.PARTIAL
    return ReactionStatus.INSUFFICIENT_DATA
