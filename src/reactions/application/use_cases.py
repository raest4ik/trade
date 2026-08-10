from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from uuid import UUID

from src.instruments.application.ports import InstrumentRepository
from src.market_data.application.ports import MarketDataRepository
from src.market_data.domain.entities import (
    IMOEX_BENCHMARK_CODE,
    MOEX_INDEX_BOARD,
    MarketBenchmark,
)
from src.news.application.ports import NewsRepository
from src.news.domain.enums import PublicationTimestampQuality
from src.reactions.application.exceptions import (
    ReactionMissingInstrumentMatchesError,
    ReactionNewsNotFoundError,
    ReactionTimestampIneligibleError,
)
from src.reactions.application.ports import ReactionRepository
from src.reactions.domain.entities import (
    DEFAULT_REACTION_HORIZONS_MINUTES,
    REACTION_VERSION,
    NewsMarketReaction,
    ReactionBenchmarkAdjustment,
    ReactionPoint,
)
from src.reactions.domain.enums import (
    BenchmarkAdjustmentStatus,
    ReactionPointStatus,
    ReactionStatus,
)

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
        if news_item.publication_timestamp_quality != PublicationTimestampQuality.EXACT:
            raise ReactionTimestampIneligibleError(
                "market reactions require publication_timestamp_quality=EXACT"
            )
        matches = await self._instrument_repository.get_news_matches(news_id)
        if not matches:
            raise ReactionMissingInstrumentMatchesError("instrument matching was not run")
        benchmark = await self._market_data_repository.get_benchmark_by_code(IMOEX_BENCHMARK_CODE)
        if benchmark is None:
            benchmark = await self._market_data_repository.save_benchmark(
                MarketBenchmark.create(
                    code=IMOEX_BENCHMARK_CODE,
                    name="MOEX Russia Index",
                    board=MOEX_INDEX_BOARD,
                )
            )
        reactions: list[NewsMarketReaction] = []
        for match in matches:
            baseline = await self._market_data_repository.get_last_candle_ending_at_or_before(
                instrument_id=match.instrument_id,
                interval_minutes=1,
                at=news_item.published_at,
            )
            effective = await self._market_data_repository.get_first_candle_beginning_at_or_after(
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
                        points=_missing_points(
                            None,
                            self._horizons_minutes,
                            benchmark=benchmark,
                            reason="security_baseline_or_effective_candle_missing",
                        ),
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
                        _with_not_applicable_benchmark(
                            ReactionPoint.create(
                                reaction_id=UUID(int=0),
                                horizon_minutes=horizon,
                                target_at=target_at,
                                observed_at=None,
                                price=None,
                                simple_return=None,
                                log_return=None,
                                status=ReactionPointStatus.MISSING_CANDLE,
                            ),
                            benchmark=benchmark,
                            reason="security_target_candle_missing",
                        )
                    )
                    continue
                simple_return = target.close / baseline.close - Decimal("1")
                with localcontext() as context:
                    context.prec = 28
                    log_return = (target.close / baseline.close).ln()
                security_point = ReactionPoint.create(
                    reaction_id=UUID(int=0),
                    horizon_minutes=horizon,
                    target_at=target_at,
                    observed_at=target.end_at,
                    price=target.close,
                    simple_return=simple_return,
                    log_return=log_return,
                    status=ReactionPointStatus.AVAILABLE,
                )
                points.append(
                    await _with_benchmark_adjustment(
                        security_point,
                        security_baseline_observed_at=baseline.end_at,
                        security_baseline_price=baseline.close,
                        benchmark=benchmark,
                        repository=self._market_data_repository,
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
        reactions = await self._repository.get_news_reactions(
            news_id=news_id,
            reaction_version=REACTION_VERSION,
        )
        if not reactions:
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
    *,
    benchmark: MarketBenchmark,
    reason: str,
) -> list[ReactionPoint]:
    from datetime import UTC, datetime

    target_seed = datetime.now(UTC) if effective_event_at is None else effective_event_at
    return [
        _with_not_applicable_benchmark(
            ReactionPoint.create(
                reaction_id=UUID(int=0),
                horizon_minutes=horizon,
                target_at=target_seed,
                observed_at=None,
                price=None,
                simple_return=None,
                log_return=None,
                status=ReactionPointStatus.OUTSIDE_AVAILABLE_RANGE,
            ),
            benchmark=benchmark,
            reason=reason,
        )
        for horizon in horizons_minutes
    ]


def _with_not_applicable_benchmark(
    point: ReactionPoint,
    *,
    benchmark: MarketBenchmark,
    reason: str,
) -> ReactionPoint:
    return replace(
        point,
        benchmark_adjustment=ReactionBenchmarkAdjustment.create(
            reaction_point_id=point.id,
            benchmark_id=benchmark.id,
            benchmark_code=benchmark.code,
            baseline_value=None,
            target_value=None,
            baseline_observed_at=None,
            target_observed_at=None,
            simple_return=None,
            log_return=None,
            abnormal_simple_return=None,
            abnormal_log_return=None,
            status=BenchmarkAdjustmentStatus.NOT_APPLICABLE,
            missing_reason=reason,
        ),
    )


async def _with_benchmark_adjustment(
    point: ReactionPoint,
    *,
    security_baseline_observed_at: datetime,
    security_baseline_price: Decimal,
    benchmark: MarketBenchmark,
    repository: MarketDataRepository,
) -> ReactionPoint:
    if point.observed_at is None or point.price is None:
        return _with_not_applicable_benchmark(
            point,
            benchmark=benchmark,
            reason="security_target_candle_missing",
        )
    benchmark_baseline = await repository.get_last_benchmark_candle_ending_at_or_before(
        benchmark_id=benchmark.id,
        interval_minutes=1,
        at=security_baseline_observed_at,
    )
    benchmark_target = await repository.get_first_benchmark_candle_ending_at_or_after(
        benchmark_id=benchmark.id,
        interval_minutes=1,
        at=point.observed_at,
    )
    if benchmark_baseline is None or benchmark_target is None:
        missing: list[str] = []
        if benchmark_baseline is None:
            missing.append("baseline")
        if benchmark_target is None:
            missing.append("target")
        adjustment = ReactionBenchmarkAdjustment.create(
            reaction_point_id=point.id,
            benchmark_id=benchmark.id,
            benchmark_code=benchmark.code,
            baseline_value=None if benchmark_baseline is None else benchmark_baseline.close,
            target_value=None if benchmark_target is None else benchmark_target.close,
            baseline_observed_at=None if benchmark_baseline is None else benchmark_baseline.end_at,
            target_observed_at=None if benchmark_target is None else benchmark_target.end_at,
            simple_return=None,
            log_return=None,
            abnormal_simple_return=None,
            abnormal_log_return=None,
            status=BenchmarkAdjustmentStatus.MISSING,
            missing_reason=f"benchmark_{'_and_'.join(missing)}_candle_missing",
        )
        return replace(point, benchmark_adjustment=adjustment)
    if benchmark_baseline.close <= Decimal("0"):
        adjustment = ReactionBenchmarkAdjustment.create(
            reaction_point_id=point.id,
            benchmark_id=benchmark.id,
            benchmark_code=benchmark.code,
            baseline_value=benchmark_baseline.close,
            target_value=benchmark_target.close,
            baseline_observed_at=benchmark_baseline.end_at,
            target_observed_at=benchmark_target.end_at,
            simple_return=None,
            log_return=None,
            abnormal_simple_return=None,
            abnormal_log_return=None,
            status=BenchmarkAdjustmentStatus.MISSING,
            missing_reason="benchmark_baseline_not_positive",
        )
        return replace(point, benchmark_adjustment=adjustment)
    benchmark_simple = benchmark_target.close / benchmark_baseline.close - Decimal("1")
    with localcontext() as context:
        context.prec = 28
        benchmark_log = (benchmark_target.close / benchmark_baseline.close).ln()
    security_simple = point.price / security_baseline_price - Decimal("1")
    with localcontext() as context:
        context.prec = 28
        security_log = (point.price / security_baseline_price).ln()
    adjustment = ReactionBenchmarkAdjustment.create(
        reaction_point_id=point.id,
        benchmark_id=benchmark.id,
        benchmark_code=benchmark.code,
        baseline_value=benchmark_baseline.close,
        target_value=benchmark_target.close,
        baseline_observed_at=benchmark_baseline.end_at,
        target_observed_at=benchmark_target.end_at,
        simple_return=benchmark_simple,
        log_return=benchmark_log,
        abnormal_simple_return=security_simple - benchmark_simple,
        abnormal_log_return=security_log - benchmark_log,
        status=BenchmarkAdjustmentStatus.AVAILABLE,
    )
    return replace(point, benchmark_adjustment=adjustment)


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
