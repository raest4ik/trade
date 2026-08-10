from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.events.domain.entities import (
    DetectedEvent,
    ExtractedFinancialFact,
    NewsEventAnalysis,
)
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
from src.instruments.domain.entities import Instrument, InstrumentMatch, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.market_data.domain.entities import BenchmarkCandle, MarketCandle
from src.ml_features.application.feature_builder import (
    BuildMlFeatureDataset,
    assert_label_separation,
    build_event_fact_features,
    build_labels,
    classify_abnormal_return,
    select_metric_fact,
)
from src.ml_features.application.point_in_time import (
    PointInTimeFeatureBuilder,
    PointInTimeViolationError,
)
from src.ml_features.application.ports import CandidateInstrumentMatch, FeatureCandidate
from src.ml_features.domain.entities import (
    FeatureDatasetBuildResult,
    FeatureDatasetConfig,
    FeatureDatasetRow,
    MlFeatureDatasetRun,
)
from src.ml_features.domain.enums import FeatureDatasetRunStatus, FeatureExclusionReason
from src.ml_features.infrastructure.export import (
    build_manifest,
    csv_columns,
    dataset_stats,
    feature_columns,
    load_jsonl_rows,
    write_csv,
    write_jsonl,
)
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
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

AS_OF = datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Candle:
    end_at: datetime
    close: Decimal
    volume: Decimal


def _config(**overrides: object) -> FeatureDatasetConfig:
    values: dict[str, object] = {
        "date_from": AS_OF - timedelta(days=1),
        "date_to": AS_OF + timedelta(days=1),
    }
    values.update(overrides)
    return FeatureDatasetConfig(**values)  # type: ignore[arg-type]


def _candles(
    *,
    current: Decimal = Decimal("101"),
    volume: Decimal = Decimal("10"),
) -> list[Candle]:
    return [
        Candle(
            end_at=AS_OF - timedelta(minutes=minute),
            close=Decimal("100") + Decimal(60 - minute) / Decimal("100"),
            volume=volume,
        )
        for minute in range(60, 0, -1)
    ] + [Candle(end_at=AS_OF, close=current, volume=volume)]


def _event(event_type: EventType) -> DetectedEvent:
    return DetectedEvent(
        id=uuid4(),
        analysis_id=UUID(int=0),
        event_type=event_type,
        confidence=Decimal("1"),
        rule_id="test-rule",
        matched_rule="test",
        evidence_text="test",
        start_position=0,
        end_position=4,
    )


def _fact(
    metric: FinancialMetric,
    *,
    value: str = "10",
    role: FactRole = FactRole.ACTUAL,
    unit: FactUnit = FactUnit.MONEY,
    change_value: str | None = None,
    change_unit: FactUnit | None = None,
    direction: ChangeDirection = ChangeDirection.UNKNOWN,
    year: int | None = 2026,
    confidence: str = "0.9",
    extractor_version: str = "financial-facts-v2",
    start: int = 0,
) -> ExtractedFinancialFact:
    return ExtractedFinancialFact(
        id=uuid4(),
        analysis_id=UUID(int=0),
        metric=metric,
        raw_value=Decimal(value),
        normalized_value=Decimal(value),
        unit=unit,
        currency=Currency.RUB,
        scale=ValueScale.ONE,
        period_type=PeriodType.YEAR,
        year=year,
        quarter=None,
        month=None,
        date_from=date(year, 1, 1) if year else None,
        date_to=date(year, 12, 31) if year else None,
        raw_period=None,
        comparison_type=ComparisonType.YEAR_OVER_YEAR,
        fact_role=role,
        change_direction=direction,
        change_value=None if change_value is None else Decimal(change_value),
        change_unit=change_unit,
        confidence=Decimal(confidence),
        rule_id="fact-test",
        evidence_text=value,
        start_position=start,
        end_position=start + len(value),
        extractor_version=extractor_version,
        matched_rule="test",
    )


def _analysis(
    *,
    events: list[EventType] | None = None,
    facts: list[ExtractedFinancialFact] | None = None,
) -> NewsEventAnalysis:
    event_types = events or [EventType.FINANCIAL_RESULTS]
    return NewsEventAnalysis.create(
        news_id=uuid4(),
        status=EventAnalysisStatus.COMPLETE,
        primary_event_type=event_types[0],
        events=[_event(item) for item in event_types],
        financial_facts=facts or [],
    )


def _reaction(*, horizons: tuple[int, ...] = (1, 5, 15, 30, 60)) -> NewsMarketReaction:
    points: list[ReactionPoint] = []
    for horizon in horizons:
        point = ReactionPoint.create(
            reaction_id=UUID(int=0),
            horizon_minutes=horizon,
            target_at=AS_OF + timedelta(minutes=horizon),
            observed_at=AS_OF + timedelta(minutes=horizon),
            price=Decimal("101"),
            simple_return=Decimal("0.01"),
            log_return=Decimal("0.00995"),
            status=ReactionPointStatus.AVAILABLE,
        )
        adjustment = ReactionBenchmarkAdjustment.create(
            reaction_point_id=point.id,
            benchmark_id=uuid4(),
            benchmark_code="IMOEX",
            baseline_value=Decimal("1000"),
            target_value=Decimal("1004"),
            baseline_observed_at=AS_OF,
            target_observed_at=AS_OF + timedelta(minutes=horizon),
            simple_return=Decimal("0.004"),
            log_return=Decimal("0.00399"),
            abnormal_simple_return=Decimal("0.006"),
            abnormal_log_return=Decimal("0.00596"),
            status=BenchmarkAdjustmentStatus.AVAILABLE,
        )
        points.append(
            ReactionPoint(
                id=point.id,
                reaction_id=point.reaction_id,
                horizon_minutes=point.horizon_minutes,
                target_at=point.target_at,
                observed_at=point.observed_at,
                price=point.price,
                simple_return=point.simple_return,
                log_return=point.log_return,
                status=point.status,
                benchmark_adjustment=adjustment,
            )
        )
    return NewsMarketReaction.create(
        news_id=uuid4(),
        instrument_id=uuid4(),
        published_at=AS_OF,
        received_at=AS_OF,
        effective_event_at=AS_OF,
        baseline_observed_at=AS_OF,
        baseline_price=Decimal("100"),
        status=ReactionStatus.COMPLETE,
        is_ambiguous_instrument=False,
        points=points,
    )


def _row() -> FeatureDatasetRow:
    return FeatureDatasetRow(
        metadata={
            "news_id": uuid4(),
            "instrument_id": uuid4(),
            "ticker": "SBER",
            "published_at": AS_OF,
            "timestamp_quality": "EXACT",
            "generated_at": AS_OF,
        },
        features={
            "primary_event_type": "FINANCIAL_RESULTS",
            "pre_return_15m": Decimal("0.002"),
            "missing_value": None,
        },
        labels={
            "15m": {
                "available": True,
                "abnormal_simple_return": Decimal("0.006"),
            }
        },
        quality={"missing_features": ["pre_return_60m"]},
    )


def test_config_hash_is_deterministic_and_normalizes_tickers() -> None:
    first = _config(tickers=("gazp", "SBER", "sber"))
    second = _config(tickers=("SBER", "GAZP"))
    assert first.hash() == second.hash()
    assert first.normalized().tickers == ("GAZP", "SBER")


def test_config_hash_changes_with_label_policy() -> None:
    assert _config().hash() != _config(require_label_horizon=15).hash()


@pytest.mark.parametrize("horizon", [5, 15, 30, 60])
def test_pre_returns_are_computed_for_each_horizon(horizon: int) -> None:
    result = PointInTimeFeatureBuilder().build(candles=_candles(), as_of=AS_OF, prefix="s")
    assert result.returns[horizon] is not None
    assert result.log_returns[horizon] is not None


def test_candle_ending_exactly_at_cutoff_is_allowed() -> None:
    result = PointInTimeFeatureBuilder().build(candles=_candles(), as_of=AS_OF, prefix="s")
    assert result.last_observation_end_at == AS_OF


def test_candle_ending_after_cutoff_is_rejected() -> None:
    candles = [
        *_candles(),
        Candle(AS_OF + timedelta(seconds=1), Decimal("999"), Decimal("999")),
    ]
    with pytest.raises(PointInTimeViolationError):
        PointInTimeFeatureBuilder().build(candles=candles, as_of=AS_OF, prefix="s")


def test_post_event_price_and_volume_are_not_silently_used() -> None:
    future = Candle(AS_OF + timedelta(minutes=1), Decimal("999"), Decimal("999"))
    with pytest.raises(PointInTimeViolationError):
        PointInTimeFeatureBuilder().build(candles=[*_candles(), future], as_of=AS_OF, prefix="s")


def test_volatility_uses_only_completed_pre_event_candles() -> None:
    result = PointInTimeFeatureBuilder().build(candles=_candles(), as_of=AS_OF, prefix="s")
    assert result.realized_volatility[15] is not None
    assert result.realized_volatility[30] is not None
    assert result.realized_volatility[60] is not None


def test_volume_features_use_only_pre_event_candles() -> None:
    result = PointInTimeFeatureBuilder().build(
        candles=_candles(volume=Decimal("2")), as_of=AS_OF, prefix="s"
    )
    assert result.volume_last_1m == Decimal("2")
    assert result.volume_sums[5] == Decimal("10")
    assert result.volume_sums[15] == Decimal("30")
    assert result.volume_sums[60] == Decimal("120")
    assert result.volume_ratio_5m_vs_60m == Decimal("1") / Decimal("12")


def test_empty_market_context_has_nulls_and_missing_flag() -> None:
    result = PointInTimeFeatureBuilder().build(candles=[], as_of=AS_OF, prefix="s")
    assert result.returns[15] is None
    assert result.volume_last_1m is None
    assert result.missing == ("s_candles",)


def test_event_flags_cover_existing_event_enum() -> None:
    features = build_event_fact_features(
        _analysis(events=[EventType.DIVIDEND, EventType.SANCTIONS]),
        "financial-facts-v2",
    )
    assert features["has_dividend"] is True
    assert features["has_sanctions"] is True
    for event_type in EventType:
        assert f"event_type_{event_type.value.lower()}" in features


def test_missing_fact_remains_null_with_false_flag() -> None:
    features = build_event_fact_features(_analysis(), "financial-facts-v2")
    assert features["has_net_profit"] is False
    assert features["net_profit_value"] is None


def test_financial_fact_aggregation_uses_semantic_columns() -> None:
    features = build_event_fact_features(
        _analysis(facts=[_fact(FinancialMetric.REVENUE, value="123")]),
        "financial-facts-v2",
    )
    assert features["has_revenue"] is True
    assert features["revenue_value"] == Decimal("123")
    assert "value_1" not in features


def test_duplicate_metric_selection_prefers_actual_then_forecast_then_previous() -> None:
    facts = [
        _fact(FinancialMetric.NET_PROFIT, value="8", role=FactRole.PREVIOUS),
        _fact(FinancialMetric.NET_PROFIT, value="12", role=FactRole.FORECAST),
        _fact(FinancialMetric.NET_PROFIT, value="10", role=FactRole.ACTUAL),
    ]
    assert select_metric_fact(facts).normalized_value == Decimal("10")  # type: ignore[union-attr]


def test_duplicate_metric_selection_prefers_latest_period_within_role() -> None:
    facts = [
        _fact(FinancialMetric.EBITDA, value="10", year=2025),
        _fact(FinancialMetric.EBITDA, value="12", year=2026),
    ]
    assert select_metric_fact(facts).normalized_value == Decimal("12")  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (ChangeDirection.UP, Decimal("18")),
        (ChangeDirection.DOWN, Decimal("-18")),
        (ChangeDirection.UNCHANGED, Decimal("0")),
    ],
)
def test_percent_change_feature_is_signed(direction: ChangeDirection, expected: Decimal) -> None:
    fact = _fact(
        FinancialMetric.NET_PROFIT,
        change_value="18",
        change_unit=FactUnit.PERCENT,
        direction=direction,
    )
    features = build_event_fact_features(_analysis(facts=[fact]), "financial-facts-v2")
    assert features["net_profit_change_pct"] == expected
    assert features["net_profit_change_direction"] == direction.value
    assert features["net_profit_change_unit"] == FactUnit.PERCENT.value
    assert features["net_profit_change_comparison_type"] == fact.comparison_type.value


def test_percentage_points_are_not_mixed_with_percent() -> None:
    fact = _fact(
        FinancialMetric.NET_PROFIT,
        change_value="3",
        change_unit=FactUnit.PERCENTAGE_POINTS,
        direction=ChangeDirection.UP,
    )
    features = build_event_fact_features(_analysis(facts=[fact]), "financial-facts-v2")
    assert features["net_profit_change_pct"] is None
    assert features["net_profit_change_unit"] is None
    assert features["net_profit_change_comparison_type"] is None


def test_wrong_fact_extractor_version_is_not_used() -> None:
    fact = _fact(FinancialMetric.REVENUE, extractor_version="financial-facts-v1")
    features = build_event_fact_features(_analysis(facts=[fact]), "financial-facts-v2")
    assert features["fact_count"] == 0
    assert features["revenue_value"] is None


def test_guidance_counts_are_deterministic() -> None:
    facts = [
        _fact(FinancialMetric.REVENUE, role=FactRole.FORECAST, direction=ChangeDirection.UP),
        _fact(FinancialMetric.CAPEX, role=FactRole.TARGET, direction=ChangeDirection.DOWN),
        _fact(
            FinancialMetric.EBITDA,
            role=FactRole.FORECAST,
            direction=ChangeDirection.UNCHANGED,
        ),
    ]
    features = build_event_fact_features(
        _analysis(events=[EventType.GUIDANCE], facts=facts), "financial-facts-v2"
    )
    assert features["guidance_fact_count"] == 3
    assert features["guidance_up_count"] == 1
    assert features["guidance_down_count"] == 1
    assert features["guidance_unchanged_count"] == 1


def test_dividend_value_and_role_are_explicit() -> None:
    fact = _fact(
        FinancialMetric.DIVIDEND_PER_SHARE,
        value="25",
        role=FactRole.FORECAST,
    )
    features = build_event_fact_features(
        _analysis(events=[EventType.DIVIDEND], facts=[fact]), "financial-facts-v2"
    )
    assert features["dividend_per_share"] == Decimal("25")
    assert features["dividend_role"] == "FORECAST"


@pytest.mark.parametrize("horizon", [1, 5, 15, 30, 60])
def test_all_label_horizons_are_separate_and_available(horizon: int) -> None:
    labels = build_labels(_reaction(), _config())
    assert labels[f"{horizon}m"]["abnormal_simple_return"] == Decimal("0.006")


def test_missing_reaction_horizon_remains_unavailable() -> None:
    labels = build_labels(_reaction(horizons=(15,)), _config())
    assert labels["1m"]["available"] is False
    assert labels["1m"]["abnormal_simple_return"] is None


def test_classification_is_optional_by_default() -> None:
    assert classify_abnormal_return(Decimal("0.006"), threshold=None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0.006"), "UP"),
        (Decimal("-0.006"), "DOWN"),
        (Decimal("0.001"), "FLAT"),
    ],
)
def test_configurable_research_classification(value: Decimal, expected: str) -> None:
    assert classify_abnormal_return(value, threshold=Decimal("0.002")) == expected


def test_labels_never_appear_inside_features() -> None:
    row = _row()
    assert_label_separation(row)
    assert "abnormal_simple_return" not in row.features


def test_label_leakage_assertion_rejects_forbidden_feature() -> None:
    row = _row()
    row.features["abnormal_simple_return"] = Decimal("0.006")
    with pytest.raises(AssertionError):
        assert_label_separation(row)


def test_jsonl_schema_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, [_row()])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"metadata", "features", "labels", "quality"}
    assert load_jsonl_rows(path)[0].labels["15m"]["abnormal_simple_return"] == Decimal("0.006")


def test_csv_schema_and_column_order_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    write_csv(path, [_row()])
    with path.open(encoding="utf-8", newline="") as source:
        header = next(csv.reader(source))
    assert tuple(header) == csv_columns()
    assert header.index("features.pre_return_15m") < header.index(
        "labels.15m_abnormal_simple_return"
    )


def test_feature_columns_are_not_data_dependent() -> None:
    assert "features.random" not in csv_columns()
    assert "pre_return_60m" in feature_columns()


def test_manifest_records_exact_versions_and_config_hash() -> None:
    config = _config(require_label_horizon=15)
    run = MlFeatureDatasetRun.start(config, git_sha="abc")
    result = FeatureDatasetBuildResult(rows=[_row()], exclusions=[], run=run)
    manifest = build_manifest(result=result, config=config, stats=dataset_stats([_row()]))
    assert manifest["config_hash"] == config.hash()
    assert manifest["event_analysis_version"] == "event-rules-v2"
    assert manifest["fact_extractor_version"] == "financial-facts-v2"
    assert manifest["reaction_version"] == "reaction-v2-benchmark-adjusted"


def test_stats_report_tiny_sample_as_insufficient() -> None:
    stats = dataset_stats([_row()])
    assert stats["rows_total"] == 1
    assert stats["label_availability_by_horizon"]["15m"] == 1
    assert stats["sample_interpretation"] == "INSUFFICIENT_SAMPLE_FOR_INFERENCE"
    distribution = stats["abnormal_simple_return_distributions"]["15m"]
    assert distribution["mean"] == Decimal("0.006")


def test_run_audit_finishes_with_counts() -> None:
    run = MlFeatureDatasetRun.start(_config(), git_sha="abc").finish(
        status=FeatureDatasetRunStatus.PARTIAL,
        candidate_count=2,
        eligible_count=1,
        built_count=1,
        excluded_count=1,
        failed_count=0,
    )
    assert run.finished_at is not None
    assert run.built_count == 1
    assert run.status == FeatureDatasetRunStatus.PARTIAL


class FakeMlRepository:
    def __init__(
        self,
        candidates: list[FeatureCandidate],
        *,
        security: list[MarketCandle] | None = None,
        benchmark: list[BenchmarkCandle] | None = None,
    ) -> None:
        self.candidates = candidates
        self.security = security if security is not None else _security_market_candles()
        self.benchmark = benchmark if benchmark is not None else _benchmark_market_candles()
        self.created_runs = 0

    async def list_candidates(self, config: FeatureDatasetConfig) -> list[FeatureCandidate]:
        del config
        return self.candidates

    async def list_security_candles_as_of(
        self, *, instrument_id: UUID, as_of: datetime, lookback_minutes: int
    ) -> list[MarketCandle]:
        del instrument_id, as_of, lookback_minutes
        return self.security

    async def list_benchmark_candles_as_of(
        self, *, benchmark_code: str, as_of: datetime, lookback_minutes: int
    ) -> list[BenchmarkCandle] | None:
        del benchmark_code, as_of, lookback_minutes
        return self.benchmark

    async def create_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun:
        self.created_runs += 1
        return run

    async def finish_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun:
        return run


class FakeEventRepository:
    def __init__(self, analysis: NewsEventAnalysis | None) -> None:
        self.analysis = analysis
        self.requested_version: str | None = None

    async def replace_analysis(self, analysis: NewsEventAnalysis) -> NewsEventAnalysis:
        self.analysis = analysis
        return analysis

    async def get_by_news_id(
        self, *, news_id: UUID, analysis_version: str | None = None
    ) -> NewsEventAnalysis | None:
        del news_id
        self.requested_version = analysis_version
        return self.analysis


class FakeReactionRepository:
    def __init__(self, reactions: list[NewsMarketReaction]) -> None:
        self.reactions = reactions

    async def replace_reactions(
        self,
        *,
        news_id: UUID,
        reaction_version: str,
        reactions: list[NewsMarketReaction],
    ) -> list[NewsMarketReaction]:
        del news_id, reaction_version
        self.reactions = reactions
        return reactions

    async def get_news_reactions(
        self, *, news_id: UUID, reaction_version: str | None = None
    ) -> list[NewsMarketReaction]:
        del news_id, reaction_version
        return self.reactions


def _feature_candidate(
    quality: PublicationTimestampQuality = PublicationTimestampQuality.EXACT,
    *,
    match_count: int = 1,
    ambiguous: bool = False,
) -> tuple[FeatureCandidate, Instrument]:
    news = NewsItem.create(
        source_id="item-1",
        source_name="source",
        source_url="https://example.invalid/item-1",
        title="SBER results",
        raw_content="SBER results",
        language="en",
        published_at=AS_OF,
        received_at=AS_OF,
        publication_timestamp_quality=quality,
    )
    instrument = Instrument.create(
        ticker="SBER",
        figi=None,
        isin=None,
        short_name="Sber",
        full_name="Sber",
        issuer_name="Sber",
        exchange="MOEX",
        currency="RUB",
        instrument_type=InstrumentType.COMMON_STOCK,
        primary_board="TQBR",
    )
    matches = [
        CandidateInstrumentMatch(
            match=NewsInstrumentMatch.create(
                news_id=news.id,
                match=InstrumentMatch(
                    instrument_id=instrument.id,
                    ticker=instrument.ticker,
                    issuer_name=instrument.issuer_name,
                    matched_alias="SBER",
                    alias_type=AliasType.TICKER,
                    match_type=MatchType.EXACT_TICKER,
                    confidence=1.0,
                    start_position=0,
                    end_position=4,
                    is_ambiguous=ambiguous,
                ),
            ),
            instrument=instrument,
        )
        for _ in range(match_count)
    ]
    return (
        FeatureCandidate(
            news=news,
            source_code="TEST",
            source_item_id="item-1",
            matches=matches,
        ),
        instrument,
    )


def _security_market_candles(*, future: bool = False) -> list[MarketCandle]:
    end = AS_OF + timedelta(seconds=1) if future else AS_OF
    return [
        MarketCandle.create(
            instrument_id=UUID(int=1),
            board="TQBR",
            ticker_snapshot="SBER",
            interval_minutes=1,
            begin_at=AS_OF - timedelta(minutes=15, seconds=59),
            end_at=AS_OF - timedelta(minutes=15),
            open_price=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("10"),
            value=Decimal("1000"),
        ),
        MarketCandle.create(
            instrument_id=UUID(int=1),
            board="TQBR",
            ticker_snapshot="SBER",
            interval_minutes=1,
            begin_at=end - timedelta(seconds=59),
            end_at=end,
            open_price=Decimal("100.2"),
            high=Decimal("100.2"),
            low=Decimal("100.2"),
            close=Decimal("100.2"),
            volume=Decimal("20"),
            value=Decimal("2004"),
        ),
    ]


def _benchmark_market_candles() -> list[BenchmarkCandle]:
    benchmark_id = UUID(int=2)
    return [
        BenchmarkCandle.create(
            benchmark_id=benchmark_id,
            interval_minutes=1,
            begin_at=AS_OF - timedelta(minutes=15, seconds=59),
            end_at=AS_OF - timedelta(minutes=15),
            open_price=Decimal("1000"),
            high=Decimal("1000"),
            low=Decimal("1000"),
            close=Decimal("1000"),
            volume=Decimal("1"),
            value=Decimal("1000"),
        ),
        BenchmarkCandle.create(
            benchmark_id=benchmark_id,
            interval_minutes=1,
            begin_at=AS_OF - timedelta(seconds=59),
            end_at=AS_OF,
            open_price=Decimal("1001"),
            high=Decimal("1001"),
            low=Decimal("1001"),
            close=Decimal("1001"),
            volume=Decimal("1"),
            value=Decimal("1001"),
        ),
    ]


async def _build_with_fakes(
    candidate: FeatureCandidate,
    instrument: Instrument,
    *,
    analysis_available: bool = True,
    reactions: list[NewsMarketReaction] | None = None,
    security: list[MarketCandle] | None = None,
    benchmark: list[BenchmarkCandle] | None = None,
    dry_run: bool = True,
) -> tuple[FeatureDatasetBuildResult, FakeMlRepository, FakeEventRepository]:
    resolved_analysis = _analysis() if analysis_available else None
    event_repository = FakeEventRepository(resolved_analysis)
    if reactions is None:
        reaction = replace(
            _reaction(horizons=(15,)),
            news_id=candidate.news.id,
            instrument_id=instrument.id,
        )
        reactions = [reaction]
    repository = FakeMlRepository(
        [candidate],
        security=security,
        benchmark=benchmark,
    )
    result = await BuildMlFeatureDataset(
        repository=repository,
        event_repository=event_repository,
        reaction_repository=FakeReactionRepository(reactions),
    ).execute(config=_config(require_label_horizon=15), git_sha="abc", dry_run=dry_run)
    return result, repository, event_repository


async def test_exact_candidate_is_eligible_and_version_is_explicit() -> None:
    candidate, instrument = _feature_candidate()
    result, _, events = await _build_with_fakes(candidate, instrument)
    assert result.run.built_count == 1
    assert events.requested_version == "event-rules-v2"


@pytest.mark.parametrize(
    "quality",
    [PublicationTimestampQuality.DATE_ONLY, PublicationTimestampQuality.UNKNOWN],
)
async def test_inexact_timestamp_is_excluded(quality: PublicationTimestampQuality) -> None:
    candidate, instrument = _feature_candidate(quality)
    result, _, _ = await _build_with_fakes(candidate, instrument)
    assert result.exclusions[0].reason == FeatureExclusionReason.TIMESTAMP_NOT_EXACT


async def test_no_instrument_match_is_excluded() -> None:
    candidate, instrument = _feature_candidate(match_count=0)
    result, _, _ = await _build_with_fakes(candidate, instrument)
    assert result.exclusions[0].reason == FeatureExclusionReason.NO_INSTRUMENT_MATCH


@pytest.mark.parametrize(
    ("match_count", "ambiguous"),
    [(1, True), (2, False)],
)
async def test_ambiguous_instrument_match_is_excluded(match_count: int, ambiguous: bool) -> None:
    candidate, instrument = _feature_candidate(match_count=match_count, ambiguous=ambiguous)
    result, _, _ = await _build_with_fakes(candidate, instrument)
    assert result.exclusions[0].reason == FeatureExclusionReason.AMBIGUOUS_INSTRUMENT


async def test_missing_event_analysis_is_excluded() -> None:
    candidate, instrument = _feature_candidate()
    result, _, _ = await _build_with_fakes(candidate, instrument, analysis_available=False)
    assert result.exclusions[0].reason == FeatureExclusionReason.NO_EVENT_ANALYSIS


async def test_missing_reaction_label_is_excluded() -> None:
    candidate, instrument = _feature_candidate()
    result, _, _ = await _build_with_fakes(candidate, instrument, reactions=[])
    assert result.exclusions[0].reason == FeatureExclusionReason.NO_REACTION_LABEL


async def test_missing_market_context_has_explicit_exclusion() -> None:
    candidate, instrument = _feature_candidate()
    result, _, _ = await _build_with_fakes(candidate, instrument, security=[])
    assert result.exclusions[0].reason == FeatureExclusionReason.MARKET_DATA_MISSING


async def test_missing_benchmark_context_has_explicit_exclusion() -> None:
    candidate, instrument = _feature_candidate()
    result, _, _ = await _build_with_fakes(candidate, instrument, benchmark=[])
    assert result.exclusions[0].reason == FeatureExclusionReason.BENCHMARK_DATA_MISSING


async def test_future_candle_becomes_point_in_time_exclusion() -> None:
    candidate, instrument = _feature_candidate()
    result, _, _ = await _build_with_fakes(
        candidate,
        instrument,
        security=_security_market_candles(future=True),
    )
    assert result.exclusions[0].reason == FeatureExclusionReason.POINT_IN_TIME_VIOLATION


async def test_dry_run_does_not_persist_audit_run() -> None:
    candidate, instrument = _feature_candidate()
    _, repository, _ = await _build_with_fakes(candidate, instrument, dry_run=True)
    assert repository.created_runs == 0


async def test_repeated_build_is_deterministic_with_fixed_generation_time() -> None:
    candidate, instrument = _feature_candidate()
    first, _, _ = await _build_with_fakes(candidate, instrument)
    second, _, _ = await _build_with_fakes(candidate, instrument)
    first_payload = first.rows[0].payload()
    second_payload = second.rows[0].payload()
    first_payload["metadata"]["generated_at"] = AS_OF
    second_payload["metadata"]["generated_at"] = AS_OF
    assert first_payload == second_payload
