from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from apps.cli.export_market_reaction_dataset import (
    DATASET_SCHEMA_VERSION,
    market_reaction_row_payload,
)
from src.instruments.infrastructure.models import InstrumentRecord
from src.market_data.domain.entities import (
    MOEX_INDEX_BOARD,
    BenchmarkCandle,
    MarketBenchmark,
)
from src.market_data.domain.exceptions import MarketDataDomainError
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.reactions.domain.entities import (
    DEFAULT_REACTION_HORIZONS_MINUTES,
    NewsMarketReaction,
    ReactionBenchmarkAdjustment,
    ReactionPoint,
)
from src.reactions.domain.enums import (
    BenchmarkAdjustmentStatus,
    ReactionPointStatus,
    ReactionStatus,
)
from src.reactions.infrastructure.models import NewsMarketReactionRecord, ReactionPointRecord
from src.reactions.presentation.schemas import ReactionPointResponse


def test_news_timestamp_quality_defaults_to_unknown() -> None:
    assert _news().publication_timestamp_quality == PublicationTimestampQuality.UNKNOWN


def test_news_accepts_exact_timestamp_quality() -> None:
    item = _news(PublicationTimestampQuality.EXACT)
    assert item.publication_timestamp_quality == PublicationTimestampQuality.EXACT


def test_news_accepts_date_only_timestamp_quality() -> None:
    item = _news(PublicationTimestampQuality.DATE_ONLY)
    assert item.publication_timestamp_quality == PublicationTimestampQuality.DATE_ONLY


def test_market_benchmark_normalizes_identity() -> None:
    benchmark = MarketBenchmark.create(code=" imoex ", name=" MOEX Russia Index ", board="sndx")
    assert (benchmark.code, benchmark.name, benchmark.board) == (
        "IMOEX",
        "MOEX Russia Index",
        MOEX_INDEX_BOARD,
    )


def test_market_benchmark_rejects_blank_name() -> None:
    with pytest.raises(MarketDataDomainError):
        MarketBenchmark.create(code="IMOEX", name=" ", board="SNDX")


def test_benchmark_candle_normalizes_utc() -> None:
    candle = _benchmark_candle(
        uuid4(),
        datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        "100",
    )
    assert candle.end_at == datetime(2026, 7, 1, 10, 0, 59, tzinfo=UTC)


def test_benchmark_candle_preserves_decimal() -> None:
    candle = _benchmark_candle(uuid4(), _at(0), "123.456789")
    assert candle.close == Decimal("123.456789")
    assert isinstance(candle.close, Decimal)


def test_benchmark_candle_rejects_invalid_ohlc() -> None:
    with pytest.raises(MarketDataDomainError):
        BenchmarkCandle.create(
            benchmark_id=uuid4(),
            interval_minutes=1,
            begin_at=_at(0),
            end_at=_at(0) + timedelta(seconds=59),
            open_price=Decimal("120"),
            high=Decimal("110"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1"),
            value=Decimal("1"),
        )


@pytest.mark.parametrize("horizon", DEFAULT_REACTION_HORIZONS_MINUTES)
def test_all_required_horizons_create_stable_points(horizon: int) -> None:
    point = ReactionPoint.create(
        reaction_id=uuid4(),
        horizon_minutes=horizon,
        target_at=_at(horizon),
        observed_at=None,
        price=None,
        simple_return=None,
        log_return=None,
        status=ReactionPointStatus.MISSING_CANDLE,
    )
    assert point.horizon_minutes == horizon


def test_missing_benchmark_adjustment_never_invents_zero() -> None:
    adjustment = _adjustment(BenchmarkAdjustmentStatus.MISSING)
    assert adjustment.simple_return is None
    assert adjustment.abnormal_simple_return is None


def test_available_benchmark_adjustment_serializes_api_fields() -> None:
    point = _point_with_adjustment()
    response = ReactionPointResponse.from_entity(point)
    assert response.security_simple_return == Decimal("0.02")
    assert response.benchmark_code == "IMOEX"
    assert response.benchmark_simple_return == Decimal("0.01")
    assert response.abnormal_simple_return == Decimal("0.01")


def test_export_v2_separates_labels_from_security_values() -> None:
    point_entity = _point_with_adjustment()
    reaction_entity = NewsMarketReaction.create(
        news_id=uuid4(),
        instrument_id=uuid4(),
        published_at=_at(0),
        received_at=_at(0),
        effective_event_at=_at(1),
        baseline_observed_at=_at(0),
        baseline_price=Decimal("100"),
        status=ReactionStatus.COMPLETE,
        is_ambiguous_instrument=False,
        points=[point_entity],
    )
    reaction_record = NewsMarketReactionRecord.from_entity(reaction_entity)
    point_record = reaction_record.points[0]
    news_record = NewsItemRecord.from_entity(_news(PublicationTimestampQuality.EXACT))
    instrument_record = InstrumentRecord(
        id=reaction_entity.instrument_id,
        ticker="SBER",
        figi=None,
        isin=None,
        short_name="Sber",
        full_name="Sber",
        issuer_name="Sber",
        exchange="MOEX",
        currency="RUB",
        instrument_type="COMMON_STOCK",
        primary_board="TQBR",
        is_active=True,
        created_at=_at(0),
        updated_at=_at(0),
    )
    payload = market_reaction_row_payload(
        point=point_record,
        reaction=reaction_record,
        news=news_record,
        instrument=instrument_record,
        analysis_version="event-rules-v2",
    )
    assert payload["schema_version"] == DATASET_SCHEMA_VERSION
    assert payload["timestamp_quality"] == "EXACT"
    labels = payload["labels"]
    assert isinstance(labels, dict)
    assert labels["abnormal_simple_return"] == Decimal("0.01")


def test_reaction_adjustment_record_round_trip() -> None:
    point = _point_with_adjustment()
    record = ReactionPointRecord.from_entity(point)
    restored = record.to_entity()
    assert restored.benchmark_adjustment == point.benchmark_adjustment


def _news(
    quality: PublicationTimestampQuality = PublicationTimestampQuality.UNKNOWN,
) -> NewsItem:
    return NewsItem.create(
        source_id=str(uuid4()),
        source_name="unit-test",
        source_url="https://example.com/news",
        title="SBER",
        raw_content="SBER news",
        language="en",
        published_at=_at(0),
        received_at=_at(0),
        publication_timestamp_quality=quality,
    )


def _at(minutes: int) -> datetime:
    return datetime(2026, 7, 1, 7, 0, tzinfo=UTC) + timedelta(minutes=minutes)


def _benchmark_candle(benchmark_id: UUID, begin_at: datetime, close: str) -> BenchmarkCandle:
    price = Decimal(close)
    return BenchmarkCandle.create(
        benchmark_id=benchmark_id,
        interval_minutes=1,
        begin_at=begin_at,
        end_at=begin_at + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )


def _adjustment(status: BenchmarkAdjustmentStatus) -> ReactionBenchmarkAdjustment:
    available = status == BenchmarkAdjustmentStatus.AVAILABLE
    return ReactionBenchmarkAdjustment.create(
        reaction_point_id=UUID(int=0),
        benchmark_id=uuid4(),
        benchmark_code="IMOEX",
        baseline_value=Decimal("100") if available else None,
        target_value=Decimal("101") if available else None,
        baseline_observed_at=_at(0) if available else None,
        target_observed_at=_at(15) if available else None,
        simple_return=Decimal("0.01") if available else None,
        log_return=Decimal("0.009950330853168083") if available else None,
        abnormal_simple_return=Decimal("0.01") if available else None,
        abnormal_log_return=Decimal("0.00985229644301164") if available else None,
        status=status,
        missing_reason=None if available else "benchmark_target_candle_missing",
    )


def _point_with_adjustment() -> ReactionPoint:
    point = ReactionPoint.create(
        reaction_id=uuid4(),
        horizon_minutes=15,
        target_at=_at(15),
        observed_at=_at(15),
        price=Decimal("102"),
        simple_return=Decimal("0.02"),
        log_return=Decimal("0.01980262729617971"),
        status=ReactionPointStatus.AVAILABLE,
    )
    adjustment = _adjustment(BenchmarkAdjustmentStatus.AVAILABLE)
    return ReactionPoint(
        id=point.id,
        reaction_id=point.reaction_id,
        horizon_minutes=point.horizon_minutes,
        target_at=point.target_at,
        observed_at=point.observed_at,
        price=point.price,
        simple_return=point.simple_return,
        log_return=point.log_return,
        status=point.status,
        benchmark_adjustment=ReactionBenchmarkAdjustment(
            id=adjustment.id,
            reaction_point_id=point.id,
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
        ),
    )
