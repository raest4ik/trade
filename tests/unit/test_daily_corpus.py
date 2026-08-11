from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from src.daily_corpus.application import DailyCorpusBuildResult, source_calendar_date
from src.daily_corpus.domain import (
    FEATURE_VERSION,
    INTRADAY_REACTION_VERSION,
    LABEL_FAMILY,
    MAX_HISTORICAL_IMPORT,
    REACTION_VERSION,
    DailyCandidate,
    DailyExclusionReason,
    SessionClose,
    SourceAcceptanceStatus,
    TemporalSplit,
    build_daily_feature_row,
    build_daily_reaction,
    collapse_complete_session_closes,
    daily_eligibility,
    daily_readiness,
    deterministic_temporal_split,
    select_historical_import,
)
from src.daily_corpus.reporting import write_daily_corpus_reports
from src.daily_corpus.source_registry import daily_source_verifications
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.market_data.domain.entities import MarketCandle
from src.news.domain.enums import PublicationTimestampQuality
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_THINK

NEWS_ID = UUID(int=101)
INSTRUMENT_ID = UUID(int=202)
PUBLICATION_DATE = date(2026, 8, 11)


def test_date_only_is_eligible_but_never_becomes_exact() -> None:
    candidate = _candidate(PublicationTimestampQuality.DATE_ONLY)
    assert daily_eligibility(candidate) is None
    reaction = _reaction(candidate)
    assert reaction.timestamp_quality == PublicationTimestampQuality.DATE_ONLY
    assert reaction.label_family == LABEL_FAMILY


def test_crawl_date_never_substitutes_source_date() -> None:
    candidate = replace(_candidate(), publication_date_from_source=False)
    assert daily_eligibility(candidate) == DailyExclusionReason.CRAWL_DATE_SUBSTITUTION


def test_fixed_source_offset_preserves_source_calendar_date() -> None:
    published_at = datetime(2026, 8, 10, 21, 30, tzinfo=UTC)
    assert source_calendar_date(published_at, "UTC+03:00") == date(2026, 8, 11)


def test_missing_source_date_is_excluded() -> None:
    candidate = replace(_candidate(), publication_date=None)
    assert daily_eligibility(candidate) == DailyExclusionReason.NO_SOURCE_PUBLICATION_DATE


def test_unknown_timestamp_is_excluded() -> None:
    candidate = _candidate(PublicationTimestampQuality.UNKNOWN)
    assert daily_eligibility(candidate) == DailyExclusionReason.TIMESTAMP_UNKNOWN


def test_baseline_and_target_are_strictly_outside_publication_date() -> None:
    reaction = _reaction(_candidate())
    assert reaction.baseline_session_date < PUBLICATION_DATE
    assert reaction.target_session_date > PUBLICATION_DATE


def test_same_day_candles_are_never_used() -> None:
    candidate = _candidate()
    security = [_close(PUBLICATION_DATE, "105"), *_security_closes()]
    reaction, reason = build_daily_reaction(
        candidate, security_closes=security, benchmark_closes=_benchmark_closes()
    )
    assert reason is None
    assert reaction is not None
    assert reaction.baseline_security_close == Decimal("100")
    assert reaction.target_security_close == Decimal("110")


def test_weekend_uses_friday_and_monday() -> None:
    saturday = replace(_candidate(), publication_date=date(2026, 8, 15))
    security = [_close(date(2026, 8, 14), "100"), _close(date(2026, 8, 17), "101")]
    benchmark = [_close(date(2026, 8, 14), "1000"), _close(date(2026, 8, 17), "1002")]
    reaction, _ = build_daily_reaction(
        saturday, security_closes=security, benchmark_closes=benchmark
    )
    assert reaction is not None
    assert reaction.baseline_session_date == date(2026, 8, 14)
    assert reaction.target_session_date == date(2026, 8, 17)


def test_holiday_gap_uses_nearest_common_sessions() -> None:
    candidate = replace(_candidate(), publication_date=date(2026, 1, 7))
    security = [_close(date(2025, 12, 30), "100"), _close(date(2026, 1, 12), "103")]
    benchmark = [_close(date(2025, 12, 30), "1000"), _close(date(2026, 1, 12), "1010")]
    reaction, _ = build_daily_reaction(
        candidate, security_closes=security, benchmark_closes=benchmark
    )
    assert reaction is not None
    assert reaction.target_session_date == date(2026, 1, 12)


def test_missing_market_day_is_explicit() -> None:
    reaction, reason = build_daily_reaction(
        _candidate(), security_closes=_security_closes()[:1], benchmark_closes=_benchmark_closes()
    )
    assert reaction is None
    assert reason == DailyExclusionReason.COMMON_SESSION_WINDOW_MISSING


def test_missing_security_and_benchmark_are_distinct() -> None:
    _, security_reason = build_daily_reaction(
        _candidate(), security_closes=[], benchmark_closes=_benchmark_closes()
    )
    _, benchmark_reason = build_daily_reaction(
        _candidate(), security_closes=_security_closes(), benchmark_closes=[]
    )
    assert security_reason == DailyExclusionReason.SECURITY_MARKET_DATA_MISSING
    assert benchmark_reason == DailyExclusionReason.BENCHMARK_MARKET_DATA_MISSING


def test_imoex_uses_exact_same_session_dates() -> None:
    benchmark = [_close(date(2026, 8, 8), "990"), *_benchmark_closes()]
    reaction, _ = build_daily_reaction(
        _candidate(), security_closes=_security_closes(), benchmark_closes=benchmark
    )
    assert reaction is not None
    assert reaction.baseline_session_date == date(2026, 8, 10)
    assert reaction.target_session_date == date(2026, 8, 12)


def test_abnormal_return_formula() -> None:
    reaction = _reaction(_candidate())
    assert reaction.security_return == Decimal("0.1")
    assert reaction.benchmark_return == Decimal("0.02")
    assert reaction.abnormal_return == Decimal("0.08")


def test_daily_and_intraday_versions_are_distinct() -> None:
    assert REACTION_VERSION != INTRADAY_REACTION_VERSION
    assert FEATURE_VERSION != "ml-features-v1"


def test_exact_record_can_have_daily_label_without_replacing_intraday() -> None:
    reaction = _reaction(_candidate(PublicationTimestampQuality.EXACT))
    assert reaction.reaction_version == REACTION_VERSION
    assert reaction.timestamp_quality == PublicationTimestampQuality.EXACT


@pytest.mark.parametrize(
    ("match_count", "ambiguous"),
    [
        (2, False),
        (1, True),
    ],
)
def test_ambiguous_instrument_is_excluded(match_count: int, ambiguous: bool) -> None:
    candidate = replace(
        _candidate(),
        match_count=match_count,
        ambiguous_match=ambiguous,
    )
    assert daily_eligibility(candidate) == DailyExclusionReason.AMBIGUOUS_INSTRUMENT


def test_no_match_is_excluded() -> None:
    candidate = replace(_candidate(), match_count=0, instrument_id=None, ticker=None)
    assert daily_eligibility(candidate) == DailyExclusionReason.NO_INSTRUMENT_MATCH


def test_session_collapse_rejects_partial_day_and_keeps_last_complete_candle() -> None:
    partial = _candle(datetime(2026, 8, 10, 10, tzinfo=UTC), "99")
    close = _candle(datetime(2026, 8, 10, 16, tzinfo=UTC), "100")
    later = _candle(datetime(2026, 8, 10, 17, tzinfo=UTC), "101")
    sessions = collapse_complete_session_closes([partial, close, later])
    assert len(sessions) == 1
    assert sessions[0].close == Decimal("101")


def test_daily_features_use_only_baseline_information() -> None:
    reaction = _reaction(_candidate())
    row = build_daily_feature_row(
        reaction,
        baseline_security=_security_closes()[0],
        baseline_benchmark=_benchmark_closes()[0],
    )
    assert row.feature_available_at.date() == reaction.baseline_session_date
    assert "abnormal_return" not in row.features
    assert row.labels["abnormal_return"] == Decimal("0.08")


def test_feature_builder_rejects_same_day_availability() -> None:
    reaction = _reaction(_candidate())
    same_day = _close(PUBLICATION_DATE, "100")
    with pytest.raises(ValueError, match="session does not match"):
        build_daily_feature_row(
            reaction,
            baseline_security=same_day,
            baseline_benchmark=_benchmark_closes()[0],
        )


def test_import_selection_is_deterministic_deduplicated_and_bounded() -> None:
    candidates = [
        replace(_candidate(), news_id=UUID(int=index + 1), source_item_id=f"item-{index:03d}")
        for index in range(MAX_HISTORICAL_IMPORT + 20)
    ]
    first = select_historical_import(list(reversed(candidates)))
    second = select_historical_import(candidates)
    assert [item.news_id for item in first] == [item.news_id for item in second]
    assert len(first) == MAX_HISTORICAL_IMPORT


def test_import_selection_ignores_model_and_return_fields_by_construction() -> None:
    assert set(DailyCandidate.__dataclass_fields__).isdisjoint(
        {"rules_prediction", "qwen_prediction", "abnormal_return", "future_volume"}
    )


def test_temporal_split_is_deterministic_and_monotonic() -> None:
    rows = [
        replace(
            _feature_row(),
            news_id=UUID(int=index + 1),
            publication_date=PUBLICATION_DATE + timedelta(days=index),
        )
        for index in range(10)
    ]
    assignments = deterministic_temporal_split(list(reversed(rows)))
    ordered = [assignments[row.news_id] for row in rows]
    assert ordered == [
        *([TemporalSplit.TRAIN] * 7),
        TemporalSplit.VALIDATION,
        *([TemporalSplit.TEST] * 2),
    ]


def test_readiness_thresholds_and_diversity_downgrade() -> None:
    assert (
        daily_readiness(99, ticker_count=3, source_count=2, month_count=6)["status"] == "NOT_READY"
    )
    assert (
        daily_readiness(100, ticker_count=3, source_count=2, month_count=6)["status"]
        == "DAILY_PILOT_READY"
    )
    assert (
        daily_readiness(500, ticker_count=3, source_count=2, month_count=6)["status"]
        == "DAILY_BASELINE_EXPERIMENT_READY"
    )
    assert (
        daily_readiness(1000, ticker_count=1, source_count=1, month_count=1)["status"]
        == "DAILY_PILOT_READY"
    )


def test_source_verification_is_bounded_and_zero_cost() -> None:
    records = daily_source_verifications()
    assert len(records) == 8
    assert all(item.sample_limit <= 20 and item.sampled_items <= 20 for item in records)
    assert all(item.free for item in records)
    assert not any(item.status == SourceAcceptanceStatus.REJECTED_PAID for item in records)


def test_no_archive_is_accepted_without_automation_and_storage_policy() -> None:
    records = daily_source_verifications()
    assert not any(
        item.status == SourceAcceptanceStatus.COMPLIANT_DATE_SAFE_DAILY for item in records
    )
    assert all(item.blockers for item in records)


def test_reports_separate_verified_and_estimated_volume(tmp_path: Path) -> None:
    result = DailyCorpusBuildResult(
        candidates=[_candidate()],
        eligible=[_candidate()],
        reactions=[_reaction(_candidate())],
        features=[_feature_row()],
        exclusions={},
    )
    paths = write_daily_corpus_reports(
        tmp_path,
        result=result,
        verifications=daily_source_verifications(),
        intraday={"real_exact": 40, "reaction_ready": 26, "feature_ready": 21},
    )
    verification = json.loads(paths["source_verification"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["import_manifest"].read_text(encoding="utf-8"))
    assert verification["estimated_items"] == 9156
    assert verification["verified_accessible_items"] == 100
    assert verification["verified_daily_eligible_items"] == 0
    assert manifest["new_real"] == 0
    assert manifest["selection_uses_market_returns"] is False


def test_reports_keep_label_and_feature_namespaces_separate(tmp_path: Path) -> None:
    result = DailyCorpusBuildResult(
        candidates=[_candidate()],
        eligible=[_candidate()],
        reactions=[_reaction(_candidate())],
        features=[_feature_row()],
        exclusions={},
    )
    paths = write_daily_corpus_reports(
        tmp_path,
        result=result,
        verifications=daily_source_verifications(),
        intraday={"real_exact": 40, "reaction_ready": 26, "feature_ready": 21},
    )
    coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))
    assert coverage["label_family"] == LABEL_FAMILY
    assert coverage["reaction_version"] == REACTION_VERSION
    assert coverage["feature_version"] == FEATURE_VERSION
    assert coverage["daily_reaction_per_ticker"] == {"ROSN": 1}
    assert coverage["daily_feature_per_source"] == {"ROSNEFT_PRESS_RELEASES_RSS": 1}


def test_frozen_nlp_components_are_unchanged() -> None:
    assert (
        EXPECTED_RULES_FINGERPRINT
        == "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
    )
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    assert DEFAULT_OLLAMA_THINK is False


def test_daily_corpus_unit_tests_use_no_live_http() -> None:
    source_root = Path(__file__).parents[2] / "src" / "daily_corpus"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(source_root.glob("*.py"))
    )
    assert "httpx" not in source
    assert "requests." not in source


def _candidate(
    quality: PublicationTimestampQuality = PublicationTimestampQuality.EXACT,
) -> DailyCandidate:
    return DailyCandidate(
        news_id=NEWS_ID,
        source_code="ROSNEFT_PRESS_RELEASES_RSS",
        source_item_id="item-1",
        source_url="https://issuer.invalid/item-1",
        ticker="ROSN",
        instrument_id=INSTRUMENT_ID,
        publication_date=PUBLICATION_DATE,
        timestamp_quality=quality,
        publication_date_from_source=True,
        provenance="REAL",
        source_compliant=True,
        duplicate=False,
        match_count=1,
        ambiguous_match=False,
        text_length=100,
    )


def _close(session_date: date, close: str) -> SessionClose:
    return SessionClose(
        session_date=session_date,
        observed_at=datetime.combine(session_date, datetime.min.time(), UTC) + timedelta(hours=20),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def _security_closes() -> list[SessionClose]:
    return [_close(date(2026, 8, 10), "100"), _close(date(2026, 8, 12), "110")]


def _benchmark_closes() -> list[SessionClose]:
    return [_close(date(2026, 8, 10), "1000"), _close(date(2026, 8, 12), "1020")]


def _reaction(candidate: DailyCandidate):
    reaction, reason = build_daily_reaction(
        candidate,
        security_closes=_security_closes(),
        benchmark_closes=_benchmark_closes(),
    )
    assert reason is None
    assert reaction is not None
    return reaction


def _feature_row():
    return build_daily_feature_row(
        _reaction(_candidate()),
        baseline_security=_security_closes()[0],
        baseline_benchmark=_benchmark_closes()[0],
    )


def _candle(end_at: datetime, close: str) -> MarketCandle:
    return MarketCandle.create(
        instrument_id=INSTRUMENT_ID,
        board="TQBR",
        ticker_snapshot="ROSN",
        interval_minutes=1,
        begin_at=end_at - timedelta(minutes=1),
        end_at=end_at,
        open_price=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("1"),
        value=Decimal(close),
    )
