from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from src.corpus_quality.domain import (
    ANNOTATION_BATCH_VERSION,
    DIAGNOSIS_INPUT_FIELDS,
    FROZEN_BATCH_001_GOLD_SHA256,
    ExpansionSelection,
    PublicationTimeRecord,
    ShadowPrediction,
    SourceAcceptanceEvidence,
    UnknownDiagnosticCategory,
    build_baseline,
    cumulative_funnel,
    diagnose_unknown,
    diversity_warnings,
    readiness_report,
    required_market_windows,
    rules_vs_shadow,
    select_annotation_batch,
)
from src.news.domain.enums import PublicationTimestampQuality

PUBLISHED = datetime(2026, 6, 5, 13, 50, tzinfo=UTC)


def test_unknown_diagnostic_category_schema_is_frozen() -> None:
    assert {item.value for item in UnknownDiagnosticCategory} == {
        "TRUE_NO_SUPPORTED_EVENT",
        "CONTENT_TOO_THIN",
        "SOURCE_PARSE_OR_TRUNCATION",
        "RULE_MISS_CANDIDATE",
        "UNCERTAIN",
    }


def test_baseline_snapshot_requires_exactly_ten_rosn_rows() -> None:
    baseline = build_baseline([_record(index) for index in range(10)])
    assert baseline == {
        "schema_version": "corpus-quality-expansion-v1",
        "count": 10,
        "ticker": "ROSN",
        "EXACT": 10,
        "matched": 10,
        "reaction_ready": 10,
        "feature_ready": 10,
        "deterministic_primary_UNKNOWN": 10,
        "news_ids": [str(_news_id(index)) for index in range(10)],
        "uses_post_event_data_for_diagnosis": False,
    }


def test_unknown_diagnosis_has_no_post_event_inputs() -> None:
    assert DIAGNOSIS_INPUT_FIELDS == {
        "news_id",
        "ticker",
        "source_code",
        "source_item_id",
        "source_url",
        "title",
        "content",
        "published_at",
        "timestamp_quality",
        "storage_policy",
        "content_is_excerpt",
        "rules_primary_event",
        "rules_event_count",
        "rules_fact_count",
        "analysis_status",
        "analysis_warnings",
    }
    assert not DIAGNOSIS_INPUT_FIELDS.intersection(
        {"return", "abnormal_return", "future_volume", "reaction", "labels"}
    )


def test_clear_non_material_cooperation_is_true_no_supported_event() -> None:
    item = _record(
        1, content="Rosneft signed an agreement on personnel training with a university."
    )
    assert diagnose_unknown(item).category == UnknownDiagnosticCategory.TRUE_NO_SUPPORTED_EVENT


def test_short_strategic_excerpt_is_content_too_thin() -> None:
    item = _record(1, content="Rosneft and KAMAZ expand strategic cooperation.")
    assert diagnose_unknown(item).category == UnknownDiagnosticCategory.CONTENT_TOO_THIN


def test_explicit_truncation_is_source_parse_or_truncation() -> None:
    item = _record(1, content="A sufficiently long issuer description that ends unexpectedly...")
    assert diagnose_unknown(item).category == UnknownDiagnosticCategory.SOURCE_PARSE_OR_TRUNCATION


def test_supported_signal_is_rule_miss_candidate() -> None:
    item = _record(
        1,
        content="The company published financial results and net profit for the reporting year.",
        excerpt=False,
    )
    assert diagnose_unknown(item).category == UnknownDiagnosticCategory.RULE_MISS_CANDIDATE


def test_shadow_comparison_does_not_modify_deterministic_record() -> None:
    record = _record(1)
    before = record.rules_primary_event, record.rules_event_count, record.rules_fact_count
    rows = rules_vs_shadow(
        [record], [ShadowPrediction(record.news_id, "MAJOR_CONTRACT", 1, 2, True)]
    )
    assert rows[0]["event_agreement"] is False
    assert before == (
        record.rules_primary_event,
        record.rules_event_count,
        record.rules_fact_count,
    )


def test_shadow_prediction_cannot_carry_ml_features() -> None:
    assert {item.name for item in fields(ShadowPrediction)} == {
        "news_id",
        "primary_event",
        "event_count",
        "fact_count",
        "successful",
    }


def test_rules_ai_disagreement_report_contains_no_winner_or_reconciliation() -> None:
    record = _record(1)
    row = rules_vs_shadow([record], [ShadowPrediction(record.news_id, "OTHER", 1, 0, True)])[0]
    assert set(row) == {
        "news_id",
        "ticker",
        "rules_primary_event",
        "rules_event_count",
        "rules_fact_count",
        "ai_primary_event",
        "ai_event_count",
        "ai_fact_count",
        "event_agreement",
    }


def test_source_expansion_and_market_windows_are_bounded() -> None:
    selection = ExpansionSelection(
        "ROSNEFT_PRESS_RELEASES_RSS",
        PUBLISHED - timedelta(days=1),
        PUBLISHED + timedelta(days=1),
        10,
        ("ROSN",),
    )
    selection.validate()
    windows = required_market_windows([_record(1)])
    assert len(windows) == 1
    assert windows[0].date_to - windows[0].date_from < timedelta(days=3)
    with pytest.raises(ValueError, match="between 1 and 100"):
        ExpansionSelection(
            selection.source_code,
            selection.date_from,
            selection.date_to,
            101,
            selection.tickers,
        ).validate()


def test_source_acceptance_requires_timestamp_and_timezone_semantics() -> None:
    evidence = SourceAcceptanceEvidence(
        "AUDIT_TEST",
        ("ROSN",),
        "https://example.com/rss",
        "Issuer",
        "Issuer",
        "Date only; timezone unavailable",
        "EXCERPT_ALLOWED",
        issuer_owned=True,
        exact_publication_timestamp=False,
        timezone_semantics_confirmed=False,
        stable_item_identity=True,
        storage_policy_confirmed=True,
        https=True,
        bounded_acquisition=True,
        blocker="Exact time contract is unavailable",
    )
    evidence.validate()
    assert evidence.compliant is False


def test_date_only_cannot_be_reaction_ready() -> None:
    with pytest.raises(ValueError, match="non-EXACT"):
        _record(1, quality=PublicationTimestampQuality.DATE_ONLY)


def test_source_selection_config_has_no_return_filters() -> None:
    assert {item.name for item in fields(ExpansionSelection)} == {
        "source_code",
        "date_from",
        "date_to",
        "limit",
        "tickers",
    }


def test_cumulative_funnel_includes_event_analysis() -> None:
    funnel = cumulative_funnel([_record(index) for index in range(3)])
    assert [item["stage"] for item in funnel] == [
        "discovered",
        "validated",
        "imported",
        "EXACT",
        "matched",
        "event-analyzed",
        "market-data-ready",
        "reaction-ready",
        "feature-ready",
    ]
    assert all(item["count"] == 3 for item in funnel)


def test_unknown_rate_warning_threshold_is_strictly_over_half() -> None:
    assert "HIGH_UNKNOWN_EVENT_RATE" in diversity_warnings({"ROSN": 10}, {"UNKNOWN": 6, "OTHER": 4})
    assert "HIGH_UNKNOWN_EVENT_RATE" not in diversity_warnings(
        {"ROSN": 10}, {"UNKNOWN": 5, "OTHER": 5}
    )


def test_ticker_diversity_warning_threshold_is_over_seventy_percent() -> None:
    warnings = diversity_warnings({"ROSN": 8, "SBER": 2}, {"UNKNOWN": 5, "OTHER": 5})
    assert "LOW_TICKER_DIVERSITY" in warnings


def test_event_diversity_warning_threshold_is_over_seventy_percent() -> None:
    warnings = diversity_warnings({"ROSN": 5, "SBER": 5}, {"UNKNOWN": 2, "FINANCIAL_RESULTS": 8})
    assert "LOW_EVENT_DIVERSITY" in warnings


def test_annotation_batch_selection_is_deterministic_and_return_independent() -> None:
    records = [_record(2), _record(0), _record(1)]
    first = select_annotation_batch(records)
    second = select_annotation_batch(list(reversed(records)))
    assert first == second
    assert all("return" not in key and "label" not in key for key in first[0])


def test_annotation_batch_remains_draft_unassigned_and_not_gold() -> None:
    rows = select_annotation_batch([_record(1)])
    assert rows[0]["schema_version"] == ANNOTATION_BATCH_VERSION
    assert rows[0]["annotation_status"] == "DRAFT"
    assert rows[0]["assignment_status"] == "UNASSIGNED"
    assert rows[0]["is_gold"] is False


def test_frozen_batch_001_gold_checksum_is_unchanged() -> None:
    assert FROZEN_BATCH_001_GOLD_SHA256 == (
        "4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196"
    )


def test_idempotent_live_selection_rerun() -> None:
    records = [_record(index) for index in range(10)]
    assert select_annotation_batch(records) == select_annotation_batch(records)
    assert build_baseline(records) == build_baseline(records)


def test_unit_quality_module_has_no_live_http_client() -> None:
    source_dir = Path(__file__).parents[2] / "src" / "corpus_quality"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_dir.glob("*.py"))
    assert "import httpx" not in source
    assert "requests.get" not in source


def test_readiness_blocks_model_for_high_unknown_rate_and_low_ticker_count() -> None:
    report = readiness_report(
        reaction_rows=1000,
        annotation_rows=20,
        tickers=1,
        unknown_rate=1.0,
    )
    assert report["REACTION_DATA_READINESS"] == "BASELINE_TRAINING_READY"
    assert report["EVENT_FEATURE_QUALITY"] == "EVENT_FEATURE_QUALITY_BLOCKER"
    assert report["MODEL_TRAINING_READINESS"] == "NOT_READY"


def _record(
    index: int,
    *,
    content: str = (
        "Rosneft and the regional government signed a cooperation memorandum to promote "
        "domestic tourism and personnel education."
    ),
    excerpt: bool = True,
    quality: PublicationTimestampQuality = PublicationTimestampQuality.EXACT,
) -> PublicationTimeRecord:
    return PublicationTimeRecord(
        news_id=_news_id(index),
        ticker="ROSN",
        source_code="ROSNEFT_PRESS_RELEASES_RSS",
        source_item_id=f"item-{index:02d}",
        source_url=f"https://www.rosneft.com/press/releases/item/{index}/",
        title=f"Rosneft release {index}",
        content=content,
        published_at=PUBLISHED + timedelta(minutes=index),
        timestamp_quality=quality,
        storage_policy="EXCERPT_ALLOWED",
        content_is_excerpt=excerpt,
        rules_primary_event="UNKNOWN",
        rules_event_count=0,
        rules_fact_count=0,
        analysis_status="NO_EVENT_FOUND",
        analysis_warnings=(),
        matched=True,
        market_data_ready=True,
        reaction_ready=True,
        feature_ready=True,
        valid_label_horizons=(1, 5, 15, 30, 60),
    )


def _news_id(index: int) -> UUID:
    return UUID(int=index + 1)
