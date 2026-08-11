from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import BigInteger

from src.historical_news.domain.enums import ContentStoragePolicy
from src.market_data.infrastructure.models import MarketDataImportRecord
from src.ml_features.domain.entities import (
    FeatureDatasetBuildResult,
    FeatureDatasetConfig,
    FeatureDatasetRow,
    MlFeatureDatasetRun,
)
from src.ml_features.domain.enums import FeatureDatasetRunStatus
from src.news.domain.enums import PublicationTimestampQuality
from src.reaction_ready_corpus.application import PrepareCorpusCommand
from src.reaction_ready_corpus.domain import (
    REAL_SOURCE_CODES,
    UNIVERSE,
    CorpusProvenance,
    MarketDataStatus,
    MatchStatus,
    ReadinessStatus,
    SourceAuditEntry,
    SourceAuditStatus,
    classify_provenance,
    match_status,
    plan_market_windows,
    readiness_status,
    source_audit_payload,
    timestamp_exclusion,
)
from src.reaction_ready_corpus.reporting import (
    AcquisitionRunSummary,
    CandidateMatch,
    CandidateSnapshot,
    build_and_write_reports,
)
from src.reaction_ready_corpus.source_audit import audited_sources, write_source_audit
from src.reactions.infrastructure.models import NewsMarketReactionRecord

PUBLISHED_AT = datetime(2026, 6, 5, 15, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("source_code", "source_name", "expected"),
    [
        ("ROSNEFT_PRESS_RELEASES_RSS", "Rosneft", CorpusProvenance.REAL),
        ("SYNTHETIC_SMOKE", "synthetic", CorpusProvenance.SYNTHETIC),
        ("BATCH_001", "seed-dataset", CorpusProvenance.SEED),
        ("UNREVIEWED_SOURCE", None, CorpusProvenance.OTHER),
    ],
)
def test_provenance_is_explicit_and_unknown_is_never_real(
    source_code: str, source_name: str | None, expected: CorpusProvenance
) -> None:
    assert classify_provenance(source_code, source_name) == expected


def test_only_audited_source_code_is_real() -> None:
    assert REAL_SOURCE_CODES == {
        "ROSNEFT_PRESS_RELEASES_RSS",
        "YANDEX_IR_PRESS_RELEASES_RSS",
    }
    assert classify_provenance("TEST_ROSNEFT") != CorpusProvenance.REAL
    assert classify_provenance("SEED_ROSNEFT") != CorpusProvenance.REAL


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (PublicationTimestampQuality.EXACT, None),
        (PublicationTimestampQuality.DATE_ONLY, "DATE_ONLY"),
        (PublicationTimestampQuality.UNKNOWN, "UNKNOWN_TIMESTAMP"),
    ],
)
def test_only_exact_timestamp_is_eligible(
    quality: PublicationTimestampQuality, expected: str | None
) -> None:
    reason = timestamp_exclusion(quality)
    assert (None if reason is None else reason.value) == expected


@pytest.mark.parametrize(
    ("count", "ambiguous", "expected"),
    [
        (0, False, MatchStatus.UNMATCHED),
        (1, False, MatchStatus.MATCHED),
        (1, True, MatchStatus.AMBIGUOUS),
        (2, False, MatchStatus.AMBIGUOUS),
    ],
)
def test_match_status_preserves_ambiguity(
    count: int, ambiguous: bool, expected: MatchStatus
) -> None:
    assert match_status(count, ambiguous) == expected


def test_market_window_is_bounded_and_covers_weekend_safety() -> None:
    windows = plan_market_windows(
        [
            ("ROSN", PUBLISHED_AT),
            ("ROSN", PUBLISHED_AT + timedelta(hours=2)),
        ]
    )
    assert len(windows) == 1
    assert windows[0].ticker == "ROSN"
    assert windows[0].interval_minutes == 1
    assert windows[0].date_from == datetime(2026, 6, 2, tzinfo=UTC)
    assert windows[0].date_to.date().isoformat() == "2026-06-12"
    assert windows[0].date_to - windows[0].date_from < timedelta(days=11)


def test_market_window_filters_tickers_outside_universe() -> None:
    assert plan_market_windows([("NOT_IN_UNIVERSE", PUBLISHED_AT)]) == []


def test_benchmark_only_model_import_registers_instrument_foreign_key() -> None:
    foreign_key = next(iter(MarketDataImportRecord.__table__.c.instrument_id.foreign_keys))
    assert foreign_key.column.table.name == "instruments"


def test_historical_reaction_latency_uses_bigint() -> None:
    assert isinstance(
        NewsMarketReactionRecord.__table__.c.publication_to_receipt_ms.type,
        BigInteger,
    )
    assert isinstance(
        NewsMarketReactionRecord.__table__.c.publication_to_effective_event_ms.type,
        BigInteger,
    )


def test_prepare_config_forbids_unapproved_sources_and_unbounded_limit() -> None:
    with pytest.raises(ValueError, match="approved REAL"):
        PrepareCorpusCommand(PUBLISHED_AT, PUBLISHED_AT, ("UNKNOWN",)).normalized()
    with pytest.raises(ValueError, match="between 1 and 1000"):
        PrepareCorpusCommand(
            PUBLISHED_AT,
            PUBLISHED_AT,
            ("ROSNEFT_PRESS_RELEASES_RSS",),
            limit=1001,
        ).normalized()


def test_acquisition_selection_config_contains_no_future_label_fields() -> None:
    fields = set(PrepareCorpusCommand.__dataclass_fields__)
    assert fields == {"date_from", "date_to", "source_codes", "tickers", "limit", "dry_run"}
    assert not fields.intersection({"return", "abnormal_return", "future_volume", "event_type"})


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (0, ReadinessStatus.NOT_READY),
        (99, ReadinessStatus.NOT_READY),
        (100, ReadinessStatus.PILOT_ONLY),
        (499, ReadinessStatus.PILOT_ONLY),
        (500, ReadinessStatus.BASELINE_EXPERIMENT_READY),
        (999, ReadinessStatus.BASELINE_EXPERIMENT_READY),
        (1000, ReadinessStatus.BASELINE_TRAINING_READY),
    ],
)
def test_readiness_thresholds(rows: int, expected: ReadinessStatus) -> None:
    assert readiness_status(rows) == expected


def test_source_audit_schema_covers_universe_and_validates(tmp_path: Path) -> None:
    payload = source_audit_payload(audited_sources())
    assert set(payload["universe"]) == set(UNIVERSE)
    compliant = [item for item in payload["sources"] if item["status"] == "COMPLIANT"]
    assert {item["source_code"] for item in compliant} == {
        "ROSNEFT_PRESS_RELEASES_RSS",
        "YANDEX_IR_PRESS_RELEASES_RSS",
    }
    output = tmp_path / "source-audit.json"
    write_source_audit(output)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"].endswith("v1")


def test_source_audit_rejects_non_https_compliant_source() -> None:
    entry = SourceAuditEntry(
        tickers=("ROSN",),
        issuer="Issuer",
        source_url="http://example.com/rss",
        source_owner="Issuer",
        source_kind="ISSUER_RSS",
        https=False,
        historical_depth_observed="one page",
        timestamp_precision="seconds",
        timezone_semantics="offset",
        full_text_availability="excerpt",
        storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
        pagination_archive_capability="none",
        robots_access_restrictions="public",
        status=SourceAuditStatus.COMPLIANT,
        source_code="ROSNEFT_PRESS_RELEASES_RSS",
    )
    with pytest.raises(ValueError, match="HTTPS"):
        entry.validate()


def test_manifest_and_corpus_exclude_synthetic_and_seed_rows(tmp_path: Path) -> None:
    real_news_id = uuid4()
    snapshots = [
        _snapshot(real_news_id, "ROSNEFT_PRESS_RELEASES_RSS"),
        _snapshot(uuid4(), "SYNTHETIC_SMOKE"),
        _snapshot(uuid4(), "BATCH_001", source_name="seed-dataset"),
    ]
    rows = [
        _feature_row(real_news_id, "ROSNEFT_PRESS_RELEASES_RSS"),
        _feature_row(uuid4(), "SYNTHETIC_SMOKE"),
        _feature_row(uuid4(), "BATCH_001"),
    ]
    result = _feature_result(rows)
    paths = build_and_write_reports(
        tmp_path,
        snapshots=snapshots,
        feature_result=result,
        acquisition=AcquisitionRunSummary(1, 1, 1, 0, 0),
        date_from=PUBLISHED_AT,
        date_to=PUBLISHED_AT,
        git_sha="abc123",
        batch_reactions=0,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))
    corpus = paths["corpus"].read_text(encoding="utf-8").splitlines()
    assert manifest["real_reaction_ready_rows"] == 1
    assert manifest["synthetic_rows_counted_as_real"] == 0
    assert manifest["provenance_feature_rows"] == {"REAL": 1, "SEED": 1, "SYNTHETIC": 1}
    assert manifest["readiness_status"] == "NOT_READY"
    assert manifest["diversity_warnings"] == ["LOW_DIVERSITY"]
    assert coverage["label_availability"] == {
        "1m": 1,
        "5m": 1,
        "15m": 1,
        "30m": 1,
        "60m": 1,
    }
    assert len(corpus) == 1
    payload = json.loads(corpus[0])
    assert payload["metadata"]["provenance"] == "REAL"
    assert "features_available_at_publication" in payload


def test_funnel_and_exclusion_counts_cover_candidate_failures(tmp_path: Path) -> None:
    snapshots = [
        _snapshot(
            uuid4(), "ROSNEFT_PRESS_RELEASES_RSS", quality=PublicationTimestampQuality.DATE_ONLY
        ),
        _snapshot(
            uuid4(), "ROSNEFT_PRESS_RELEASES_RSS", quality=PublicationTimestampQuality.UNKNOWN
        ),
        _snapshot(uuid4(), "ROSNEFT_PRESS_RELEASES_RSS", matches=()),
        _snapshot(
            uuid4(),
            "ROSNEFT_PRESS_RELEASES_RSS",
            matches=(CandidateMatch(uuid4(), "ROSN", True),),
        ),
        _snapshot(uuid4(), "ROSNEFT_PRESS_RELEASES_RSS", event_type=None),
        _snapshot(
            uuid4(),
            "ROSNEFT_PRESS_RELEASES_RSS",
            market=MarketDataStatus.MARKET_DATA_MISSING_SECURITY,
            horizons=(),
        ),
        _snapshot(
            uuid4(),
            "ROSNEFT_PRESS_RELEASES_RSS",
            market=MarketDataStatus.MARKET_DATA_MISSING_BENCHMARK,
            horizons=(),
        ),
        _snapshot(
            uuid4(),
            "ROSNEFT_PRESS_RELEASES_RSS",
            market=MarketDataStatus.OTHER,
            horizons=(),
        ),
    ]
    paths = build_and_write_reports(
        tmp_path,
        snapshots=snapshots,
        feature_result=_feature_result([]),
        acquisition=AcquisitionRunSummary(10, 9, 8, 1, 1),
        date_from=PUBLISHED_AT,
        date_to=PUBLISHED_AT,
        git_sha="abc123",
        batch_reactions=0,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["exclusions_by_reason"] == {
        "AMBIGUOUS": 1,
        "DATE_ONLY": 1,
        "IMOEX_DATA_MISSING": 1,
        "NO_EVENT_ANALYSIS": 1,
        "NO_VALID_REACTION": 1,
        "SECURITY_MARKET_DATA_MISSING": 1,
        "UNKNOWN_TIMESTAMP": 1,
        "UNMATCHED": 1,
    }
    assert [stage["count"] for stage in manifest["funnel"]] == [10, 9, 8, 6, 4, 1, 1, 0]
    assert manifest["diversity_warnings"] == ["NO_REAL_FEATURE_ROWS", "LOW_DIVERSITY"]


def _snapshot(
    news_id: UUID,
    source_code: str,
    *,
    source_name: str | None = "issuer-feed",
    quality: PublicationTimestampQuality = PublicationTimestampQuality.EXACT,
    matches: tuple[CandidateMatch, ...] | None = None,
    event_type: str | None = "OTHER",
    market: MarketDataStatus = MarketDataStatus.MARKET_DATA_READY,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=uuid4(),
        news_id=news_id,
        source_code=source_code,
        source_name=source_name,
        source_item_id=str(uuid4()),
        source_url="https://example.com/item",
        published_at=PUBLISHED_AT,
        timestamp_quality=quality,
        storage_policy="EXCERPT_ALLOWED",
        imported=True,
        is_correction=False,
        matches=(CandidateMatch(uuid4(), "ROSN", False),) if matches is None else matches,
        event_type=event_type,
        market_data_status=market,
        valid_label_horizons=horizons,
    )


def _feature_row(news_id: UUID, source: str) -> FeatureDatasetRow:
    labels = {
        f"{horizon}m": {"available": True, "abnormal_simple_return": "0.01"}
        for horizon in (1, 5, 15, 30, 60)
    }
    return FeatureDatasetRow(
        metadata={
            "news_id": news_id,
            "ticker": "ROSN",
            "published_at": PUBLISHED_AT,
            "source": source,
            "source_item_id": str(news_id),
            "dataset_version": "ml-feature-dataset-v1",
            "feature_version": "ml-features-v1",
            "event_analysis_version": "event-rules-v2",
            "fact_extractor_version": "financial-facts-v2",
            "reaction_version": "reaction-v2-benchmark-adjusted",
        },
        features={"primary_event_type": "OTHER", "event_count": 1, "fact_count": 0},
        labels=labels,
        quality={},
    )


def _feature_result(rows: list[FeatureDatasetRow]) -> FeatureDatasetBuildResult:
    config = FeatureDatasetConfig(PUBLISHED_AT, PUBLISHED_AT)
    run = MlFeatureDatasetRun.start(config, git_sha="abc123").finish(
        status=FeatureDatasetRunStatus.SUCCEEDED,
        candidate_count=len(rows),
        eligible_count=len(rows),
        built_count=len(rows),
        excluded_count=0,
        failed_count=0,
    )
    return FeatureDatasetBuildResult(rows=rows, exclusions=[], run=run)
