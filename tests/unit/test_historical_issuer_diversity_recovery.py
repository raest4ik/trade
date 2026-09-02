from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from apps.cli.audit_historical_issuer_diversity_recovery import build_parser
from src.exact_dataset_readiness_audit.ml_v2 import EXPECTED_INPUT_ARTIFACT_SHA
from src.issuer_historical_diversity_recovery.application import (
    run_historical_issuer_diversity_recovery_audit,
    validate_source_candidate,
)
from src.issuer_historical_diversity_recovery.domain import (
    ARTIFACT_VERSION,
    RecoveryDecision,
    SourceOptionStatus,
)


def test_cli_defaults_to_recovery_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"


def test_future_cutoff_enforced() -> None:
    decision = validate_source_candidate(
        {
            "published_at": "2026-08-11T00:00:00+03:00",
            "timestamp_field": "datePublished",
            "timezone_evidence": "UTC+03:00 in publication payload",
            "ticker": "AAA",
            "ticker_attribution": "DETERMINISTIC",
            "publication_material_available": True,
        }
    )

    assert decision["accepted"] is False
    assert decision["rejection_reason"] == "FUTURE_OR_MISSING_PUBLICATION_REJECTED"


def test_no_synthetic_timezone() -> None:
    decision = validate_source_candidate(
        {
            "published_at": "2026-08-10T12:00:00+03:00",
            "timestamp_field": "datePublished",
            "timezone_evidence": "MSK guessed by parser",
            "synthetic_timezone": True,
            "ticker": "AAA",
            "ticker_attribution": "DETERMINISTIC",
            "publication_material_available": True,
        }
    )

    assert decision["rejection_reason"] == "SYNTHETIC_TIMEZONE_REJECTED"


def test_date_modified_rejected() -> None:
    decision = validate_source_candidate(
        {
            "published_at": "2026-08-10T12:00:00+03:00",
            "timestamp_field": "dateModified",
            "date_modified_used": True,
            "timezone_evidence": "UTC+03:00 in publication payload",
            "ticker": "AAA",
            "ticker_attribution": "DETERMINISTIC",
            "publication_material_available": True,
        }
    )

    assert decision["rejection_reason"] == "DATE_MODIFIED_REJECTED"


def test_publication_specific_timezone_accepted() -> None:
    decision = validate_source_candidate(
        {
            "published_at": "2026-08-10T12:00:00+03:00",
            "timestamp_field": "datePublished",
            "timezone_evidence": "UTC+03:00 field in publication JSON",
            "ticker": "AAA",
            "ticker_attribution": "DETERMINISTIC",
            "publication_material_available": True,
        }
    )

    assert decision == {
        "accepted": True,
        "status": SourceOptionStatus.STRICT_EXACT_HISTORICAL_CAPABLE.value,
    }


def test_ticker_attribution_ambiguity_rejected() -> None:
    decision = validate_source_candidate(
        {
            "published_at": "2026-08-10T12:00:00+03:00",
            "timestamp_field": "datePublished",
            "timezone_evidence": "UTC+03:00 field in publication JSON",
            "ticker": "AAA/BBB",
            "ticker_attribution": "AMBIGUOUS",
            "publication_material_available": True,
        }
    )

    assert decision["rejection_reason"] == "TICKER_ATTRIBUTION_AMBIGUOUS"


def test_paid_source_metadata_without_purchasing(tmp_path: Path) -> None:
    manifest = _run(tmp_path)

    assert manifest["FINAL_DECISION"] == RecoveryDecision.PAID_OR_AUTHENTICATED_SOURCE_REQUIRED
    assert manifest["DATA_COST_RUB"] == 0
    assert manifest["PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND"] == 1


def test_source_selection_cannot_inspect_outcomes(tmp_path: Path) -> None:
    manifest = _run(tmp_path)

    assert manifest["SOURCE_SELECTION_USED_MARKET_OUTCOMES"] is False
    assert manifest["SOURCE_SELECTION_USED_MODEL_PERFORMANCE"] is False
    assert manifest["SOURCE_SELECTION_USED_FUTURE_OUTCOMES"] is False


def test_existing_exhausted_candidate_is_not_remined(tmp_path: Path) -> None:
    _run(tmp_path)
    registry = _read_jsonl(tmp_path / "out" / "source-options-registry.jsonl")
    exhausted = [
        row
        for row in registry
        if row["evidence_source"] == "timezone-verified-issuer-exact-source-discovery-v2"
    ]

    assert exhausted
    assert all(row["network_requests_performed"] == 0 for row in exhausted)
    assert all(row["no_reaudit_reason"] for row in exhausted)


def test_deterministic_source_registry_sha_replays(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first["SOURCE_OPTIONS_REGISTRY_SHA"] == second["SOURCE_OPTIONS_REGISTRY_SHA"]
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]


def test_historical_only_selection(tmp_path: Path) -> None:
    manifest = _run(tmp_path)

    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["CURRENT_ISSUER_ROWS"] == 2
    assert manifest["NEW_HISTORICAL_EVENTS_FOUND"] == 0


def test_future_outcome_counters_remain_zero(tmp_path: Path) -> None:
    manifest = _run(tmp_path)

    assert manifest["FUTURE_OUTCOMES_READ"] == 0
    assert manifest["FUTURE_TARGETS_READ"] == 0
    assert manifest["FUTURE_PRICE_LOOKUPS"] == 0


def test_null_event_features_are_treated_as_unknown(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    events_path = paths["backfill"] / "events.jsonl"
    rows = _read_jsonl(events_path)
    rows[0]["event_features"] = None
    _write_jsonl(events_path, rows)

    manifest = run_historical_issuer_diversity_recovery_audit(
        output_root=tmp_path / "out",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        backfill_root=paths["backfill"],
        readiness_root=paths["readiness"],
        tz_discovery_root=paths["tz"],
        issuer_diversity_root=paths["issuer_diversity"],
        consolidated_root=paths["consolidated"],
        chep_root=paths["chep"],
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        env_names=[],
    )

    assert manifest["NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50"] == 2


def _run(tmp_path: Path) -> dict[str, object]:
    paths = _write_artifacts(tmp_path)
    return run_historical_issuer_diversity_recovery_audit(
        output_root=tmp_path / "out",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        backfill_root=paths["backfill"],
        readiness_root=paths["readiness"],
        tz_discovery_root=paths["tz"],
        issuer_diversity_root=paths["issuer_diversity"],
        consolidated_root=paths["consolidated"],
        chep_root=paths["chep"],
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        env_names=["TINVEST_READONLY_TOKEN"],
    )


def _write_artifacts(tmp_path: Path) -> dict[str, Path]:
    backfill = tmp_path / "backfill"
    readiness = tmp_path / "readiness"
    tz = tmp_path / "tz"
    issuer_diversity = tmp_path / "issuer-diversity"
    consolidated = tmp_path / "consolidated"
    chep = tmp_path / "chep"
    for path in (backfill, readiness, tz, issuer_diversity, consolidated, chep):
        path.mkdir(parents=True)

    _write_json(
        readiness / "manifest.json",
        {
            "ARTIFACT_SHA": "readiness-sha",
            "CANONICAL_COHORT": "ISSUER_ORIGINATED_STRICT_EXACT_HISTORICAL_FEATURE_READY",
        },
    )
    _write_json(
        readiness / "canonical-gate-criteria.json",
        {
            "criteria": {
                "issuer_feature_ready_rows": {"threshold": 500},
                "unique_issuer_tickers": {"threshold": 10},
                "issuer_semantic_unknown_rate": {"threshold": "0.50"},
                "top_1_ticker_share": {"threshold": "0.50"},
                "primary_15m_target_coverage": {"threshold": "0.95"},
            }
        },
    )
    _write_json(backfill / "manifest.json", {"ARTIFACT_SHA": EXPECTED_INPUT_ARTIFACT_SHA})
    events = [
        _event("event-1", "MGNT", "MAGNIT_OFFICIAL_JSON_EXACT", "DEBT_FINANCING", 0),
        _event("event-2", "T", "TBANK_OFFICIAL_PUBLIC_NEWS_EXACT", "UNKNOWN", 0),
        _event(
            "future-1",
            "MGNT",
            "MAGNIT_OFFICIAL_JSON_EXACT",
            "UNKNOWN",
            0,
            published_at="2026-08-11T10:00:00+00:00",
        ),
    ]
    _write_jsonl(backfill / "events.jsonl", events)
    _write_jsonl(backfill / "features.jsonl", [{"event_id": "event-1"}, {"event_id": "event-2"}])
    _write_jsonl(backfill / "targets.jsonl", [_target("event-1"), _target("event-2")])
    _write_jsonl(
        backfill / "semantic-material-provenance.jsonl",
        [
            {"event_id": "event-1", "publication_material_available": True},
            {"event_id": "event-2", "publication_material_available": True},
        ],
    )
    _write_jsonl(
        tz / "audited-sources.jsonl",
        [
            {
                "ticker": "AFLT",
                "issuer": "Aeroflot",
                "official_domain": "aeroflot.ru",
                "source_url": "https://aeroflot.ru/news",
                "source_mechanism": "PUBLIC_HTML_ARCHIVE",
                "status": "CLOCK_TIME_WITHOUT_TIMEZONE",
                "source_family": "AFLT_SOURCE",
            },
            {
                "ticker": "AFLT",
                "issuer": "Aeroflot",
                "official_domain": "aeroflot.ru",
                "source_url": "https://aeroflot.ru/news",
                "source_mechanism": "PUBLIC_HTML_ARCHIVE",
                "status": "CLOCK_TIME_WITHOUT_TIMEZONE",
                "source_family": "AFLT_SOURCE",
            },
        ],
    )
    _write_jsonl(
        issuer_diversity / "candidate-sources.jsonl",
        [
            {
                "ticker": "MVID",
                "issuer": "M.Video",
                "official_domain": "mvideoeldorado.ru",
                "source_url": "https://mvideoeldorado.ru/investor-news",
                "source_id": "MVIDEOELDORADO_IR_NEWS_EXACT_V1",
                "source_family": "MVIDEOELDORADO_IR_NEWS_EXACT_V1",
                "mechanism": "PUBLIC_IR_NEWS_ARCHIVE",
                "status": "NEW_EXACT_HISTORICAL_CAPABLE",
                "selection_reason": "exact timestamps found",
                "source_selection_notes": "timezone evidence missing after maturation audit",
            }
        ],
    )
    _write_json(
        consolidated / "manifest.json",
        {
            "ARTIFACT_SHA": "consolidated-sha",
            "COMBINED_HISTORICAL_INPUT": 1,
            "NEW_FEATURE_READY": 0,
            "PER_TICKER": {"AFKS": {"feature_ready": 0}},
        },
    )
    _write_json(
        chep / "manifest.json",
        {
            "ARTIFACT_SHA": "chep-sha",
            "CHEP_HISTORICAL_EVENTS_TOTAL": 1,
            "NEW_FEATURE_READY_EVENTS": 0,
            "BLOCKER_COUNTS": {"SECURITY_HISTORY_MISSING": 1},
        },
    )
    return {
        "backfill": backfill,
        "readiness": readiness,
        "tz": tz,
        "issuer_diversity": issuer_diversity,
        "consolidated": consolidated,
        "chep": chep,
    }


def _event(
    event_id: str,
    ticker: str,
    source_code: str,
    event_type: str,
    fact_count: int,
    *,
    published_at: str = "2026-08-10T10:00:00+00:00",
) -> dict[str, object]:
    return {
        "metadata": {
            "event_id": event_id,
            "ticker": ticker,
            "issuer": ticker,
            "source_code": source_code,
            "source_item_id": f"https://example.test/{event_id}",
            "publication_timestamp_utc": published_at,
            "publication_date": published_at[:10],
            "publication_timezone": "UTC",
            "timestamp_quality": "EXACT",
            "timestamp_source_field": "datePublished",
            "future_holdout": published_at[:10] >= "2026-08-11",
        },
        "event_features": {
            "primary_event_type": event_type,
            "event_count": 0 if event_type == "UNKNOWN" else 1,
            "fact_count": fact_count,
        },
    }


def _target(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "horizons": {horizon: {"available": True} for horizon in HORIZONS},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


HORIZONS = ("1m", "5m", "15m", "30m", "60m")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
