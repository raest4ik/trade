from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.discover_exact_event_official_sources import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_official_source_discovery.application import (
    build_official_source_discovery_artifact,
)
from src.exact_event_official_source_discovery.domain import (
    ARTIFACT_VERSION,
    DISCOVERY_PRIORITY_RULES,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    MAX_TICKERS,
    DiscoveryState,
    parse_exact_timestamp,
    priority_tier,
    sha256_payload,
)


def test_cli_defaults_to_v4_inputs_and_v5_output() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.input_dir == "artifacts/exact-event-source-depth-expansion-v4"
    assert (
        args.source_registry
        == "artifacts/exact-event-source-depth-expansion-v4/source-registry.jsonl"
    )
    assert args.universe == "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
    assert args.output_dir == "artifacts/exact-event-official-source-discovery-v5"
    assert args.discovery_cache_root is None


def test_priority_is_metadata_only_and_deprioritizes_dominant_tickers() -> None:
    assert (
        priority_tier(ticker="AAA", exact_count=2, feature_ready_count=0, in_exact_corpus=True)
        == "A_ZERO_FEATURE_READY"
    )
    assert (
        priority_tier(ticker="BBB", exact_count=4, feature_ready_count=4, in_exact_corpus=True)
        == "B_EXACT_1_5"
    )
    assert (
        priority_tier(ticker="CCC", exact_count=12, feature_ready_count=12, in_exact_corpus=True)
        == "C_EXACT_6_20"
    )
    assert (
        priority_tier(ticker="ZERO", exact_count=0, feature_ready_count=0, in_exact_corpus=False)
        == "D_CANONICAL_TQBR_NOT_IN_EXACT"
    )
    assert (
        priority_tier(ticker="MGNT", exact_count=2, feature_ready_count=0, in_exact_corpus=True)
        == "DEPRIORITIZED"
    )
    assert DISCOVERY_PRIORITY_RULES["forbidden_inputs"] == [
        "returns",
        "targets",
        "predictions",
        "model_metrics",
        "TEST_metrics",
    ]


def test_exact_timestamp_rejects_date_only_and_requires_timezone() -> None:
    assert parse_exact_timestamp("2026-07-30T08:15:00+00:00") == datetime(
        2026, 7, 30, 8, 15, tzinfo=UTC
    )
    assert parse_exact_timestamp("Thu, 30 Jul 2026 11:15:00 +0300") == datetime(
        2026, 7, 30, 8, 15, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="TIMESTAMP_NOT_EXACT"):
        parse_exact_timestamp("2026-07-30")
    with pytest.raises(ValueError, match="TIMESTAMP_NOT_EXACT"):
        parse_exact_timestamp("2026-07-30T08:15:00")


def test_build_discovers_new_exact_source_extracts_events_and_preserves_rows(
    tmp_path: Path,
) -> None:
    input_root, registry_path, universe_path, cache_root = _write_fixture(tmp_path)
    manifest = build_official_source_discovery_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="8fe0cf12d27e9621e7f64109ec58e7702b863691",
        git_sha="8" * 40,
        discovery_cache_root=cache_root,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["EXACT_TOTAL_BEFORE"] == 15
    assert manifest["EXACT_TOTAL_AFTER"] == 17
    assert manifest["EXACT_DELTA"] == 2
    assert manifest["NEW_OFFICIAL_SOURCES_FOUND"] == 1
    assert manifest["NEW_EXACT_CAPABLE_SOURCES"] == 1
    assert manifest["NEW_ARCHIVE_CAPABLE_SOURCES"] == 1
    assert manifest["NEW_EXACT_EVENTS"] == 2
    assert manifest["NEW_EXACT_HISTORICAL"] == 1
    assert manifest["NEW_EXACT_FUTURE_METADATA_ONLY"] == 1
    assert manifest["NEW_EXACT_TICKERS"] == ["AAA"]
    assert manifest["NEW_REACTION_READY"] == 0
    assert manifest["NEW_FEATURE_READY"] == 0
    assert manifest["MARKET_MATURATION_BLOCKERS"] == {"MARKET_HISTORY_MISSING": 1}
    assert manifest["DUPLICATE_RECONCILIATION"] == "PASS"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_TARGET_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["FETCH_TIME_USED_AS_PUBLICATION_TIME"] is False
    assert manifest["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False

    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "events.jsonl")
    assert rows[:15] == _existing_events()
    historical = cast("dict[str, Any]", rows[15])
    future = cast("dict[str, Any]", rows[16])
    historical_metadata = cast("dict[str, Any]", historical["metadata"])
    historical_availability = cast("dict[str, Any]", historical["target_availability"])
    future_metadata = cast("dict[str, Any]", future["metadata"])
    future_availability = cast("dict[str, Any]", future["target_availability"])
    future_quality = cast("dict[str, Any]", future["quality"])
    assert historical_metadata["future_holdout"] is False
    assert historical_availability["research_outcomes_visible"] is False
    assert historical_availability["missing_reason"] == "MARKET_HISTORY_MISSING"
    assert future_metadata["publication_date"] >= FUTURE_EVENT_HOLDOUT_START.isoformat()
    assert future_metadata["future_holdout"] is True
    assert future_availability["status"] == "FUTURE_HOLDOUT_METADATA_ONLY"
    assert future_availability["research_outcomes_visible"] is False
    assert future_quality["feature_cutoff"] == future_metadata["publication_timestamp_utc"]
    assert future_quality["no_forward_fill"] is True
    assert future_quality["no_source_mixing"] is True


def test_discovery_states_fail_closed_for_bad_candidates(tmp_path: Path) -> None:
    input_root, registry_path, universe_path, cache_root = _write_fixture(tmp_path)
    manifest = build_official_source_discovery_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "states",
        base_main_sha="8fe0cf12d27e9621e7f64109ec58e7702b863691",
        git_sha="8" * 40,
        discovery_cache_root=cache_root,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    audit = {
        (row["TICKER"], row["SOURCE_FAMILY"]): row
        for row in cast("list[dict[str, Any]]", manifest["SOURCE_DISCOVERY_AUDIT"])
    }
    assert audit[("BBB", "BBB_DATE_ONLY_SOURCE")]["CANDIDATE_STATE"] == (
        DiscoveryState.DATE_ONLY_SOURCE.value
    )
    assert audit[("CCC", "CCC_AMBIGUOUS_SOURCE")]["CANDIDATE_STATE"] == (
        DiscoveryState.AMBIGUOUS_SOURCE_IDENTITY.value
    )
    assert audit[("DDD", "DDD_POLICY_BLOCKED")]["CANDIDATE_STATE"] == (
        DiscoveryState.ROBOTS_BLOCKED.value
    )
    assert audit[("EEE", "EEE_AUTH_SOURCE")]["CANDIDATE_STATE"] == (
        DiscoveryState.AUTH_REQUIRED.value
    )


def test_priority_and_artifact_are_deterministic(tmp_path: Path) -> None:
    input_root, registry_path, universe_path, cache_root = _write_fixture(tmp_path)
    left = build_official_source_discovery_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "left",
        base_main_sha="8fe0cf12d27e9621e7f64109ec58e7702b863691",
        git_sha="8" * 40,
        discovery_cache_root=cache_root,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )
    right = build_official_source_discovery_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "right",
        base_main_sha="8fe0cf12d27e9621e7f64109ec58e7702b863691",
        git_sha="9" * 40,
        discovery_cache_root=cache_root,
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )

    assert left["PRIORITY_TICKERS"][:4] == ["AAA", "CCC", "DDD", "EEE"]
    assert left["PRIORITY_TICKERS"].index("BBB") > left["PRIORITY_TICKERS"].index("EEE")
    assert left["PRIORITY_TICKERS"].index("ZERO") > left["PRIORITY_TICKERS"].index("BBB")
    assert "MGNT" not in left["PRIORITY_TICKERS"][:5]
    assert len(left["PRIORITY_TICKERS"]) <= MAX_TICKERS
    assert left["PRIORITY_TICKERS_SHA"] == right["PRIORITY_TICKERS_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert left["DETERMINISTIC_REPLAY"] == "PASS"


def test_frozen_rules_qwen_docs_and_safety_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA

    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-official-source-discovery-v5.md"
    ).read_text(encoding="utf-8")
    assert "T-Invest read-only" in text
    assert "no MOEX substitution" in text
    assert "no forward-fill" in text
    assert "no sparse label family" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text


def test_hash_helper_is_stable() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_json(
        input_root / "manifest.json",
        {
            "OUTPUT_DATASET_SHA": INPUT_DATASET_SHA,
            "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
            "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
            "EXISTING_TARGET_ROWS_PRESERVED": "PASS",
            "TEST_OUTCOME_USED": False,
            "FUTURE_EVENT_HOLDOUT_USED": False,
        },
    )
    _write_jsonl(input_root / "events.jsonl", _existing_events())
    _write_jsonl(input_root / "features.jsonl", _existing_features())
    _write_jsonl(input_root / "targets.jsonl", _existing_targets())
    _write_jsonl(input_root / "clusters.jsonl", _existing_clusters())

    registry_path = tmp_path / "source-registry.jsonl"
    _write_jsonl(
        registry_path,
        [
            _registry_row("AAA", "AAA Issuer", None, "UNKNOWN"),
            _registry_row("BBB", "BBB Issuer", None, "UNKNOWN"),
            _registry_row("CCC", "CCC Issuer", None, "UNKNOWN"),
            _registry_row("DDD", "DDD Issuer", None, "UNKNOWN"),
            _registry_row("EEE", "EEE Issuer", None, "UNKNOWN"),
            _registry_row("MGNT", "Magnit", "MGNT_EXISTING", "EXACT"),
        ],
    )

    universe_path = tmp_path / "instrument-mapping.json"
    _write_json(
        universe_path,
        {
            "instruments": [
                _instrument("AAA", "AAA Issuer"),
                _instrument("BBB", "BBB Issuer"),
                _instrument("CCC", "CCC Issuer"),
                _instrument("DDD", "DDD Issuer"),
                _instrument("EEE", "EEE Issuer"),
                _instrument("ZERO", "Zero New Issuer"),
                _instrument("MGNT", "Magnit"),
            ]
        },
    )

    cache_root = tmp_path / "source-cache"
    _write_json(cache_root / "AAA" / "candidate.json", _exact_candidate("AAA"))
    _write_json(
        cache_root / "BBB" / "candidate.json",
        {
            **_base_candidate("BBB", "BBB_DATE_ONLY_SOURCE"),
            "timestamp_capability": "DATE_ONLY",
            "timestamp_field": "date",
            "timezone_provenance": None,
        },
    )
    _write_json(
        cache_root / "CCC" / "candidate.json",
        {
            **_base_candidate("CCC", "CCC_AMBIGUOUS_SOURCE"),
            "ambiguous_source_identity": True,
        },
    )
    _write_json(
        cache_root / "DDD" / "candidate.json",
        {
            **_base_candidate("DDD", "DDD_POLICY_BLOCKED"),
            "policy_status": "ROBOTS_BLOCKED",
        },
    )
    _write_json(
        cache_root / "EEE" / "candidate.json",
        {
            **_base_candidate("EEE", "EEE_AUTH_SOURCE"),
            "auth_required": True,
        },
    )
    return input_root, registry_path, universe_path, cache_root


def _existing_events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker, count in [("AAA", 2), ("BBB", 7), ("CCC", 3), ("DDD", 1), ("EEE", 1), ("MGNT", 1)]:
        for index in range(count):
            rows.append(_event_row(f"{ticker.lower()}-{index}", ticker))
    return rows


def _existing_features() -> list[dict[str, object]]:
    feature_tickers = {"BBB", "CCC", "DDD", "EEE", "MGNT"}
    return [
        {
            "event_id": cast("dict[str, Any]", row["metadata"])["event_id"],
            "feature_cutoff": "2026-07-01T06:59:00+00:00",
            "event_features": {"primary_event_type": "OTHER"},
            "market_features": {"pre_return_5m": "0.001"},
        }
        for row in _existing_events()
        if cast("dict[str, Any]", row["metadata"])["ticker"] in feature_tickers
    ]


def _existing_targets() -> list[dict[str, object]]:
    return [
        {"event_id": cast("dict[str, Any]", row["metadata"])["event_id"], "horizons": {}}
        for row in _existing_events()
    ]


def _existing_clusters() -> list[dict[str, object]]:
    return [
        {
            "event_id": cast("dict[str, Any]", row["metadata"])["event_id"],
            "event_cluster_id": f"cluster-{cast('dict[str, Any]', row['metadata'])['event_id']}",
        }
        for row in _existing_events()
    ]


def _event_row(event_id: str, ticker: str) -> dict[str, object]:
    link = f"https://issuer.example/{event_id}"
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": f"{ticker}_EXISTING_SOURCE",
            "source_item_id": link,
            "canonical_url": link,
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": "2026-07-01T07:00:00+00:00",
            "publication_timestamp_raw": "Wed, 01 Jul 2026 10:00:00 +0300",
            "publication_date": "2026-07-01",
            "publication_time": "07:00:00",
            "publication_timezone": "UTC",
            "timestamp_source_field": "official exact fixture",
            "timestamp_quality": "EXACT",
            "future_holdout": False,
            "session_state": "DURING_MAIN_SESSION",
            "title_hash": event_id,
        },
        "event_features": {"primary_event_type": "OTHER", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": {"pre_return_5m": "0.001"},
        "target_availability": {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": True,
            "status": "REACTION_READY",
            "missing_reason": None,
        },
        "quality": {
            "feature_cutoff": "2026-07-01T06:59:00+00:00",
            "reaction_starts_after_or_at_publication": True,
            "security_benchmark_same_window": True,
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
        },
    }


def _registry_row(
    ticker: str, issuer: str, source_family: str | None, timestamp_capability: str
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "issuer": issuer,
        "instrument_uid": f"uid-{ticker}",
        "source_url": f"https://issuer.example/{ticker.lower()}" if source_family else None,
        "source_family": source_family,
        "timestamp_capability": timestamp_capability,
        "archive": bool(source_family),
        "source_found": bool(source_family),
        "provenance": "synthetic registry row",
    }


def _instrument(ticker: str, name: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": name,
        "class_code": "TQBR",
        "currency": "rub",
        "instrument_uid": f"uid-{ticker}",
    }


def _exact_candidate(ticker: str) -> dict[str, object]:
    return {
        **_base_candidate(ticker, f"{ticker}_OFFICIAL_IR_JSON"),
        "timestamp_capability": "EXACT",
        "timestamp_field": "published_at",
        "timezone_provenance": "explicit JSON offset",
        "archive_capability": True,
        "items": [
            {
                "source_item_id": "aaa-historical-1",
                "canonical_url": "https://aaa.example/ir/news/aaa-historical-1",
                "title": "AAA official exact historical disclosure",
                "published_at": "2026-07-30T11:15:00+03:00",
                "timestamp_field": "published_at",
            },
            {
                "source_item_id": "aaa-future-1",
                "canonical_url": "https://aaa.example/ir/news/aaa-future-1",
                "title": "AAA official exact future disclosure",
                "published_at": "2026-08-13T11:15:00+03:00",
                "timestamp_field": "published_at",
            },
            {
                "source_item_id": "aaa-date-only",
                "canonical_url": "https://aaa.example/ir/news/aaa-date-only",
                "title": "AAA date only rejected",
                "published_at": "2026-07-30",
            },
        ],
    }


def _base_candidate(ticker: str, family: str) -> dict[str, object]:
    domain = f"{ticker.lower()}.example"
    return {
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": f"uid-{ticker}",
        "source_url": f"https://{domain}/ir/news",
        "source_domain": domain,
        "source_type": "ISSUER_IR_JSON",
        "source_family": family,
        "official_source_confirmed": True,
        "timestamp_capability": "EXACT",
        "timestamp_field": "published_at",
        "timezone_provenance": "explicit JSON offset",
        "archive_capability": True,
        "discovery_method": "official JSON endpoint fixture",
        "policy_status": "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED",
        "technical_status": "SOURCE_READY",
        "public_access": True,
        "payment_required": False,
        "auth_required": False,
        "captcha_required": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
