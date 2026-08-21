from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.build_exact_event_source_diversity_v3 import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_source_diversity_v3.application import (
    build_source_diversity_v3_artifact,
)
from src.exact_event_source_diversity_v3.domain import (
    ARTIFACT_VERSION,
    INPUT_WARMUP_DATASET_SHA,
    SourceStatus,
    concentration,
    parse_rss_pubdate_utc,
    sha256_payload,
    source_diversity_safety_flags,
)


def test_cli_defaults_to_source_diversity_v3_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "c" * 40])
    assert args.warmup_dir == "artifacts/exact-event-market-history-warmup-recovery-v1"
    assert args.v2_dir == "artifacts/exact-event-market-dataset-v2"
    assert args.output_dir == "artifacts/exact-event-source-diversity-v3"


def test_rss_pubdate_requires_timezone_and_normalizes_to_utc() -> None:
    assert parse_rss_pubdate_utc("Thu, 13 Aug 2026 10:00:00 +0300") == datetime(
        2026, 8, 13, 7, 0, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="TIMESTAMP_TIMEZONE_UNRESOLVED"):
        parse_rss_pubdate_utc("Thu, 13 Aug 2026 10:00:00")


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = source_diversity_safety_flags()
    assert flags["DATA_ACQUISITION_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["CONFIRMED_SIGNAL"] is False
    assert flags["BACKTEST_APPROVED"] is False
    assert flags["PAPER_TRADING_APPROVED"] is False
    assert flags["REAL_TRADING_APPROVED"] is False


def test_source_diversity_v3_build_is_self_contained_and_preserves_rows(
    tmp_path: Path,
) -> None:
    warmup, v2, universe, feed = _write_fixture(tmp_path)
    manifest = build_source_diversity_v3_artifact(
        warmup_root=warmup,
        v2_root=v2,
        universe_path=universe,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="c0f072362fa99266f9c736a2ac8730093c62861b",
        git_sha="d" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        moex_feed_path=feed,
    )
    assert manifest["INPUT_DATASET_SHA"] == INPUT_WARMUP_DATASET_SHA
    assert manifest["EXACT_TOTAL_BEFORE"] == 1
    assert manifest["EXACT_TOTAL_AFTER"] == 2
    assert manifest["EXACT_DELTA"] == 1
    assert manifest["EXACT_TICKERS_BEFORE"] == 1
    assert manifest["EXACT_TICKERS_AFTER"] == 2
    assert manifest["NEW_EXACT_TICKERS"] == ["NEW"]
    assert manifest["REACTION_READY_BEFORE"] == manifest["REACTION_READY_AFTER"] == 1
    assert manifest["FEATURE_READY_BEFORE"] == manifest["FEATURE_READY_AFTER"] == 1
    assert manifest["EXACT_V2_PRESERVED"] == "YES"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["DUPLICATE_RECONCILIATION"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert len(manifest["ARTIFACT_SHA"]) == 64
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})
    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "events.jsonl")
    assert rows[0] == _event_row("old-event", "OLD", feature_ready=True)
    new_metadata = cast("dict[str, Any]", rows[1]["metadata"])
    new_availability = cast("dict[str, Any]", rows[1]["target_availability"])
    assert new_metadata["ticker"] == "NEW"
    assert new_metadata["future_holdout"] is True
    assert new_availability["research_outcomes_visible"] is False
    assert new_availability["status"] == "FUTURE_HOLDOUT_METADATA_ONLY"


def test_source_registry_preserves_date_only_and_fail_closed_statuses(tmp_path: Path) -> None:
    warmup, v2, universe, feed = _write_fixture(tmp_path)
    build_source_diversity_v3_artifact(
        warmup_root=warmup,
        v2_root=v2,
        universe_path=universe,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="c0f072362fa99266f9c736a2ac8730093c62861b",
        git_sha="d" * 40,
        moex_feed_path=feed,
    )
    registry = {
        row["ticker"]: row
        for row in _read_jsonl(tmp_path / ARTIFACT_VERSION / "source-registry.jsonl")
    }
    assert registry["OLD"]["timestamp_capability"] == SourceStatus.EXACT
    assert registry["DATE"]["timestamp_capability"] == SourceStatus.DATE_ONLY
    assert registry["UNKNOWN"]["timestamp_capability"] == SourceStatus.UNKNOWN
    assert registry["NEW"]["timestamp_capability"] == SourceStatus.EXACT
    assert registry["NEW"]["official_domain"] == "www.moex.com"
    assert registry["NEW"]["policy_status"] == "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED"


def test_deterministic_artifact_hashes(tmp_path: Path) -> None:
    warmup, v2, universe, feed = _write_fixture(tmp_path)
    left = build_source_diversity_v3_artifact(
        warmup_root=warmup,
        v2_root=v2,
        universe_path=universe,
        output_root=tmp_path / "left",
        base_main_sha="c0f072362fa99266f9c736a2ac8730093c62861b",
        git_sha="d" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        moex_feed_path=feed,
    )
    right = build_source_diversity_v3_artifact(
        warmup_root=warmup,
        v2_root=v2,
        universe_path=universe,
        output_root=tmp_path / "right",
        base_main_sha="c0f072362fa99266f9c736a2ac8730093c62861b",
        git_sha="d" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        moex_feed_path=feed,
    )
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["SOURCE_REGISTRY_SHA"] == right["SOURCE_REGISTRY_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def test_frozen_rules_v3_and_qwen_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA


def test_concentration_metrics_are_target_free() -> None:
    result = concentration(Counter({"A": 3, "B": 1}))
    assert result["top1_share"] == 0.75
    assert result["top3_share"] == 1.0
    assert result["hhi"] == 0.625
    assert result["effective_count"] == 1.6


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    warmup = tmp_path / "warmup"
    v2 = tmp_path / "v2"
    warmup.mkdir()
    v2.mkdir()
    _write_json(
        warmup / "manifest.json",
        {
            "OUTPUT_DATASET_SHA": INPUT_WARMUP_DATASET_SHA,
            "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
            "LEAKAGE_CHECK": "PASS",
            "safety": {
                "TEST_OUTCOME_USED": False,
                "FUTURE_EVENT_HOLDOUT_USED": False,
            },
        },
    )
    _write_jsonl(warmup / "events.jsonl", [_event_row("old-event", "OLD", feature_ready=True)])
    _write_jsonl(
        warmup / "features.jsonl",
        [
            {
                "event_id": "old-event",
                "feature_cutoff": "2026-08-10T10:00:00+00:00",
                "event_features": {"primary_event_type": "DIVIDEND"},
                "market_features": {"pre_return_5m": "0.001"},
            }
        ],
    )
    _write_jsonl(v2 / "targets.jsonl", [{"event_id": "old-event", "horizons": {}}])
    _write_jsonl(v2 / "clusters.jsonl", [{"event_id": "old-event", "event_cluster_id": "c-old"}])
    _write_jsonl(
        v2 / "source-registry.jsonl",
        [
            _registry_row("OLD", "Old Issuer", "EXACT", "OLD_OFFICIAL_RSS"),
            _registry_row("DATE", "Date Issuer", "DATE_ONLY", "DATE_ONLY_ARCHIVE"),
            _registry_row("NEW", "New Issuer", "UNKNOWN", None),
            _registry_row("UNKNOWN", "Unknown Issuer", "UNKNOWN", None),
        ],
    )
    universe = tmp_path / "instrument-mapping.json"
    _write_json(
        universe,
        {
            "instruments": [
                _instrument("OLD", "Old Issuer"),
                _instrument("DATE", "Date Issuer"),
                _instrument("NEW", "New Issuer"),
                _instrument("UNKNOWN", "Unknown Issuer"),
                {"ticker": "IMOEX", "name": "Benchmark", "class_code": "INDX"},
            ]
        },
    )
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<rss><channel>
<item>
<title>Official listing notice NEW</title>
<link>https://www.moex.com/n-new</link>
<pubDate>Thu, 13 Aug 2026 10:00:00 +0300</pubDate>
<description>Official public decision for security NEW.</description>
</item>
<item>
<title>Official listing notice OLD</title>
<link>https://www.moex.com/n-old</link>
<pubDate>Thu, 13 Aug 2026 11:00:00 +0300</pubDate>
<description>Official public decision for security OLD.</description>
</item>
</channel></rss>
""",
        encoding="utf-8",
    )
    return warmup, v2, universe, feed


def _event_row(event_id: str, ticker: str, *, feature_ready: bool) -> dict[str, object]:
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": f"{ticker}_OFFICIAL_RSS",
            "source_item_id": event_id,
            "canonical_url": f"https://issuer.example/{event_id}",
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": "2026-08-10T10:00:00+00:00",
            "publication_timestamp_raw": "2026-08-10T10:00:00+00:00",
            "publication_date": "2026-08-10",
            "publication_time": "10:00:00",
            "publication_timezone": "UTC",
            "timestamp_source_field": "synthetic exact timestamp",
            "timestamp_quality": "EXACT",
            "future_holdout": False,
            "session_state": "DURING_MAIN_SESSION",
            "title_hash": event_id,
        },
        "event_features": {"primary_event_type": "DIVIDEND", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": {"pre_return_5m": "0.001"},
        "target_availability": {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": feature_ready,
            "status": "REACTION_READY",
            "missing_reason": None,
        },
        "quality": {
            "feature_cutoff": "2026-08-10T10:00:00+00:00",
            "reaction_starts_after_or_at_publication": True,
            "security_benchmark_same_window": True,
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
        },
    }


def _registry_row(
    ticker: str, issuer: str, timestamp_capability: str, source_family: str | None
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "issuer": issuer,
        "instrument_uid": f"uid-{ticker}",
        "official_domain": "issuer.example" if source_family else None,
        "source_url": f"https://issuer.example/{ticker}" if source_family else None,
        "source_family": source_family,
        "timestamp_capability": timestamp_capability,
        "timezone_semantics": "EXPLICIT" if timestamp_capability == "EXACT" else "UNKNOWN",
        "historical_archive_start": "2026-01-01" if source_family else None,
        "collector_status": "SOURCE_READY" if source_family else "NO_OFFICIAL_NEWS_ARCHIVE",
        "source_policy_status": (
            "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED" if source_family else "UNKNOWN_FAIL_CLOSED"
        ),
        "reason": "synthetic registry row",
    }


def _instrument(ticker: str, name: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": name,
        "class_code": "TQBR",
        "instrument_uid": f"uid-{ticker}",
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
