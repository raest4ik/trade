from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.expand_exact_event_archive_depth import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_source_depth_expansion.application import (
    build_source_depth_expansion_artifact,
)
from src.exact_event_source_depth_expansion.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    MAX_ITEMS_PER_SOURCE,
    MAX_PAGES_PER_SOURCE,
    ArchiveBlocker,
    parse_rfc822_timestamp,
    sha256_payload,
    source_depth_safety_flags,
)


def test_cli_defaults_to_source_depth_expansion_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "3" * 40])

    assert args.input_dir == "artifacts/exact-event-security-history-recovery-v1"
    assert args.source_registry == "artifacts/exact-event-source-diversity-v3/source-registry.jsonl"
    assert args.output_dir == "artifacts/exact-event-source-depth-expansion-v4"
    assert args.archive_cache_root is None


def test_exact_timestamp_requires_timezone_and_never_coerces_date_only() -> None:
    assert parse_rfc822_timestamp("Thu, 30 Jul 2026 11:15:00 +0300") == datetime(
        2026, 7, 30, 8, 15, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="TIMESTAMP_NOT_EXACT"):
        parse_rfc822_timestamp("Thu, 30 Jul 2026 11:15:00")

    flags = source_depth_safety_flags()
    assert flags["DATE_ONLY_COERCIONS"] == 0
    assert flags["FETCH_TIME_USED_AS_PUBLICATION_TIME"] is False


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = source_depth_safety_flags()
    assert flags["DATA_ACQUISITION_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["CONFIRMED_SIGNAL"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["MOEX_SUBSTITUTION_USED"] is False
    assert flags["FORWARD_FILL_USED"] is False
    assert flags["DATA_COST_RUB"] == 0


def test_build_adds_only_exact_official_archive_items_and_preserves_existing_rows(
    tmp_path: Path,
) -> None:
    input_root, registry_path, cache_root = _write_fixture(tmp_path)
    manifest = build_source_depth_expansion_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="3d90ec2336a69a9ddaf48a2b595b77f20d994c03",
        git_sha="4" * 40,
        archive_cache_root=cache_root,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["EXACT_TOTAL_BEFORE"] == 91
    assert manifest["EXACT_TOTAL_AFTER"] == 93
    assert manifest["EXACT_DELTA"] == 2
    assert manifest["REACTION_READY_BEFORE"] == manifest["REACTION_READY_AFTER"] == 91
    assert manifest["FEATURE_READY_BEFORE"] == manifest["FEATURE_READY_AFTER"] == 91
    assert manifest["NEW_EXACT_EVENTS"] == 2
    assert manifest["NEW_EXACT_HISTORICAL"] == 1
    assert manifest["NEW_EXACT_FUTURE_METADATA_ONLY"] == 1
    assert manifest["NEW_EXACT_TICKERS"] == ["AAA"]
    assert manifest["NEW_EVENTS_PRIORITY_TIER_1"] == 2
    assert manifest["NEW_EVENTS_PRIORITY_TIER_2"] == 0
    assert manifest["NEW_EVENTS_PRIORITY_TIER_3"] == 0
    assert manifest["NEW_EVENTS_DEPRIORITIZED"] == 0
    assert manifest["NEW_REACTION_READY"] == 0
    assert manifest["NEW_FEATURE_READY"] == 0
    assert manifest["MARKET_MATURATION_BLOCKERS"] == {"MARKET_HISTORY_MISSING": 1}
    assert manifest["DUPLICATE_RECONCILIATION"] == "PASS"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_TARGET_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["FETCH_TIME_USED_AS_PUBLICATION_TIME"] is False
    assert len(str(manifest["SOURCE_DEPTH_PRIORITY_RULES_SHA"])) == 64
    assert len(str(manifest["PRIORITY_TICKERS_SHA"])) == 64
    assert len(str(manifest["ARTIFACT_SHA"])) == 64

    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "events.jsonl")
    assert rows[:91] == _existing_events()
    new_rows = rows[91:]
    historical = cast("dict[str, Any]", new_rows[0])
    future = cast("dict[str, Any]", new_rows[1])
    assert historical["metadata"]["future_holdout"] is False
    assert historical["target_availability"]["research_outcomes_visible"] is False
    assert historical["target_availability"]["missing_reason"] == "MARKET_HISTORY_MISSING"
    assert future["metadata"]["future_holdout"] is True
    assert future["metadata"]["publication_date"] >= FUTURE_EVENT_HOLDOUT_START.isoformat()
    assert future["target_availability"]["status"] == "FUTURE_HOLDOUT_METADATA_ONLY"
    assert future["target_availability"]["research_outcomes_visible"] is False
    assert future["quality"]["feature_cutoff"] == future["metadata"]["publication_timestamp_utc"]
    assert future["quality"]["no_forward_fill"] is True
    assert future["quality"]["no_source_mixing"] is True


def test_priority_is_outcome_free_deterministic_and_includes_registry_only_tickers(
    tmp_path: Path,
) -> None:
    input_root, registry_path, cache_root = _write_fixture(tmp_path)
    left = build_source_depth_expansion_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        output_root=tmp_path / "left",
        base_main_sha="3d90ec2336a69a9ddaf48a2b595b77f20d994c03",
        git_sha="4" * 40,
        archive_cache_root=cache_root,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    right = build_source_depth_expansion_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        output_root=tmp_path / "right",
        base_main_sha="3d90ec2336a69a9ddaf48a2b595b77f20d994c03",
        git_sha="5" * 40,
        archive_cache_root=cache_root,
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
    )

    assert left["PRIORITY_TICKERS"].index("ZERO") < left["PRIORITY_TICKERS"].index("BBB")
    assert left["PRIORITY_TICKERS"].index("AAA") < left["PRIORITY_TICKERS"].index("BBB")
    assert left["PRIORITY_TICKERS"].index("AAA") < left["PRIORITY_TICKERS"].index("ZERO")
    assert left["PRIORITY_TICKERS"].index("BBB") < left["PRIORITY_TICKERS"].index("CCC")
    assert left["PRIORITY_TICKERS"].index("CCC") < left["PRIORITY_TICKERS"].index("DDD")
    assert left["SOURCE_DEPTH_PRIORITY_RULES"]["forbidden_inputs"] == [
        "returns",
        "targets",
        "predictions",
        "model_metrics",
        "TEST_metrics",
    ]
    assert left["PRIORITY_TICKERS_SHA"] == right["PRIORITY_TICKERS_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert left["DETERMINISTIC_REPLAY"] == "PASS"


def test_archive_pagination_and_item_probe_are_bounded(tmp_path: Path) -> None:
    input_root, registry_path, cache_root = _write_fixture(
        tmp_path, include_many_archive_pages=True
    )
    manifest = build_source_depth_expansion_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        output_root=tmp_path / "bounded",
        base_main_sha="3d90ec2336a69a9ddaf48a2b595b77f20d994c03",
        git_sha="4" * 40,
        archive_cache_root=cache_root,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    audit = {row["TICKER"]: row for row in cast("list[dict[str, Any]]", manifest["ARCHIVE_AUDIT"])}
    assert audit["AAA"]["PAGES_PROBED"] <= MAX_PAGES_PER_SOURCE
    assert audit["AAA"]["ITEMS_DISCOVERED"] <= MAX_ITEMS_PER_SOURCE
    assert audit["AAA"]["BLOCKER"] is None


def test_blockers_fail_closed_for_missing_date_only_policy_and_duplicates(tmp_path: Path) -> None:
    input_root, registry_path, cache_root = _write_fixture(tmp_path)
    manifest = build_source_depth_expansion_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        output_root=tmp_path / "blockers",
        base_main_sha="3d90ec2336a69a9ddaf48a2b595b77f20d994c03",
        git_sha="4" * 40,
        archive_cache_root=cache_root,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    audit = {row["TICKER"]: row for row in cast("list[dict[str, Any]]", manifest["ARCHIVE_AUDIT"])}
    assert audit["ZERO"]["BLOCKER"] == ArchiveBlocker.NO_OFFICIAL_SOURCE_FOUND.value
    assert audit["BBB"]["BLOCKER"] == ArchiveBlocker.SOURCE_DATE_ONLY.value
    assert audit["CCC"]["BLOCKER"] == ArchiveBlocker.DUPLICATE_ONLY.value
    assert audit["DDD"]["BLOCKER"] == ArchiveBlocker.NO_ARCHIVE.value
    assert audit["POLICY"]["BLOCKER"] == ArchiveBlocker.ROBOTS_OR_POLICY_BLOCKED.value
    assert audit["AMBIG"]["BLOCKER"] == ArchiveBlocker.TICKER_AMBIGUOUS.value


def test_frozen_rules_qwen_and_documentation_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA

    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-source-depth-expansion-v4.md"
    ).read_text(encoding="utf-8")
    assert "T-Invest read-only" in text
    assert "no MOEX substitution" in text
    assert "no forward-fill" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text


def _write_fixture(
    tmp_path: Path, *, include_many_archive_pages: bool = False
) -> tuple[Path, Path, Path]:
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
            _registry_row("ZERO", "Zero Issuer", None, "UNKNOWN", archive=False),
            _registry_row("AAA", "AAA Issuer", "AAA_OFFICIAL_ARCHIVE", "EXACT", archive=True),
            _registry_row("BBB", "BBB Issuer", "BBB_DATE_ONLY", "DATE_ONLY", archive=True),
            _registry_row("CCC", "CCC Issuer", "CCC_OFFICIAL_ARCHIVE", "EXACT", archive=True),
            _registry_row("DDD", "DDD Issuer", "DDD_CURRENT_ONLY", "EXACT", archive=False),
            _registry_row(
                "POLICY",
                "Policy Issuer",
                "POLICY_OFFICIAL_ARCHIVE",
                "EXACT",
                archive=True,
            ),
            _registry_row(
                "AMBIG",
                "Ambiguous Issuer",
                "AMBIG_OFFICIAL_ARCHIVE",
                "EXACT",
                archive=True,
            ),
        ],
    )

    cache_root = tmp_path / "archive-cache"
    _write_archive(
        cache_root / "AAA" / "page-1.xml",
        [
            (
                "Historical exact AAA",
                "https://issuer.example/aaa-historical",
                "Thu, 30 Jul 2026 11:15:00 +0300",
            ),
            (
                "Future exact AAA",
                "https://issuer.example/aaa-future",
                "Thu, 13 Aug 2026 11:15:00 +0300",
            ),
            ("Date only AAA", "https://issuer.example/aaa-date", "Thu, 30 Jul 2026 11:15:00"),
        ],
    )
    _write_archive(
        cache_root / "CCC" / "page-1.xml",
        [
            (
                "Duplicate CCC",
                "https://issuer.example/ccc-existing",
                "Thu, 30 Jul 2026 11:15:00 +0300",
            ),
        ],
    )
    (cache_root / "POLICY").mkdir(parents=True)
    _write_json(cache_root / "POLICY" / "robots-policy-blocked.json", {"status": "blocked"})
    (cache_root / "AMBIG").mkdir(parents=True)
    _write_json(cache_root / "AMBIG" / "ticker-ambiguous.json", {"status": "ambiguous"})

    if include_many_archive_pages:
        for index in range(MAX_PAGES_PER_SOURCE + 1):
            items = [
                (
                    f"Bounded AAA {index}-{item}",
                    f"https://issuer.example/aaa-bounded-{index}-{item}",
                    "Thu, 30 Jul 2026 11:15:00 +0300",
                )
                for item in range(MAX_ITEMS_PER_SOURCE)
            ]
            _write_archive(cache_root / "AAA" / f"page-{index + 2}.xml", items)

    return input_root, registry_path, cache_root


def _existing_events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker, count in [("AAA", 2), ("BBB", 7), ("CCC", 22), ("DDD", 60)]:
        for index in range(count):
            rows.append(_event_row(f"{ticker.lower()}-{index}", ticker))
    return rows


def _existing_features() -> list[dict[str, object]]:
    return [
        {
            "event_id": cast("dict[str, Any]", row["metadata"])["event_id"],
            "feature_cutoff": "2026-07-01T06:59:00+00:00",
            "event_features": {"primary_event_type": "OTHER"},
            "market_features": {"pre_return_5m": "0.001"},
        }
        for row in _existing_events()
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
    link = (
        "https://issuer.example/ccc-existing"
        if event_id == "ccc-0"
        else f"https://issuer.example/{event_id}"
    )
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": f"{ticker}_OFFICIAL_ARCHIVE",
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
            "timestamp_source_field": "official archive fixture",
            "timestamp_quality": "EXACT",
            "future_holdout": False,
            "session_state": "DURING_MAIN_SESSION",
            "source_family": f"{ticker}_OFFICIAL_ARCHIVE",
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
    ticker: str,
    issuer: str,
    source_family: str | None,
    timestamp_capability: str,
    *,
    archive: bool,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "issuer": issuer,
        "instrument_uid": f"uid-{ticker}",
        "official_domain": "issuer.example" if source_family else None,
        "source_url": f"https://issuer.example/{ticker.lower()}" if source_family else None,
        "source_family": source_family,
        "transport_type": "official_rss" if source_family else None,
        "timestamp_capability": timestamp_capability,
        "timezone_semantics": "EXPLICIT" if timestamp_capability == "EXACT" else "UNKNOWN",
        "archive": archive,
        "historical_archive_start": "2026-01-01" if archive else None,
        "source_policy_status": "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED"
        if source_family
        else "UNKNOWN_FAIL_CLOSED",
        "collector_status": "SOURCE_READY" if source_family else "NO_OFFICIAL_NEWS_ARCHIVE",
        "provenance": "synthetic official zero-cost fixture",
    }


def _write_archive(path: Path, items: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"<item><title>{title}</title><link>{link}</link><pubDate>{published}</pubDate></item>"
        for title, link, published in items
    )
    path.write_text(
        f'<?xml version="1.0"?><rss><channel>{body}</channel></rss>\n', encoding="utf-8"
    )


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


def test_artifact_hash_helper_is_stable() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})
