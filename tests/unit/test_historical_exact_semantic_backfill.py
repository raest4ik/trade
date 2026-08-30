from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import src.historical_exact_semantic_backfill.application as backfill_app
from apps.cli.backfill_historical_exact_semantic_features import build_parser
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_feature_readiness_recovery.domain import artifact_sha as diagnosis_artifact_sha
from src.historical_exact_semantic_backfill.application import (
    run_historical_exact_semantic_backfill,
)
from src.historical_exact_semantic_backfill.domain import (
    ARTIFACT_VERSION,
    DEFAULT_DIAGNOSIS_ARTIFACT_ROOT,
    DEFAULT_MARKET_ARTIFACT_ROOT,
    DEFAULT_SNAPSHOT_ROOTS,
    EXPECTED_RULES_V3_FINGERPRINT,
    SemanticBackfillBlocker,
    artifact_sha,
    safety_flags,
)


def test_cli_defaults_to_historical_semantic_backfill_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.diagnosis_dir == DEFAULT_DIAGNOSIS_ARTIFACT_ROOT
    assert args.market_dir == DEFAULT_MARKET_ARTIFACT_ROOT
    assert args.snapshot_roots == list(DEFAULT_SNAPSHOT_ROOTS)
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"


def test_safety_flags_forbid_model_test_backtest_trading_and_semantic_leakage() -> None:
    flags = safety_flags()

    assert flags["RESEARCH_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["BACKTEST_PERFORMED"] is False
    assert flags["FUTURE_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_REACTIONS_COMPUTED"] == 0
    assert flags["FUTURE_TARGETS_COMPUTED"] == 0
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["USES_MARKET_DATA_FOR_SEMANTICS"] is False
    assert flags["USES_REACTION_DATA_FOR_SEMANTICS"] is False
    assert flags["USES_TARGET_DATA_FOR_SEMANTICS"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_semantic_backfill_matches_exact_identity_and_rejects_fuzzy_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnosis_root = _write_diagnosis_artifact(tmp_path / "diagnosis")
    snapshot_root = _write_snapshot_root(tmp_path / "snapshots")
    expected_sha = _read_json(diagnosis_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(backfill_app, "EXPECTED_DIAGNOSIS_ARTIFACT_SHA", expected_sha)

    manifest = run_historical_exact_semantic_backfill(
        diagnosis_root=diagnosis_root,
        market_root=diagnosis_root,
        snapshot_roots=(snapshot_root,),
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert manifest["ARTIFACT_SHA"] == artifact_sha(manifest)
    assert manifest["TARGET_EVENTS"] == 8
    assert manifest["FUTURE_EVENTS_IN_TARGET"] == 0
    assert manifest["SNAPSHOT_MATCHED_EXACT"] == 5
    assert manifest["SNAPSHOT_IDENTITY_UNRESOLVED"] == 3
    assert manifest["PUBLICATION_MATERIAL_AVAILABLE"] == 4
    assert manifest["SEMANTIC_EXTRACTION_SUCCEEDED"] == 4
    assert manifest["SEMANTIC_EXTRACTION_FAILED"] == 4
    assert manifest["ANALYZER_PRODUCED_UNKNOWN"] == 1
    assert manifest["FEATURE_READY_RECOVERED"] == 3
    assert manifest["FEATURE_READY_STILL_BLOCKED"] == 5
    assert manifest["FEATURE_READY_BEFORE"] == 564
    assert manifest["FEATURE_READY_AFTER"] == 567
    assert manifest["NETWORK_MARKET_FETCHES"] == 0
    assert manifest["REACTION_ROWS_CHANGED"] == 0
    assert manifest["REACTION_ROWS_SHA_BEFORE"] == manifest["REACTION_ROWS_SHA_AFTER"]
    assert manifest["RULES_V3_FINGERPRINT"] == EXPECTED_RULES_V3_FINGERPRINT
    assert manifest["DECISION"] == "SEMANTIC_BACKFILL_PARTIALLY_RECOVERED"
    assert manifest["PER_SOURCE_FAMILY"]["MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"] == {
        "FEATURE_READY": 1,
        "MATCHED": 1,
        "SEMANTIC_READY": 1,
        "TARGET": 1,
        "UNKNOWN_COUNT": 0,
    }
    assert manifest["PER_SOURCE_FAMILY"]["ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1"] == {
        "FEATURE_READY": 2,
        "MATCHED": 4,
        "SEMANTIC_READY": 3,
        "TARGET": 7,
        "UNKNOWN_COUNT": 1,
    }
    assert manifest["PER_TICKER"]["MOEX"]["FEATURE_READY"] == 1
    assert manifest["PER_TICKER"]["GOOD"]["FEATURE_READY"] == 1

    output_root = tmp_path / "output"
    cohort = {row["event_id"]: row for row in _read_jsonl(output_root / "target-cohort.jsonl")}
    assert "future-holdout" not in cohort
    assert set(cohort) == {
        "good",
        "ticker-only",
        "timestamp-near",
        "ambiguous",
        "missing-material",
        "unknown",
        "moex",
        "incomplete-market",
    }

    matches = {row["event_id"]: row for row in _read_jsonl(output_root / "snapshot-matches.jsonl")}
    assert (
        matches["good"]["match_method"] == "RAW_PUBLICATION_SNAPSHOT_SOURCE_ID_AND_SOURCE_ITEM_ID"
    )
    assert matches["good"]["match_confidence"] == "EXACT_IDENTITY_ONLY"
    assert matches["moex"]["match_method"] == "FEED_XML_SOURCE_ID_AND_LINK_OR_GUID"
    assert matches["moex"]["match_confidence"] == "EXACT_IDENTITY_ONLY"
    assert (
        matches["ticker-only"]["primary_blocker"]
        == SemanticBackfillBlocker.SNAPSHOT_IDENTITY_UNRESOLVED
    )
    assert (
        matches["timestamp-near"]["primary_blocker"]
        == SemanticBackfillBlocker.SNAPSHOT_IDENTITY_UNRESOLVED
    )
    assert (
        matches["ambiguous"]["primary_blocker"]
        == SemanticBackfillBlocker.SNAPSHOT_IDENTITY_UNRESOLVED
    )

    material = {
        row["event_id"]: row
        for row in _read_jsonl(output_root / "semantic-material-provenance.jsonl")
    }
    assert material["missing-material"]["publication_material_available"] is False
    assert material["unknown"]["publication_material_available"] is True
    assert material["unknown"]["semantic_input_scope"] == "PUBLICATION_MATERIAL_ONLY"

    semantic = {
        row["event_id"]: row
        for row in _read_jsonl(output_root / "semantic-extraction-results.jsonl")
    }
    assert semantic["good"]["semantic_features"]["primary_event_type"] == "DIVIDEND"
    assert semantic["unknown"]["semantic_features"] == {
        "event_count": 0,
        "fact_count": 0,
        "primary_event_type": "UNKNOWN",
    }
    assert semantic["unknown"]["semantic_input_scope"] == "PUBLICATION_MATERIAL_ONLY"
    assert semantic["missing-material"]["semantic_features"] is None
    assert semantic["missing-material"]["primary_blocker"] == "PUBLICATION_MATERIAL_MISSING"
    assert semantic["ticker-only"]["primary_blocker"] == "PUBLICATION_SNAPSHOT_IDENTITY_UNRESOLVED"

    readiness = {
        row["event_id"]: row for row in _read_jsonl(output_root / "feature-readiness-results.jsonl")
    }
    assert readiness["good"]["feature_ready_after"] is True
    assert readiness["unknown"]["feature_ready_after"] is True
    assert readiness["moex"]["feature_ready_after"] is True
    assert readiness["missing-material"]["feature_ready_after"] is False
    assert readiness["incomplete-market"]["feature_ready_after"] is False
    assert readiness["incomplete-market"]["primary_blocker"] == "MARKET_FEATURES_INCOMPLETE"
    assert all(row["reaction_changed"] is False for row in readiness.values())
    assert all(row["network_market_fetch_performed"] is False for row in readiness.values())
    assert all(row["uses_market_data_for_semantics"] is False for row in readiness.values())
    assert all(row["uses_reaction_data_for_semantics"] is False for row in readiness.values())
    assert all(row["uses_target_data_for_semantics"] is False for row in readiness.values())

    events = {row["metadata"]["event_id"]: row for row in _read_jsonl(output_root / "events.jsonl")}
    assert events["good"]["target_availability"]["feature_ready"] is True
    assert events["ticker-only"]["event_features"] is None
    assert events["missing-material"]["event_features"] is None
    assert events["unknown"]["event_features"]["primary_event_type"] == "UNKNOWN"
    assert events["future-holdout"]["event_features"] is None

    features = _read_jsonl(output_root / "features.jsonl")
    assert [row["event_id"] for row in features] == ["old-ready", "good", "moex", "unknown"]
    assert len({row["event_id"] for row in features}) == len(features)
    assert features[1]["market_features"] == _market_features("2026-07-20T10:00:30+00:00")
    assert _read_jsonl(output_root / "targets.jsonl") == _read_jsonl(
        diagnosis_root / "targets.jsonl"
    )


def test_semantic_backfill_hashes_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnosis_root = _write_diagnosis_artifact(tmp_path / "diagnosis")
    snapshot_root = _write_snapshot_root(tmp_path / "snapshots")
    expected_sha = _read_json(diagnosis_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(backfill_app, "EXPECTED_DIAGNOSIS_ARTIFACT_SHA", expected_sha)

    left = run_historical_exact_semantic_backfill(
        diagnosis_root=diagnosis_root,
        market_root=diagnosis_root,
        snapshot_roots=(snapshot_root,),
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    right = run_historical_exact_semantic_backfill(
        diagnosis_root=diagnosis_root,
        market_root=diagnosis_root,
        snapshot_roots=(snapshot_root,),
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="0" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert left["TARGET_COHORT_SHA"] == right["TARGET_COHORT_SHA"]
    assert left["SNAPSHOT_MATCH_SHA"] == right["SNAPSHOT_MATCH_SHA"]
    assert left["SEMANTIC_MATERIAL_PROVENANCE_SHA"] == right["SEMANTIC_MATERIAL_PROVENANCE_SHA"]
    assert left["SEMANTIC_EXTRACTION_RESULT_SHA"] == right["SEMANTIC_EXTRACTION_RESULT_SHA"]
    assert left["FEATURE_READINESS_RESULT_SHA"] == right["FEATURE_READINESS_RESULT_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def test_rules_v3_fingerprint_is_frozen() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_V3_FINGERPRINT


def _write_diagnosis_artifact(root: Path) -> Path:
    events = [
        _event("old-ready", "OLD", "old-source", "old-item", feature_ready=True),
        _event("good", "GOOD", "good-source", "GOOD:https://official.example/good"),
        _event(
            "ticker-only",
            "TICK",
            "ticker-source",
            "TICK:https://official.example/expected",
            extra={"leaky_reaction_text": "dividends of 99 rub per share"},
        ),
        _event("timestamp-near", "NEAR", "near-source", "NEAR:https://official.example/expected"),
        _event(
            "ambiguous",
            "AMB",
            "AMB_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "AMB:https://official.example/amb",
        ),
        _event("missing-material", "MISS", "miss-source", "MISS:https://official.example/miss"),
        _event(
            "unknown",
            "UNKN",
            "unknown-source",
            "UNKN:https://official.example/unknown",
            extra={"leaky_target_text": "dividends of 42 rub per share"},
        ),
        _event(
            "moex",
            "MOEX",
            "MOEX_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "MOEX:https://www.moex.com/n777",
            source_family="MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        ),
        _event(
            "incomplete-market",
            "INC",
            "inc-source",
            "INC:https://official.example/inc",
            market_features=_market_features("2026-07-20T10:00:30+00:00", complete=False),
        ),
        _event(
            "future-holdout",
            "FUTR",
            "future-source",
            "FUTR:https://official.example/future",
            published_at="2026-08-12T10:00:30+00:00",
        ),
    ]
    targets = [
        _target_row("good"),
        _target_row("ticker-only"),
        _target_row("timestamp-near"),
        _target_row("ambiguous"),
        _target_row("missing-material"),
        _target_row("unknown"),
        _target_row("moex"),
        _target_row("incomplete-market"),
        _target_row("future-holdout"),
    ]
    _write_jsonl(root / "input-target-cohort.jsonl", targets)
    _write_jsonl(root / "events.jsonl", events)
    _write_jsonl(root / "features.jsonl", [{"event_id": "old-ready", "marker": "preserved"}])
    _write_jsonl(root / "targets.jsonl", [_reaction_row(row["event_id"]) for row in targets])
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "exact-feature-readiness-recovery-v1",
        "FEATURE_READY_AFTER": 564,
        "FEATURE_READY_RECOVERED": 0,
        "FEATURE_READY_STILL_BLOCKED": 8,
        "SEMANTIC_EVENT_FEATURES_PRESENT": 0,
        "SEMANTIC_EVENT_FEATURES_MISSING": 8,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
    }
    manifest["ARTIFACT_SHA"] = diagnosis_artifact_sha(manifest)
    _write_json(root / "manifest.json", manifest)
    return root


def _write_snapshot_root(root: Path) -> Path:
    raw_rows = [
        _raw_snapshot(
            "good-source",
            "GOOD:https://official.example/good",
            title="Board recommends dividends of 12 rub per share for 2025.",
            description="The board recommends dividends of 12 rub per share for 2025.",
        ),
        _raw_snapshot("miss-source", "MISS:https://official.example/miss"),
        _raw_snapshot(
            "unknown-source",
            "UNKN:https://official.example/unknown",
            title="Regular exchange bulletin.",
        ),
        _raw_snapshot(
            "inc-source",
            "INC:https://official.example/inc",
            title="Board recommends dividends of 3 rub per share.",
        ),
    ]
    _write_jsonl(root / "raw-publication-snapshots.jsonl", raw_rows)
    _write_xml(
        root / "raw-snapshots" / "ticker-source.xml",
        [
            _xml_item(
                "TICK headline mentions ticker",
                "https://official.example/not-expected",
                "Mon, 20 Jul 2026 13:00:30 +0300",
                "TICK unrelated description",
                "TICK unrelated content",
            )
        ],
    )
    _write_xml(
        root / "raw-snapshots" / "near-source.xml",
        [
            _xml_item(
                "Near timestamp item",
                "https://official.example/other",
                "Mon, 20 Jul 2026 13:00:30 +0300",
                "Same timestamp, different identity.",
                "Same timestamp, different identity.",
            )
        ],
    )
    _write_xml(
        root / "raw-snapshots" / "AMB_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1.xml",
        [
            _xml_item(
                "Ambiguous one",
                "https://official.example/amb",
                "Mon, 20 Jul 2026 13:00:30 +0300",
                "First exact identity.",
                "First exact identity.",
            ),
            _xml_item(
                "Ambiguous two",
                "https://official.example/amb",
                "Mon, 20 Jul 2026 13:00:31 +0300",
                "Second exact identity.",
                "Second exact identity.",
            ),
        ],
    )
    _write_xml(
        root / "raw-snapshots" / "MOEX_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1.xml",
        [
            _xml_item(
                "Risk parameter update",
                "https://www.moex.com/n777",
                "Mon, 20 Jul 2026 13:00:30 +0300",
                "Exchange risk parameters changed.",
                "Board recommends dividends of 7 rub per share.",
            )
        ],
    )
    return root


def _event(
    event_id: str,
    ticker: str,
    source_id: str,
    source_item_id: str,
    *,
    published_at: str = "2026-07-20T10:00:30+00:00",
    source_family: str = "ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1",
    market_features: dict[str, object] | None = None,
    feature_ready: bool = False,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    features = _market_features(published_at) if market_features is None else market_features
    metadata: dict[str, object] = {
        "event_id": event_id,
        "ticker": ticker,
        "source_id": source_id,
        "source_family": source_family,
        "source_item_id": source_item_id,
        "publication_timestamp_utc": published_at,
        "publication_date": published_at[:10],
        "publication_time": "10:00:30",
        "timestamp_quality": "EXACT",
        "session_state": "DURING_MAIN_SESSION",
    }
    if extra is not None:
        metadata.update(extra)
    return {
        "metadata": metadata,
        "event_features": (
            {"primary_event_type": "OTHER", "event_count": 1, "fact_count": 0}
            if feature_ready
            else None
        ),
        "pre_event_market_features": features,
        "target_availability": {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": feature_ready,
            "status": "REACTION_READY" if feature_ready else "METADATA_ONLY",
            "missing_reason": None if feature_ready else "EVENT_FEATURES_MISSING",
        },
        "quality": {"feature_cutoff": published_at},
    }


def _market_features(published_at: str, *, complete: bool = True) -> dict[str, object]:
    return {
        "feature_cutoff": published_at,
        "post_event_values_in_features": False,
        "pre_return_5m": "0.01",
        "pre_return_15m": "0.02",
        "pre_return_30m": "0.03",
        "pre_return_60m": "0.04" if complete else None,
        "imoex_pre_return_5m": "0.001",
        "imoex_pre_return_15m": "0.002",
        "imoex_pre_return_30m": "0.003",
        "imoex_pre_return_60m": "0.004" if complete else None,
    }


def _target_row(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ticker": event_id.upper(),
        "source_id": "unused-target-source",
        "source_family": "unused-target-family",
        "published_at_utc": "2026-07-20T10:00:30+00:00",
        "reaction_ready": True,
        "feature_ready": False,
        "existing_primary_blocker": "EVENT_FEATURES_MISSING",
    }


def _reaction_row(event_id: object) -> dict[str, object]:
    return {
        "event_id": str(event_id),
        "horizon": "1m",
        "abnormal_return": "0.01",
        "leaky_target_text": "dividends of 100 rub per share",
    }


def _raw_snapshot(
    source_id: str,
    source_item_id: str,
    *,
    title: str = "",
    description: str = "",
    content: str = "",
) -> dict[str, object]:
    payload = {
        "source_id": source_id,
        "source_item_id": source_item_id,
        "title": title,
        "description": description,
        "content": content,
        "link": source_item_id.split(":", 1)[-1],
        "guid": source_item_id.split(":", 1)[-1],
    }
    return {
        **payload,
        "snapshot_id": f"snapshot-{source_id}-{source_item_id}",
        "publication_material_available": any((title, description, content)),
    }


def _xml_item(
    title: str,
    link: str,
    pub_date: str,
    description: str,
    content: str,
) -> dict[str, str]:
    return {
        "title": title,
        "link": link,
        "pubDate": pub_date,
        "description": description,
        "content": content,
    }


def _write_xml(path: Path, items: Sequence[dict[str, str]]) -> None:
    body = "\n".join(
        f"""
        <item>
            <title>{item["title"]}</title>
            <link>{item["link"]}</link>
            <pubDate>{item["pubDate"]}</pubDate>
            <description>{item["description"]}</description>
            <content:encoded>{item["content"]}</content:encoded>
            <guid>{item["link"]}</guid>
        </item>
        """
        for item in items
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
    <channel>{body}</channel>
</rss>
""",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
