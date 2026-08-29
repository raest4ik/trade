from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import src.exact_event_live_source_breadth_expansion_v2.application as v2_app
from apps.cli.expand_exact_event_live_source_breadth_v2 import build_parser
from src.exact_event_live_official_collection.http_client import FetchResult
from src.exact_event_live_source_breadth_expansion_v2.domain import (
    ARTIFACT_VERSION,
    EXPECTED_V1_ARTIFACT_SHA,
    sha256_payload,
)


def test_cli_defaults_to_v2_artifact_paths() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])

    assert args.v1_artifact_root == "artifacts/exact-event-live-source-breadth-expansion-v1"
    assert args.base_events == "artifacts/chep-historical-exact-maturation-v1/events.jsonl"
    assert args.live_registry == "config/exact-event-live-official-sources.json"
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"


def test_v2_selects_next_five_excludes_existing_and_replays_dedupe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    _patch_expected_candidate_sha(monkeypatch, paths)
    manifest = v2_app.build_live_source_breadth_expansion_v2_artifact(
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="f" * 40,
        git_sha="e" * 40,
        v1_artifact_root=paths["v1"],
        base_events_path=paths["base_events"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert manifest["CANDIDATES_AVAILABLE"] == 7
    assert manifest["CANDIDATES_ATTEMPTED"] == 5
    assert manifest["NEW_EXACT_LIVE_SOURCES"] == 5
    assert manifest["NEW_TICKERS_WITH_EXACT_SOURCE"] == ["AQUA", "BELU", "BSPB", "CBOM", "DATA"]
    assert manifest["V1_CANDIDATES_REMAINING_AFTER_BATCH"] == 2
    assert manifest["ITEMS_FETCHED"] == 5
    assert manifest["ITEMS_NEW"] == 5
    assert manifest["ITEMS_TIMESTAMP_INVALID"] == 0
    assert manifest["NEW_HISTORICAL_EXACT_EVENTS"] == 0
    assert manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"] == 5
    assert manifest["REPLAY_ITEMS_NEW"] == 0
    assert manifest["REPLAY_ITEMS_DUPLICATE"] == 5
    assert manifest["FINAL_DECISION"] == "SOURCE_BREADTH_GAINED"
    assert manifest["MARKET_MATURATION_INVOKED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["UNRELATED_ITEMS_REJECTED"] > 0

    selected = _read_jsonl(tmp_path / ARTIFACT_VERSION / "selected-source-cohort.jsonl")
    assert [row["ticker"] for row in selected] == ["AQUA", "BELU", "BSPB", "CBOM", "DATA"]
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    assert [row["ticker"] for row in registry["sources"]] == [
        "CHEP",
        "OZON",
        "ASTR",
        "ELMT",
        "RUAL",
        "AFKS",
        "AQUA",
        "BELU",
        "BSPB",
        "CBOM",
        "DATA",
    ]

    rows = _read_jsonl(
        tmp_path / ARTIFACT_VERSION / "live-collection" / "collected-event-metadata.jsonl"
    )
    assert {row["metadata"]["ticker"] for row in rows} == {"AQUA", "BELU", "BSPB", "CBOM", "DATA"}
    assert all(row["metadata"]["future_holdout"] is True for row in rows)
    assert all(row["pre_event_market_features"] is None for row in rows)


def test_v2_rejects_non_ready_and_non_tradable_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path, include_blocked=True)
    _patch_expected_candidate_sha(monkeypatch, paths)
    manifest = v2_app.build_live_source_breadth_expansion_v2_artifact(
        output_root=tmp_path / "blocked",
        base_main_sha="f" * 40,
        git_sha="e" * 40,
        v1_artifact_root=paths["v1"],
        base_events_path=paths["base_events"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
        write_registry=False,
    )

    selected = _read_jsonl(tmp_path / "blocked" / "selected-source-cohort.jsonl")
    assert "SKIP" not in {row["ticker"] for row in selected}
    assert manifest["NEW_EXACT_LIVE_SOURCES"] == 5


def test_v2_hashes_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_fixture(tmp_path)
    _patch_expected_candidate_sha(monkeypatch, paths)
    left = v2_app.build_live_source_breadth_expansion_v2_artifact(
        output_root=tmp_path / "left",
        base_main_sha="f" * 40,
        git_sha="e" * 40,
        v1_artifact_root=paths["v1"],
        base_events_path=paths["base_events"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
        write_registry=False,
    )
    right = v2_app.build_live_source_breadth_expansion_v2_artifact(
        output_root=tmp_path / "right",
        base_main_sha="f" * 40,
        git_sha="d" * 40,
        v1_artifact_root=paths["v1"],
        base_events_path=paths["base_events"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        write_registry=False,
    )

    assert left["INPUT_CANDIDATE_SET_SHA"] == paths["candidate_sha"]
    assert left["SELECTED_SOURCE_COHORT_SHA"] == right["SELECTED_SOURCE_COHORT_SHA"]
    assert left["SOURCE_VALIDATION_SHA"] == right["SOURCE_VALIDATION_SHA"]
    assert left["COLLECTION_RESULT_SHA"] == right["COLLECTION_RESULT_SHA"]
    assert left["REPLAY_RESULT_SHA"] == right["REPLAY_RESULT_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})


def _write_fixture(tmp_path: Path, *, include_blocked: bool = False) -> dict[str, Any]:
    v1 = tmp_path / "v1"
    v1.mkdir()
    candidates = [_candidate(ticker) for ticker in ["OZON", "ASTR", "ELMT", "RUAL", "AFKS"]]
    candidates.extend(_candidate(ticker) for ticker in ["AQUA", "BELU", "BSPB", "CBOM", "DATA"])
    candidates.extend(_candidate(ticker) for ticker in ["DIAS", "FEES"])
    if include_blocked:
        candidates.insert(5, {**_candidate("SKIP"), "status": "CURRENTLY_NON_TRADABLE"})
    _write_jsonl(v1 / "source-candidates.jsonl", candidates)
    actual_sha = sha256_payload(candidates)
    _write_json(
        v1 / "manifest.json",
        {
            "ARTIFACT_SHA": EXPECTED_V1_ARTIFACT_SHA,
            "SOURCE_CANDIDATES_SHA": actual_sha,
            "NEW_EXACT_LIVE_SOURCES": 5,
            "NEW_TICKERS_WITH_EXACT_SOURCE": ["AFKS", "ASTR", "ELMT", "OZON", "RUAL"],
            "NEW_CANONICAL_EXACT_EVENTS": 56,
            "NEW_HISTORICAL_EXACT_EVENTS": 45,
            "NEW_FUTURE_METADATA_ONLY_EVENTS": 11,
            "REPLAY_ITEMS_NEW": 0,
            "REPLAY_ITEMS_DUPLICATE": 56,
            "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
            "CANONICAL_EXACT_EVENTS_TOTAL_AFTER": 817,
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER": 685,
        },
    )
    _write_jsonl(v1 / "target-universe.jsonl", [{"ticker": row["ticker"]} for row in candidates])
    collection = v1 / "live-collection"
    collection.mkdir()
    _write_jsonl(collection / "collected-event-metadata.jsonl", [_event("OZON", future=True)])
    base_events = tmp_path / "base-events.jsonl"
    _write_jsonl(base_events, [_event("MGNT", future=False)])
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "source_registry_version": "exact-event-live-official-source-registry-v1",
            "sources": [
                _registry_source(ticker)
                for ticker in ["CHEP", "OZON", "ASTR", "ELMT", "RUAL", "AFKS"]
            ],
        },
    )
    return {
        "v1": v1,
        "base_events": base_events,
        "registry": registry,
        "candidate_sha": actual_sha,
    }


def _patch_expected_candidate_sha(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Any]) -> None:
    monkeypatch.setattr(v2_app, "EXPECTED_V1_CANDIDATE_SET_SHA", paths["candidate_sha"])


def _candidate(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": f"uid-{ticker}",
        "status": "EXACT_LIVE_READY",
        "official_domain": "www.moex.com",
        "source_url": "https://www.moex.com/export/news.aspx?cat=122",
        "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        "mechanism_type": "RSS",
        "timestamp_field": "RSS item pubDate",
        "timestamp_policy": "Require item-level RSS pubDate with explicit +0300.",
        "item_match_any": [ticker],
        "items_matched": 1,
        "exact_event_count_before": 0,
        "already_in_live_registry": False,
    }


def _registry_source(ticker: str) -> dict[str, Any]:
    domain = "chtpz.tmk-group.ru" if ticker == "CHEP" else "www.moex.com"
    return {
        "source_registry_version": "exact-event-live-official-source-registry-v1",
        "source_id": f"{ticker}_SOURCE",
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": f"uid-{ticker}",
        "source_family": f"{ticker}_SOURCE",
        "source_url": f"https://{domain}/rss",
        "official_domain": domain,
        "mechanism_type": "RSS",
        "timestamp_field": "RSS item pubDate",
        "timestamp_policy": "Require item-level RSS pubDate with explicit +0300.",
        "archive_capability": False,
        "live_capability": True,
        "provenance_evidence_url": "artifact#source",
        "provenance_evidence_sha": "sha",
        "enabled": True,
        "parser_version": "rss-item-pubdate-exact-v1",
        "item_match_any": [ticker] if ticker != "CHEP" else [],
    }


def _event(ticker: str, *, future: bool) -> dict[str, Any]:
    date = "2026-08-25" if future else "2026-07-01"
    timestamp = f"{date}T08:44:16+00:00"
    return {
        "metadata": {
            "event_id": f"event-{ticker}-{date}",
            "source_code": f"{ticker}_SOURCE",
            "source_item_id": f"{ticker}:{date}",
            "canonical_url": f"https://example.com/{ticker}/{date}",
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_raw": "Tue, 25 Aug 2026 11:44:16 +0300",
            "publication_timestamp_utc": timestamp,
            "publication_date": date,
            "publication_time": "08:44:16",
            "publication_timezone": "UTC",
            "timestamp_source_field": "RSS item pubDate",
            "timestamp_quality": "EXACT",
            "future_holdout": future,
        },
        "event_features": {},
        "pre_event_market_features": None if future else {},
        "target_availability": {
            "feature_ready": False,
            "reaction_ready": False,
            "research_outcomes_visible": False,
        },
        "quality": {},
    }


def _rss_body() -> bytes:
    items: list[str] = []
    for ticker in ["AQUA", "BELU", "BSPB", "CBOM", "DATA", "DIAS", "FEES"]:
        items.append(
            f"""
            <item>
              <title>Risk parameters changed for {ticker}</title>
              <link>https://www.moex.com/n-{ticker}</link>
              <guid>https://www.moex.com/n-{ticker}</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        )
    items.append(
        """
        <item>
          <title>Risk parameters changed for OTHER</title>
          <link>https://www.moex.com/n-other</link>
          <guid>https://www.moex.com/n-other</guid>
          <pubDate>2026-08-25</pubDate>
        </item>
        """
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>{"".join(items)}</channel></rss>
    """.encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _FakeClient:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def get(self, url: str) -> FetchResult:
        return FetchResult(
            request_url=url,
            final_url=url,
            status=200,
            content_type="application/rss+xml",
            body=self.body,
            redirects=0,
            redirect_chain=(),
            blocker=None,
        )
