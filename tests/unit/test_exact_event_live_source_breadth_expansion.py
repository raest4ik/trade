from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.cli.expand_exact_event_live_source_breadth import build_parser
from src.exact_event_live_official_collection.http_client import FetchResult
from src.exact_event_live_source_breadth_expansion.application import (
    build_live_source_breadth_expansion_artifact,
)
from src.exact_event_live_source_breadth_expansion.domain import (
    ARTIFACT_VERSION,
    CandidateStatus,
    sha256_payload,
)


def test_cli_defaults_to_live_source_breadth_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])

    assert args.universe == "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
    assert args.input_events == "artifacts/chep-historical-exact-maturation-v1/events.jsonl"
    assert (
        args.eligibility_manifest
        == "artifacts/exact-event-security-tradability-eligibility-v1/manifest.json"
    )
    assert args.live_registry == "config/exact-event-live-official-sources.json"
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"


def test_build_discovers_only_tradable_unambiguous_exact_live_sources_and_replays_dedupe(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    manifest = build_live_source_breadth_expansion_artifact(
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="a57b159ad6ecd81b95c558e4ecfa5c15e433d33c",
        git_sha="b" * 40,
        universe_path=paths["universe"],
        input_events_path=paths["events"],
        eligibility_manifest_path=paths["eligibility"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert manifest["TARGET_TICKERS"] == 3
    assert manifest["TRADABLE_TARGET_TICKERS"] == 1
    assert manifest["NON_TRADABLE_SKIPPED"] == 1
    assert manifest["IDENTITY_BLOCKED"] == 1
    assert manifest["NEW_EXACT_LIVE_SOURCES"] == 1
    assert manifest["NEW_TICKERS_WITH_EXACT_SOURCE"] == ["AAA"]
    assert manifest["NEW_CANONICAL_EXACT_EVENTS"] == 1
    assert manifest["NEW_HISTORICAL_EXACT_EVENTS"] == 0
    assert manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"] == 1
    assert manifest["REPLAY_ITEMS_NEW"] == 0
    assert manifest["REPLAY_ITEMS_DUPLICATE"] == 1
    assert manifest["FINAL_DECISION"] == "SOURCE_BREADTH_GAINED"
    assert manifest["MARKET_MATURATION_INVOKED"] is False
    assert manifest["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert manifest["DATE_ONLY_COERCIONS"] == 0

    target_rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "target-universe.jsonl")
    statuses = {row["ticker"]: row["target_selection_status"] for row in target_rows}
    assert statuses == {
        "AAA": "TRADABLE_TARGET",
        "BBB": CandidateStatus.CURRENTLY_NON_TRADABLE.value,
        "DUP": CandidateStatus.IDENTITY_AMBIGUOUS.value,
    }

    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    source = registry["sources"][0]
    assert source["source_id"] == "AAA_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
    assert source["item_match_any"] == ["AAA"]
    assert source["enabled"] is True

    events = _read_jsonl(
        tmp_path / ARTIFACT_VERSION / "live-collection" / "collected-event-metadata.jsonl"
    )
    assert events[0]["metadata"]["ticker"] == "AAA"
    assert events[0]["metadata"]["future_holdout"] is True
    assert events[0]["pre_event_market_features"] is None
    assert events[0]["target_availability"]["research_outcomes_visible"] is False


def test_artifact_hash_inputs_are_stable(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    left = build_live_source_breadth_expansion_artifact(
        output_root=tmp_path / "left",
        base_main_sha="a57b159ad6ecd81b95c558e4ecfa5c15e433d33c",
        git_sha="b" * 40,
        universe_path=paths["universe"],
        input_events_path=paths["events"],
        eligibility_manifest_path=paths["eligibility"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
        write_registry=False,
    )
    right = build_live_source_breadth_expansion_artifact(
        output_root=tmp_path / "right",
        base_main_sha="a57b159ad6ecd81b95c558e4ecfa5c15e433d33c",
        git_sha="c" * 40,
        universe_path=paths["universe"],
        input_events_path=paths["events"],
        eligibility_manifest_path=paths["eligibility"],
        live_registry_path=paths["registry"],
        client=_FakeClient(_rss_body()),
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        write_registry=False,
    )

    assert len(str(left["ARTIFACT_SHA"])) == 64
    assert left["TARGET_UNIVERSE_SHA"] == right["TARGET_UNIVERSE_SHA"]
    assert left["SOURCE_CANDIDATES_SHA"] == right["SOURCE_CANDIDATES_SHA"]
    assert left["COLLECTION_RESULT_SHA"] == right["COLLECTION_RESULT_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    universe = tmp_path / "universe.json"
    _write_json(
        universe,
        {
            "instruments": [
                _instrument("AAA", "AAA Issuer", "moex_mrng_evng_e_wknd_dlr"),
                _instrument("BBB", "BBB Issuer", "unknown"),
                _instrument("DUP", "Duplicate One", "moex_mrng_evng_e_wknd_dlr"),
                _instrument("DUP", "Duplicate Two", "moex_mrng_evng_e_wknd_dlr"),
            ]
        },
    )
    events = tmp_path / "events.jsonl"
    _write_jsonl(events, [_existing_event("MGNT")])
    eligibility = tmp_path / "eligibility.json"
    _write_json(
        eligibility,
        {
            "ARTIFACT_SHA": "106a32fbc732b6e0813827993d60af80d75a53dcea06a36a208ffb0d04f1669d",
            "CANONICAL_EXACT_EVENTS_TOTAL": 761,
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS": 685,
            "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS": 44,
            "REACTION_READY_EVENTS": 565,
            "FEATURE_READY_EVENTS": 564,
            "FINAL_DECISION": "SOURCE_BREADTH_EXPANSION_NEXT",
        },
    )
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "source_registry_version": "exact-event-live-official-source-registry-v1",
            "sources": [],
        },
    )
    return {
        "universe": universe,
        "events": events,
        "eligibility": eligibility,
        "registry": registry,
    }


def _instrument(ticker: str, name: str, exchange: str) -> dict[str, str]:
    return {
        "ticker": ticker,
        "name": name,
        "instrument_uid": f"uid-{ticker}",
        "figi": f"figi-{ticker}",
        "class_code": "TQBR",
        "currency": "rub",
        "exchange": exchange,
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
    }


def _existing_event(ticker: str) -> dict[str, Any]:
    return {
        "metadata": {
            "event_id": "existing",
            "source_code": "MAGNIT_OFFICIAL_JSON_EXACT",
            "source_item_id": "existing",
            "canonical_url": "https://example.com/existing",
            "ticker": ticker,
            "issuer": ticker,
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": "2026-07-01T07:00:00+00:00",
            "publication_timestamp_raw": "Wed, 01 Jul 2026 10:00:00 +0300",
            "publication_date": "2026-07-01",
            "publication_time": "07:00:00",
            "publication_timezone": "UTC",
            "timestamp_source_field": "fixture",
            "timestamp_quality": "EXACT",
        },
        "event_features": {},
        "pre_event_market_features": {},
        "target_availability": {"feature_ready": True, "research_outcomes_visible": True},
        "quality": {},
    }


def _rss_body() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Risk parameters changed for AAA</title>
          <link>https://www.moex.com/n1</link>
          <guid>https://www.moex.com/n1</guid>
          <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
        </item>
        <item>
          <title>Risk parameters changed for BBB</title>
          <link>https://www.moex.com/n2</link>
          <guid>https://www.moex.com/n2</guid>
          <pubDate>Tue, 25 Aug 2026 12:44:16 +0300</pubDate>
        </item>
        <item>
          <title>Risk parameters changed for DUP</title>
          <link>https://www.moex.com/n3</link>
          <guid>https://www.moex.com/n3</guid>
          <pubDate>Tue, 25 Aug 2026 13:44:16 +0300</pubDate>
        </item>
        <item>
          <title>Date-only item for ZZZ</title>
          <link>https://www.moex.com/n4</link>
          <guid>https://www.moex.com/n4</guid>
          <pubDate>2026-08-25</pubDate>
        </item>
      </channel>
    </rss>
    """


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
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
