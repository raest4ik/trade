from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from apps.cli.acquire_exact_event_live_official import build_parser
from src.event_predictive_baseline.domain import guard_future_holdout_outcome_read
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_live_official_collection.application import (
    build_live_official_collection_artifact,
)
from src.exact_event_live_official_collection.domain import (
    ARTIFACT_VERSION,
    SourceStatus,
    parse_publication_timestamp_exact,
    parse_rss_pubdate_exact,
    publication_material,
    publication_material_sha,
    sha256_payload,
)
from src.exact_event_live_official_collection.http_client import FetchResult


def test_cli_defaults_to_registry_and_v5_events() -> None:
    args = build_parser().parse_args(["--base-main-sha", "5" * 40])

    assert args.source_registry == "config/exact-event-live-official-sources.json"
    assert args.input_events == "artifacts/exact-event-official-source-discovery-v5/events.jsonl"
    assert args.output_dir == "artifacts/exact-event-live-official-collection-v1"


def test_valid_rss_pubdate_normalizes_to_utc_and_metadata_only(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>CHEP recommends dividends of 12 rub per share</title>
              <description>CHEP recommends dividends of 12 rub per share</description>
              <content:encoded><![CDATA[
                CHEP board recommends dividends of 12 rub per share.
              ]]></content:encoded>
              <link>https://chtpz.tmk-group.ru/news/1</link>
              <guid>chep-1</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        ),
    )

    assert manifest["ITEMS_FETCHED"] == 1
    assert manifest["ITEMS_TIMESTAMP_VALID"] == 1
    assert manifest["ITEMS_WITH_PUBLICATION_MATERIAL"] == 1
    assert manifest["ITEMS_WITHOUT_PUBLICATION_MATERIAL"] == 0
    assert manifest["ITEMS_NEW"] == 1
    assert manifest["SNAPSHOTS_WRITTEN"] == 1
    assert manifest["DUPLICATE_SNAPSHOTS"] == 0
    assert manifest["RAW_PUBLICATION_PRESERVATION_ENABLED"] is True
    assert manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"] == 1
    event = _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl")[0]
    snapshot = _read_jsonl(tmp_path / ARTIFACT_VERSION / "raw-publication-snapshots.jsonl")[0]
    semantic = _read_jsonl(tmp_path / ARTIFACT_VERSION / "semantic-material-provenance.jsonl")[0]
    extracted = _read_jsonl(tmp_path / ARTIFACT_VERSION / "semantic-extraction-results.jsonl")[0]
    metadata = event["metadata"]
    assert metadata["publication_timestamp_utc"] == "2026-08-25T08:44:16+00:00"
    assert metadata["publication_timestamp_raw"] == "Tue, 25 Aug 2026 11:44:16 +0300"
    assert metadata["timestamp_source_field"] == "RSS item pubDate"
    assert metadata["future_holdout"] is True
    assert metadata["source_family"] == "CHEP_OFFICIAL_RSS_EXACT_LIVE_V1"
    assert metadata["event_origin"] == "ISSUER_ORIGINATED"
    assert metadata["publication_snapshot_id"] == snapshot["snapshot_id"]
    assert metadata["publication_material_available"] is True
    assert metadata["publication_material_sha"] == snapshot["publication_material_sha"]
    assert event["event_features"] == {
        "event_count": 1,
        "fact_count": 3,
        "primary_event_type": "DIVIDEND",
    }
    assert event["pre_event_market_features"] is None
    assert event["target_availability"]["research_outcomes_visible"] is False
    assert snapshot["title"] == "CHEP recommends dividends of 12 rub per share"
    assert snapshot["description"] == "CHEP recommends dividends of 12 rub per share"
    assert snapshot["content"] == "CHEP board recommends dividends of 12 rub per share."
    assert snapshot["publication_timestamp_raw"] == "Tue, 25 Aug 2026 11:44:16 +0300"
    assert snapshot["publication_timestamp_utc"] == "2026-08-25T08:44:16+00:00"
    assert snapshot["link"] == "https://chtpz.tmk-group.ru/news/1"
    assert snapshot["guid"] == "chep-1"
    assert snapshot["source_format"] == "RSS_ITEM"
    assert snapshot["raw_payload"]["pubDate"] == "Tue, 25 Aug 2026 11:44:16 +0300"
    assert snapshot["raw_payload"]["link"] == "https://chtpz.tmk-group.ru/news/1"
    assert snapshot["raw_payload"]["guid"] == "chep-1"
    assert snapshot["publication_material_available"] is True
    snapshot_text = json.dumps(snapshot, ensure_ascii=False)
    assert "Authorization" not in snapshot_text
    assert "Bearer" not in snapshot_text
    assert "token" not in snapshot_text.lower()
    assert publication_material(snapshot) == (
        "CHEP recommends dividends of 12 rub per share\n"
        "CHEP board recommends dividends of 12 rub per share."
    )
    assert publication_material_sha(snapshot) == snapshot["publication_material_sha"]
    assert semantic["publication_material_fields"] == ["title", "description", "content"]
    assert semantic["uses_market_data"] is False
    assert semantic["uses_reaction_data"] is False
    assert semantic["uses_target_data"] is False
    assert extracted["event_features"] == event["event_features"]
    assert extracted["publication_material_sha"] == snapshot["publication_material_sha"]
    assert extracted["semantic_features_sha"] == sha256_payload(event["event_features"])
    assert extracted["semantic_input_fields"] == ["title", "description", "content"]
    assert extracted["uses_market_data"] is False
    assert extracted["uses_reaction_data"] is False
    assert extracted["uses_target_data"] is False
    assert manifest["ITEMS_SEMANTIC_READY"] == 1
    assert manifest["SEMANTIC_READY_EVENTS"] == 1
    assert manifest["ANALYZER_UNKNOWN"] == 0
    assert manifest["TINVEST_REQUESTS"] == 0
    assert manifest["MARKET_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_REACTIONS_COMPUTED"] == 0
    assert manifest["FUTURE_TARGETS_COMPUTED"] == 0
    assert manifest["FUTURE_OUTCOMES_READ"] == 0
    assert manifest["WINDOWS_SCHEDULER_CHANGED"] is False
    assert manifest["BACKGROUND_AUTOMATION_ENABLED"] is False
    analysis = EventAnalyzerV3().analyze(
        news_id=UUID(metadata["event_id"]),
        raw_content=publication_material(snapshot) or "",
    )
    assert analysis.primary_event_type.value == "DIVIDEND"
    assert rules_v3_fingerprint() == (
        "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
    )
    with pytest.raises(Exception, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        guard_future_holdout_outcome_read(datetime(2026, 8, 25, tzinfo=UTC).date(), context="test")


def test_missing_pubdate_fails_closed(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss("<item><title>No timestamp</title><link>https://chtpz.tmk-group.ru/n/1</link></item>"),
    )

    assert manifest["LIVE_EXACT_SOURCES_FAILED"] == 1
    assert manifest["ITEMS_NEW"] == 0
    assert manifest["BLOCKERS_BY_TYPE"] == {SourceStatus.MISSING_EXACT_TIMESTAMP.value: 1}


def test_date_only_pubdate_is_rejected_as_exact(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>Date only</title>
              <link>https://chtpz.tmk-group.ru/n/1</link>
              <pubDate>2026-08-25</pubDate>
            </item>
            """
        ),
    )

    assert manifest["ITEMS_NEW"] == 0
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["BLOCKERS_BY_TYPE"] == {SourceStatus.MISSING_EXACT_TIMESTAMP.value: 1}


def test_invalid_timezone_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=SourceStatus.INVALID_TIMEZONE.value):
        parse_rss_pubdate_exact("Tue, 25 Aug 2026 11:44:16 MSK")

    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>Bad timezone</title>
              <link>https://chtpz.tmk-group.ru/n/1</link>
              <pubDate>Tue, 25 Aug 2026 11:44:16 MSK</pubDate>
            </item>
            """
        ),
    )
    assert manifest["BLOCKERS_BY_TYPE"] == {SourceStatus.INVALID_TIMEZONE.value: 1}


def test_atom_timezone_aware_timestamp_qualifies(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Atom exact published</title>
            <summary>Company announces production update.</summary>
            <id>atom-1</id>
            <link href="https://chtpz.tmk-group.ru/news/atom-1"/>
            <published>2026-09-01T12:41:32+03:00</published>
          </entry>
        </feed>""",
        source_overrides={
            "mechanism_type": "ATOM",
            "timestamp_field": "Atom entry published",
            "timestamp_policy": "Require entry-level Atom published with explicit timezone offset.",
        },
    )

    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl")
    snapshots = _read_jsonl(tmp_path / ARTIFACT_VERSION / "raw-publication-snapshots.jsonl")
    assert manifest["ITEMS_TIMESTAMP_VALID"] == 1
    assert rows[0]["metadata"]["publication_timestamp_utc"] == "2026-09-01T09:41:32+00:00"
    assert rows[0]["metadata"]["timestamp_source_field"] == "Atom entry published"
    assert snapshots[0]["source_format"] == "ATOM_ENTRY"


def test_official_json_iso_offset_qualifies(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        json.dumps(
            {
                "items": [
                    {
                        "id": "json-1",
                        "url": "https://chtpz.tmk-group.ru/news/json-1",
                        "title": "JSON exact",
                        "description": "Company published financial results.",
                        "published_at": "2026-09-01T09:41:32Z",
                    }
                ]
            }
        ).encode(),
        source_overrides={
            "mechanism_type": "JSON",
            "timestamp_field": "JSON published_at",
            "timestamp_policy": (
                "Require item-level JSON published_at with explicit UTC Z or offset."
            ),
        },
    )

    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl")
    snapshots = _read_jsonl(tmp_path / ARTIFACT_VERSION / "raw-publication-snapshots.jsonl")
    assert manifest["ITEMS_TIMESTAMP_VALID"] == 1
    assert rows[0]["metadata"]["publication_timestamp_utc"] == "2026-09-01T09:41:32+00:00"
    assert snapshots[0]["source_format"] == "JSON_ITEM"


def test_html_datepublished_qualifies_and_date_modified_does_not(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        b"""<html><head>
        <link rel="canonical" href="https://chtpz.tmk-group.ru/news/html-1">
        <meta property="article:published_time" content="2026-09-01T12:41:32+03:00">
        <meta property="article:modified_time" content="2026-09-02T12:41:32+03:00">
        </head><body><h1>HTML exact</h1><p>Board elected chief executive.</p></body></html>""",
        source_overrides={
            "mechanism_type": "HTML",
            "timestamp_field": "HTML article:published_time",
            "timestamp_policy": (
                "Require publication-specific HTML article:published_time with explicit offset."
            ),
        },
    )
    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl")
    assert manifest["ITEMS_TIMESTAMP_VALID"] == 1
    assert rows[0]["metadata"]["timestamp_source_field"] == "HTML article:published_time"

    modified = _source(enabled=True)
    modified.update({"mechanism_type": "HTML", "timestamp_field": "dateModified"})
    with pytest.raises(ValueError, match="MODIFICATION_TIMESTAMP_DOES_NOT_QUALIFY"):
        build_live_official_collection_artifact(
            output_root=tmp_path / "modified",
            base_main_sha="5" * 40,
            git_sha="6" * 40,
            input_events_path=_empty_events(tmp_path),
            source_registry_path=_registry(tmp_path, [modified]),
            client=_FakeClient(
                b"""<html><head><meta name="dateModified" content="2026-09-01T12:41:32+03:00">
                </head><body><h1>Modified only</h1></body></html>"""
            ),
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )


def test_bare_clock_date_only_and_analytics_timestamp_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=SourceStatus.INVALID_TIMEZONE.value):
        parse_publication_timestamp_exact("2026-09-01T12:41:32", field_name="HTML datePublished")

    bare = _build(
        tmp_path / "bare",
        _rss(
            """
            <item>
              <title>Bare clock</title>
              <link>https://chtpz.tmk-group.ru/n/bare</link>
              <pubDate>2026-09-01T12:41:32</pubDate>
            </item>
            """
        ),
    )
    date_only = _build(
        tmp_path / "date-only",
        _rss(
            """
            <item>
              <title>Date only</title>
              <link>https://chtpz.tmk-group.ru/n/date</link>
              <pubDate>2026-09-01</pubDate>
            </item>
            """
        ),
    )
    analytics = _build(
        tmp_path / "analytics",
        b"""<html><body><h1>Analytics only</h1>
        <script>window.analyticsAt="2026-09-01T12:41:32+03:00"</script>
        <p>01.09.2026 12:41 issuer text</p></body></html>""",
        source_overrides={
            "mechanism_type": "HTML",
            "timestamp_field": "HTML datePublished",
            "timestamp_policy": (
                "Require publication-specific HTML datePublished with explicit offset."
            ),
        },
    )

    assert bare["ITEMS_NEW"] == 0
    assert date_only["ITEMS_NEW"] == 0
    assert analytics["ITEMS_NEW"] == 0
    assert bare["BLOCKERS_BY_TYPE"] == {SourceStatus.INVALID_TIMEZONE.value: 1}
    assert date_only["BLOCKERS_BY_TYPE"] == {SourceStatus.MISSING_EXACT_TIMESTAMP.value: 1}
    assert analytics["BLOCKERS_BY_TYPE"] == {SourceStatus.INVALID_TIMEZONE.value: 1}


def test_missing_publication_text_is_not_filled_with_unknown(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <link>https://chtpz.tmk-group.ru/n/1</link>
              <guid>textless</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        ),
    )

    assert manifest["ITEMS_FETCHED"] == 1
    assert manifest["ITEMS_WITH_PUBLICATION_MATERIAL"] == 0
    assert manifest["ITEMS_WITHOUT_PUBLICATION_MATERIAL"] == 1
    assert manifest["ITEMS_NEW"] == 0
    assert manifest["SNAPSHOTS_WRITTEN"] == 0
    assert _read_jsonl(tmp_path / ARTIFACT_VERSION / "raw-publication-snapshots.jsonl") == []
    assert _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl") == []
    invalid = _read_jsonl(tmp_path / ARTIFACT_VERSION / "invalid-items.jsonl")
    assert invalid[0]["blocker"] == SourceStatus.PUBLICATION_MATERIAL_MISSING.value


def test_deterministic_event_identity_duplicate_replay_and_existing_state(tmp_path: Path) -> None:
    body = _rss(
        """
        <item>
          <title>Replay stable</title>
          <link>https://chtpz.tmk-group.ru/news/stable</link>
          <guid>stable-guid</guid>
          <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
        </item>
        """
    )
    first = _build(tmp_path / "first", body)
    replay = _build(tmp_path / "replay", body)
    state_replay = _build(
        tmp_path / "state-replay",
        body,
        state_path=tmp_path / "first" / ARTIFACT_VERSION / "dedupe-state.json",
    )

    first_event = _read_jsonl(
        tmp_path / "first" / ARTIFACT_VERSION / "collected-event-metadata.jsonl"
    )[0]["metadata"]
    replay_event = _read_jsonl(
        tmp_path / "replay" / ARTIFACT_VERSION / "collected-event-metadata.jsonl"
    )[0]["metadata"]
    assert first_event["event_id"] == replay_event["event_id"]
    assert first["ITEMS_NEW"] == 1
    assert first["SNAPSHOTS_WRITTEN"] == 1
    assert replay["ITEMS_NEW"] == 1
    assert state_replay["ITEMS_NEW"] == 0
    assert state_replay["ITEMS_DUPLICATE"] == 1
    assert state_replay["SNAPSHOTS_WRITTEN"] == 0
    assert state_replay["DUPLICATE_SNAPSHOTS"] == 0
    assert state_replay["ITEMS_SEMANTIC_READY"] == 0
    assert (
        _read_jsonl(
            tmp_path / "state-replay" / ARTIFACT_VERSION / "raw-publication-snapshots.jsonl"
        )
        == []
    )


def test_multiple_items_sorted_and_source_disabled_behavior(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>Second</title>
              <link>https://chtpz.tmk-group.ru/news/2</link>
              <guid>2</guid>
              <pubDate>Tue, 26 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            <item>
              <title>First</title>
              <link>https://chtpz.tmk-group.ru/news/1</link>
              <guid>1</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        ),
        enabled=False,
    )

    assert manifest["LIVE_EXACT_SOURCES_ENABLED"] == 0
    assert manifest["LIVE_EXACT_SOURCES_ATTEMPTED"] == 0
    assert manifest["ITEMS_FETCHED"] == 0
    assert manifest["BLOCKERS_BY_TYPE"] == {SourceStatus.SOURCE_DISABLED.value: 1}

    enabled = _build(
        tmp_path / "enabled",
        manifest_body=_rss(
            """
        <item>
          <title>Second</title>
          <link>https://chtpz.tmk-group.ru/news/2</link>
          <guid>2</guid>
          <pubDate>Wed, 26 Aug 2026 11:44:16 +0300</pubDate>
        </item>
        <item>
          <title>First</title>
          <link>https://chtpz.tmk-group.ru/news/1</link>
          <guid>1</guid>
          <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
        </item>
        """
        ),
    )
    events = _read_jsonl(tmp_path / "enabled" / ARTIFACT_VERSION / "collected-event-metadata.jsonl")
    assert enabled["ITEMS_NEW"] == 2
    assert [row["metadata"]["source_item_id"] for row in events] == ["1", "2"]


def test_item_match_filter_keeps_shared_rss_feed_ticker_safe(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "sources.json"
    source = _source(enabled=True)
    source.update(
        {
            "official_domain": "www.moex.com",
            "source_url": "https://www.moex.com/export/news.aspx?cat=122",
            "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_id": "AAA_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "ticker": "AAA",
            "issuer": "AAA Issuer",
            "instrument_uid": "uid-aaa",
            "event_origin": "EXCHANGE_ORIGINATED",
            "item_match_any": ["AAA"],
        }
    )
    registry.write_text(
        json.dumps(
            {
                "source_registry_version": "exact-event-live-official-source-registry-v1",
                "sources": [source],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")

    manifest = build_live_official_collection_artifact(
        output_root=tmp_path / "out",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        input_events_path=events,
        source_registry_path=registry,
        client=_FakeClient(
            _rss(
                """
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
                """
            )
        ),
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )

    rows = _read_jsonl(tmp_path / "out" / "collected-event-metadata.jsonl")
    snapshots = _read_jsonl(tmp_path / "out" / "raw-publication-snapshots.jsonl")
    assert manifest["ITEMS_FETCHED"] == 1
    assert manifest["ITEMS_NEW"] == 1
    assert rows[0]["metadata"]["ticker"] == "AAA"
    assert rows[0]["metadata"]["source_item_id"] == "AAA:https://www.moex.com/n1"
    assert rows[0]["metadata"]["source_family"] == "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
    assert rows[0]["metadata"]["event_origin"] == "EXCHANGE_ORIGINATED"
    assert snapshots[0]["link"] == "https://www.moex.com/n1"
    assert snapshots[0]["guid"] == "https://www.moex.com/n1"
    assert snapshots[0]["source_family"] == "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
    assert snapshots[0]["event_origin"] == "EXCHANGE_ORIGINATED"


def test_item_match_filter_does_not_match_official_domain_url(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "sources.json"
    source = _source(enabled=True)
    source.update(
        {
            "official_domain": "www.moex.com",
            "source_url": "https://www.moex.com/export/news.aspx?cat=122",
            "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_id": "MOEX_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "ticker": "MOEX",
            "issuer": "Moscow Exchange",
            "instrument_uid": "uid-moex",
            "item_match_any": ["MOEX"],
        }
    )
    registry.write_text(
        json.dumps(
            {
                "source_registry_version": "exact-event-live-official-source-registry-v1",
                "sources": [source],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")

    manifest = build_live_official_collection_artifact(
        output_root=tmp_path / "out",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        input_events_path=events,
        source_registry_path=registry,
        client=_FakeClient(
            _rss(
                """
                <item>
                  <title>Risk parameters changed for AAA</title>
                  <link>https://www.moex.com/n1</link>
                  <guid>https://www.moex.com/n1</guid>
                  <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
                </item>
                """
            )
        ),
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert manifest["ITEMS_FETCHED"] == 0
    assert manifest["ITEMS_NEW"] == 0


def test_event_origin_filter_excludes_exchange_origin_sources(tmp_path: Path) -> None:
    source = _source(enabled=True)
    source.update(
        {
            "event_origin": "EXCHANGE_ORIGINATED",
            "official_domain": "www.moex.com",
            "source_url": "https://www.moex.com/export/news.aspx?cat=122",
            "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_id": "AAA_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "ticker": "AAA",
            "ticker_attribution_method": "SHARED_FEED_ITEM_TEXT_TOKEN",
            "item_match_any": ["AAA"],
        }
    )
    registry = _registry(tmp_path, [source])
    manifest = build_live_official_collection_artifact(
        output_root=tmp_path / "out",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        input_events_path=_empty_events(tmp_path),
        source_registry_path=registry,
        client=_FakeClient(
            _rss(
                """
                <item>
                  <title>Risk parameters changed for AAA</title>
                  <description>AAA risk parameters changed.</description>
                  <link>https://www.moex.com/n1</link>
                  <guid>https://www.moex.com/n1</guid>
                  <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
                </item>
                """
            )
        ),
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        event_origin_filter=("ISSUER_ORIGINATED",),
    )

    assert manifest["LIVE_EXACT_SOURCES_ENABLED"] == 0
    assert manifest["LIVE_EXACT_SOURCES_ATTEMPTED"] == 0
    assert manifest["NEW_EXACT_EVENTS"] == 0


def test_shared_feed_link_and_guid_do_not_create_ticker_match(tmp_path: Path) -> None:
    source = _source(enabled=True)
    source.update(
        {
            "item_match_any": ["AAA"],
            "ticker": "AAA",
            "ticker_attribution_method": "SHARED_FEED_ITEM_TEXT_TOKEN",
        }
    )
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>No ticker in text</title>
              <description>Issuer update without deterministic token.</description>
              <link>https://chtpz.tmk-group.ru/news/AAA</link>
              <guid>https://chtpz.tmk-group.ru/news/AAA</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        ),
        source_overrides=source,
    )

    assert manifest["ITEMS_FETCHED"] == 0
    assert manifest["ITEMS_NEW"] == 0


def test_shared_html_source_requires_text_token_match(tmp_path: Path) -> None:
    source = _source(enabled=True)
    source.update(
        {
            "item_match_any": ["AAA"],
            "ticker": "AAA",
            "mechanism_type": "HTML",
            "timestamp_field": "HTML article:published_time",
            "timestamp_policy": (
                "Require publication-specific HTML article:published_time with explicit offset."
            ),
            "ticker_attribution_method": "SHARED_HTML_PUBLICATION_TEXT_TOKEN",
        }
    )
    manifest = _build(
        tmp_path,
        b"""<html><head>
        <link rel="canonical" href="https://chtpz.tmk-group.ru/news/AAA">
        <meta property="article:published_time" content="2026-09-01T12:41:32+03:00">
        </head><body>
        <h1>No ticker in material</h1><p>Issuer update without token.</p>
        </body></html>""",
        source_overrides=source,
    )

    assert manifest["ITEMS_FETCHED"] == 0
    assert manifest["ITEMS_NEW"] == 0


def test_html_parser_requires_declared_publication_timestamp_field(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        b"""<html><head>
        <meta property="article:published_time" content="2026-09-01T12:41:32+03:00">
        </head><body><h1>Wrong declared field</h1></body></html>""",
        source_overrides={
            "mechanism_type": "HTML",
            "timestamp_field": "HTML datePublished",
            "timestamp_policy": (
                "Require publication-specific HTML datePublished with explicit offset."
            ),
        },
    )

    assert manifest["ITEMS_NEW"] == 0
    assert manifest["BLOCKERS_BY_TYPE"] == {SourceStatus.MISSING_EXACT_TIMESTAMP.value: 1}


def test_legitimate_unknown_semantic_features_are_preserved(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        _rss(
            """
            <item>
              <title>Neutral corporate update</title>
              <description>Company held a community open day.</description>
              <link>https://chtpz.tmk-group.ru/news/unknown</link>
              <guid>unknown</guid>
              <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
            </item>
            """
        ),
    )
    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "collected-event-metadata.jsonl")
    semantic = _read_jsonl(tmp_path / ARTIFACT_VERSION / "semantic-extraction-results.jsonl")
    assert manifest["ANALYZER_UNKNOWN"] == 1
    assert rows[0]["event_features"] == {
        "event_count": 0,
        "fact_count": 0,
        "primary_event_type": "UNKNOWN",
    }
    assert semantic[0]["event_features"] == rows[0]["event_features"]


def test_cli_has_no_scheduler_or_tinvest_paths() -> None:
    help_text = build_parser().format_help()

    assert "schedule" not in help_text.lower()
    assert "tinvest" not in help_text.lower()


def _build(
    tmp_path: Path,
    manifest_body: bytes,
    *,
    enabled: bool = True,
    state_path: Path | None = None,
    source_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = _source(enabled=enabled)
    if source_overrides is not None:
        if source_overrides.get("source_registry_version"):
            source = source_overrides
        else:
            source.update(source_overrides)
    registry = _registry(tmp_path, [source])
    events = _empty_events(tmp_path)
    return build_live_official_collection_artifact(
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        input_events_path=events,
        source_registry_path=registry,
        state_path=state_path,
        client=_FakeClient(manifest_body),
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )


def _source(*, enabled: bool) -> dict[str, Any]:
    return {
        "archive_capability": False,
        "enabled": enabled,
        "event_origin": "ISSUER_ORIGINATED",
        "instrument_uid": "uid-chep",
        "issuer": "ЧТПЗ",
        "live_capability": True,
        "mechanism_type": "RSS",
        "official_domain": "chtpz.tmk-group.ru",
        "provenance_evidence_sha": "audit-sha",
        "provenance_evidence_url": "artifact#CHEP",
        "source_family": "CHEP_OFFICIAL_RSS_EXACT_LIVE_V1",
        "source_id": "CHEP_CHTPZ_TMK_RSS_EXACT_LIVE_V1",
        "source_registry_version": "exact-event-live-official-source-registry-v1",
        "source_url": "https://chtpz.tmk-group.ru/rss",
        "ticker": "CHEP",
        "ticker_attribution_method": "ISSUER_OWNED_SINGLE_ISSUER_SOURCE",
        "timestamp_field": "RSS item pubDate",
        "timestamp_policy": "Require item-level RSS pubDate with explicit +0300 timezone.",
    }


def _rss(items: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns="http://backend.userland.com/rss2" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel>
        <title>CHEP RSS</title>
        {items}
      </channel>
    </rss>
    """.encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _registry(tmp_path: Path, sources: list[dict[str, Any]]) -> Path:
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps(
            {
                "source_registry_version": "exact-event-live-official-source-registry-v1",
                "sources": sources,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return registry


def _empty_events(tmp_path: Path) -> Path:
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    return events


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
