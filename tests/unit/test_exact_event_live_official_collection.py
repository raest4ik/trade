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
    parse_rss_pubdate_exact,
    publication_material,
    publication_material_sha,
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


def _build(
    tmp_path: Path,
    manifest_body: bytes,
    *,
    enabled: bool = True,
    state_path: Path | None = None,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps(
            {
                "source_registry_version": "exact-event-live-official-source-registry-v1",
                "sources": [_source(enabled=enabled)],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
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
