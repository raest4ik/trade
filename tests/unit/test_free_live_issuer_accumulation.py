from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_accumulation.application import (
    audit_live_issuer_sources,
    collect_live_issuer_news,
)
from src.free_live_issuer_accumulation.domain import (
    EXPECTED_RULES_V3_FINGERPRINT,
    LiveModelPredictionError,
    PointInTimeFeatureBoundError,
    SealedLiveEpochOutcomeReadError,
    assert_pre_event_feature_upper_bound,
    guard_sealed_live_epoch_model_prediction,
    guard_sealed_live_epoch_outcome_read,
    guard_sealed_live_epoch_post_event_price_read,
    parse_publication_timestamp,
)
from src.instruments.application.ports import InstrumentRepository
from src.market_data.application.ports import MarketDataRepository
from src.news.application.ports import NewsRepository
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.reactions.application.ports import ReactionRepository
from src.reactions.application.use_cases import CalculateNewsMarketReactions


def test_explicit_offset_timestamp_accepted_and_naive_rejected() -> None:
    assert parse_publication_timestamp("Tue, 25 Aug 2026 11:44:16 +0300") == datetime(
        2026, 8, 25, 8, 44, 16, tzinfo=UTC
    )
    assert parse_publication_timestamp("2026-09-02T14:31:00+03:00") == datetime(
        2026, 9, 2, 11, 31, tzinfo=UTC
    )
    assert parse_publication_timestamp("2026-09-02T11:31:00Z") == datetime(
        2026, 9, 2, 11, 31, tzinfo=UTC
    )
    assert parse_publication_timestamp(
        "2026-09-02T11:31:00",
        {
            "evidence_type": "TIMESTAMP_EVIDENCE_TYPE=DOCUMENTED_TIMEZONE_UTC",
            "evidence_value": "First-party source contract says publication timestamps are UTC",
        },
    ) == datetime(2026, 9, 2, 11, 31, tzinfo=UTC)
    with pytest.raises(ValueError, match="INVALID_TIMEZONE"):
        parse_publication_timestamp("Tue, 25 Aug 2026 11:44:16")
    with pytest.raises(ValueError, match="MISSING_EXACT_TIMESTAMP"):
        parse_publication_timestamp("2026-08-25")


def test_crawl_time_cannot_substitute_publication_time(tmp_path: Path) -> None:
    manifest = _collect(
        tmp_path,
        {
            "https://issuer.test/rss": _rss_items(
                """
                <item>
                  <title>No date</title>
                  <link>https://issuer.test/news/no-date</link>
                  <guid>no-date</guid>
                </item>
                """
            )
        },
    )

    assert manifest["EVENTS_COLLECTED"] == 0
    assert manifest["RAW_SNAPSHOTS_FROZEN"] == 0
    assert manifest["LIVE_POST_EVENT_PRICE_READS"] == 0


def test_date_modified_cannot_substitute_publication_time(tmp_path: Path) -> None:
    source = _source("https://issuer.test/rss", "issuer.test", "AAA", "AAA_SOURCE_V1")
    source["timestamp_path"] = "dateModified"
    registry = _registry(tmp_path / "registry.json", sources=[source])

    with pytest.raises(ValueError, match="DATE_MODIFIED_CANNOT_BE_PUBLICATION_TIMESTAMP"):
        collect_live_issuer_news(
            output_root=tmp_path / "out",
            base_main_sha="5" * 40,
            git_sha="6" * 40,
            registry_path=registry,
            client=_FakeClient({"https://issuer.test/rss": _rss("<title>Result</title>")}),
            created_at=_NOW,
        )


def test_duplicate_poll_idempotent_and_updated_publication_creates_revision(tmp_path: Path) -> None:
    body = _rss(
        """
        <title>Stable headline</title>
        <description>Stable description</description>
        <link>https://issuer.test/news/stable</link>
        <guid>stable-guid</guid>
        <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
        """
    )
    first = _collect(tmp_path / "first", {"https://issuer.test/rss": body})
    replay = _collect(
        tmp_path / "replay",
        {"https://issuer.test/rss": body},
        state_path=tmp_path / "first" / "dedupe-state.json",
    )
    updated = _collect(
        tmp_path / "updated",
        {
            "https://issuer.test/rss": _rss(
                """
                <title>Updated headline</title>
                <description>Updated description</description>
                <link>https://issuer.test/news/stable</link>
                <guid>stable-guid</guid>
                <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
                """
            )
        },
        state_path=tmp_path / "first" / "dedupe-state.json",
    )

    assert first["EVENTS_COLLECTED"] == 1
    assert replay["EVENTS_COLLECTED"] == 0
    assert replay["DUPLICATES_ENCOUNTERED"] == 1
    assert updated["REVISIONS_CREATED"] == 1
    first_snapshots = sorted((tmp_path / "first" / "raw-snapshots").glob("*.json"))
    assert len(first_snapshots) == 1
    assert json.loads(first_snapshots[0].read_text(encoding="utf-8"))["title"] == "Stable headline"


def test_canonical_url_dedup_when_guid_changes(tmp_path: Path) -> None:
    manifest = _collect(
        tmp_path,
        {
            "https://issuer.test/rss": _rss_items(
                """
                <item>
                  <title>First</title>
                  <link>https://issuer.test/news/same</link>
                  <guid>guid-1</guid>
                  <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
                </item>
                <item>
                  <title>Second</title>
                  <link>https://issuer.test/news/same</link>
                  <guid>guid-2</guid>
                  <pubDate>Tue, 25 Aug 2026 12:44:16 +0300</pubDate>
                </item>
                """
            )
        },
    )

    assert manifest["EVENTS_COLLECTED"] == 1
    assert manifest["DUPLICATES_ENCOUNTERED"] == 1


def test_bounded_poll_uses_latest_publications_after_cutoff(tmp_path: Path) -> None:
    old_items = "\n".join(
        f"""
        <item>
          <title>Old {index}</title>
          <link>https://issuer.test/news/old-{index}</link>
          <guid>old-{index}</guid>
          <pubDate>Tue, 01 Jul 2025 10:0{index}:00 +0000</pubDate>
        </item>
        """
        for index in range(5)
    )
    manifest = _collect(
        tmp_path,
        {
            "https://issuer.test/rss": _rss_items(
                old_items
                + """
                <item>
                  <title>New after cutoff</title>
                  <link>https://issuer.test/news/new</link>
                  <guid>new</guid>
                  <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
                </item>
                """
            )
        },
    )
    rows = _read_jsonl(tmp_path / "live-shadow-corpus.jsonl")

    assert manifest["EVENTS_COLLECTED"] == 1
    assert rows[0]["source_item_id"] == "new"


def test_ticker_binding_and_ambiguous_ticker_rejected(tmp_path: Path) -> None:
    _collect(tmp_path, {"https://issuer.test/rss": _rss("<title>Issuer update</title>")})
    row = _read_jsonl(tmp_path / "live-shadow-corpus.jsonl")[0]

    assert row["ticker"] == "AAA"
    with pytest.raises(ValueError, match="ENABLED_SOURCE_REQUIRES_DETERMINISTIC_TICKER"):
        collect_live_issuer_news(
            output_root=tmp_path / "bad",
            base_main_sha="5" * 40,
            git_sha="6" * 40,
            registry_path=_registry(tmp_path / "bad-registry.json", ticker="MULTI"),
            client=_FakeClient({"https://issuer.test/rss": _rss("<title>Bad</title>")}),
            created_at=_NOW,
        )


def test_sealed_epoch_target_and_post_event_price_reads_rejected() -> None:
    published = datetime(2026, 8, 25, 8, 44, tzinfo=UTC)

    with pytest.raises(
        SealedLiveEpochOutcomeReadError, match="SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT"
    ):
        guard_sealed_live_epoch_outcome_read(
            epoch="LIVE_SHADOW_CORPUS",
            target_status="SEALED",
            context="target",
        )
    with pytest.raises(
        SealedLiveEpochOutcomeReadError, match="SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT"
    ):
        guard_sealed_live_epoch_post_event_price_read(
            epoch="LIVE_SHADOW_CORPUS",
            published_at=published,
            query_end_at=published + timedelta(minutes=15),
            context="price",
        )
    with pytest.raises(LiveModelPredictionError, match="SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT"):
        guard_sealed_live_epoch_model_prediction(
            epoch="LIVE_SHADOW_CORPUS",
            target_status="SEALED",
            context="model",
        )


def test_reaction_pipeline_rejects_sealed_live_epoch_before_market_reads() -> None:
    news_id = UUID(int=1)
    news = NewsItem.create(
        source_id="live-issuer-shadow-corpus-v1",
        source_name="live shadow",
        source_url="https://issuer.test/news/1",
        title="sealed",
        raw_content="sealed",
        language="en",
        published_at=datetime(2026, 8, 25, 8, 44, tzinfo=UTC),
        publication_timestamp_quality=PublicationTimestampQuality.EXACT,
    )
    use_case = CalculateNewsMarketReactions(
        news_repository=cast(NewsRepository, _NewsRepository(news_id, news)),
        instrument_repository=cast(InstrumentRepository, _InstrumentRepository()),
        market_data_repository=cast(MarketDataRepository, _MarketDataRepository()),
        reaction_repository=cast(ReactionRepository, _ReactionRepository()),
    )

    with pytest.raises(
        SealedLiveEpochOutcomeReadError, match="SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT"
    ):
        import asyncio

        asyncio.run(use_case.execute(news_id))


def test_pre_event_feature_upper_bound() -> None:
    published = datetime(2026, 8, 25, 8, 44, tzinfo=UTC)
    assert_pre_event_feature_upper_bound(feature_timestamp=published, published_at=published)
    with pytest.raises(PointInTimeFeatureBoundError):
        assert_pre_event_feature_upper_bound(
            feature_timestamp=published + timedelta(microseconds=1),
            published_at=published,
        )


def test_frozen_rules_fingerprint_paid_source_audit_and_zero_future_counters(
    tmp_path: Path,
) -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_V3_FINGERPRINT

    audit = audit_live_issuer_sources(
        output_root=tmp_path / "audit",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=_registry(tmp_path / "registry.json"),
        created_at=_NOW,
    )
    manifest = _collect(
        tmp_path / "collect", {"https://issuer.test/rss": _rss("<title>Result</title>")}
    )

    assert audit["PAID_OUT_OF_SCOPE_SOURCES"] == 1
    assert audit["PAID_SOURCES_USED"] is False
    assert audit["STRICT_ANSWER"] == "NO"
    for artifact_name in (
        "manifest.json",
        "source-registry.json",
        "source-audit.jsonl",
        "ticker-coverage.json",
        "shadow-corpus-stats.json",
        "timestamp-contracts.json",
        "rejections.jsonl",
        "safety.json",
        "report.md",
    ):
        assert (tmp_path / "audit" / artifact_name).exists()
    assert manifest["LIVE_OUTCOMES_READ"] == 0
    assert manifest["LIVE_TARGETS_COMPUTED"] == 0
    assert manifest["LIVE_POST_EVENT_PRICE_READS"] == 0
    assert manifest["LIVE_MODEL_PREDICTIONS"] == 0
    assert manifest["OLD_FUTURE_HOLDOUT_OPENED"] is False


def test_one_source_failure_does_not_stop_other_sources(tmp_path: Path) -> None:
    manifest = collect_live_issuer_news(
        output_root=tmp_path / "out",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=_registry(tmp_path / "registry.json", include_second=True),
        client=_FakeClient(
            {
                "https://issuer.test/rss": FetchResult(
                    request_url="https://issuer.test/rss",
                    final_url="https://issuer.test/rss",
                    status=500,
                    content_type=None,
                    body=b"",
                    redirects=0,
                    redirect_chain=(),
                    blocker="HTTP_FAILURE",
                ),
                "https://issuer2.test/rss": _rss("<title>Second source</title>"),
            }
        ),
        created_at=_NOW,
    )

    assert manifest["EVENTS_COLLECTED"] == 1
    assert manifest["metrics"]["source_failures"] == 1


def test_poll_one_source_only(tmp_path: Path) -> None:
    manifest = collect_live_issuer_news(
        output_root=tmp_path / "out",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=_registry(tmp_path / "registry.json", include_second=True),
        client=_FakeClient(
            {
                "https://issuer.test/rss": _rss("<title>First source</title>"),
                "https://issuer2.test/rss": _rss("<title>Second source</title>"),
            }
        ),
        created_at=_NOW,
        source_id="BBB_SOURCE_V1",
    )
    rows = _read_jsonl(tmp_path / "out" / "live-shadow-corpus.jsonl")

    assert manifest["EVENTS_COLLECTED"] == 1
    assert rows[0]["ticker"] == "BBB"


def test_registry_and_raw_snapshot_sha_are_deterministic(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    first = collect_live_issuer_news(
        output_root=tmp_path / "first",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=registry,
        client=_FakeClient({"https://issuer.test/rss": _rss("<title>Stable</title>")}),
        created_at=_NOW,
    )
    second = collect_live_issuer_news(
        output_root=tmp_path / "second",
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=registry,
        client=_FakeClient({"https://issuer.test/rss": _rss("<title>Stable</title>")}),
        created_at=_NOW,
    )
    first_snapshot = _read_jsonl(tmp_path / "first" / "raw-publication-snapshots.jsonl")[0]
    second_snapshot = _read_jsonl(tmp_path / "second" / "raw-publication-snapshots.jsonl")[0]

    assert first["SOURCE_REGISTRY_SHA"] == second["SOURCE_REGISTRY_SHA"]
    assert first_snapshot["raw_snapshot_sha"] == second_snapshot["raw_snapshot_sha"]


_NOW = datetime(2026, 9, 2, 9, tzinfo=UTC)


def _collect(
    tmp_path: Path,
    bodies: dict[str, bytes | FetchResult],
    *,
    state_path: Path | None = None,
) -> dict[str, Any]:
    return collect_live_issuer_news(
        output_root=tmp_path,
        base_main_sha="5" * 40,
        git_sha="6" * 40,
        registry_path=_registry(tmp_path.parent / f"{tmp_path.name}-registry.json"),
        state_path=state_path,
        client=_FakeClient(bodies),
        created_at=_NOW,
    )


def _registry(
    path: Path,
    *,
    ticker: str = "AAA",
    include_second: bool = False,
    sources: list[dict[str, Any]] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry_sources = (
        sources
        if sources is not None
        else [_source("https://issuer.test/rss", "issuer.test", ticker, "AAA_SOURCE_V1")]
    )
    if include_second:
        registry_sources.append(
            _source("https://issuer2.test/rss", "issuer2.test", "BBB", "BBB_SOURCE_V1")
        )
    registry_sources.append(
        {
            **_source("https://paid.test/api", "paid.test", "MULTI", "PAID_SOURCE_V1"),
            "enabled": False,
            "source_status": "OUT_OF_SCOPE_PAID_SOURCE",
            "source_origin": "COMMERCIAL_PROVIDER",
            "parser": "out_of_scope_paid_source",
            "timestamp_path": "out_of_scope",
        }
    )
    path.write_text(
        json.dumps(
            {
                "historical_frozen_issuer_tickers": [
                    "GMKN",
                    "MGNT",
                    "ROSN",
                    "T",
                    "VKCO",
                    "X5",
                    "YDEX",
                ],
                "milestone": {
                    "minimum_new_issuer_tickers": 3,
                    "minimum_total_issuer_tickers": 10,
                    "name": "LIVE_DIVERSITY_MILESTONE_V1",
                },
                "source_registry_version": "live-issuer-sources-v1",
                "sources": registry_sources,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _source(url: str, domain: str, ticker: str, source_id: str) -> dict[str, Any]:
    return {
        "canonical_domain": domain,
        "content_path": ["rss.channel.item.title", "rss.channel.item.description"],
        "discovery_type": "official_issuer_rss",
        "discovery_url": url,
        "enabled": True,
        "expected_publication_frequency": "test",
        "identity_path": "rss.channel.item.guid || rss.channel.item.link",
        "issuer": f"{ticker} Issuer",
        "parser": "rss-item-pubdate-explicit-offset-v1",
        "polling_policy": {"interval_minutes": 60, "max_items_per_poll": 5},
        "source_id": source_id,
        "source_origin": "ISSUER_ORIGINATED",
        "source_status": "LIVE_STRICT_EXACT_READY",
        "source_version": 1,
        "stable_identity": "rss_guid_or_link",
        "ticker": ticker,
        "ticker_binding": {
            "binding": "single_issuer_source",
            "publication_date_validity": "test",
        },
        "timestamp_contract": {
            "evidence_type": "TIMESTAMP_EVIDENCE_TYPE=RFC822_EXPLICIT_OFFSET",
            "evidence_value": "RSS pubDate includes +0300",
            "policy": "accept explicit offset only",
        },
        "timestamp_path": "rss.channel.item.pubDate",
    }


def _rss(item_body: str) -> bytes:
    if "<pubDate>" not in item_body:
        item_body = (
            item_body
            + "<link>https://issuer.test/news/1</link>"
            + "<guid>issuer-1</guid>"
            + "<pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>"
        )
    return _rss_items(f"<item>{item_body}</item>")


def _rss_items(items: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Issuer RSS</title>
    {items}
  </channel>
</rss>
""".encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _FakeClient:
    def __init__(self, bodies: dict[str, bytes | FetchResult]) -> None:
        self._bodies = bodies

    def get(self, url: str) -> FetchResult:
        body = self._bodies[url]
        if isinstance(body, FetchResult):
            return body
        return FetchResult(
            request_url=url,
            final_url=url,
            status=200,
            content_type="application/rss+xml",
            body=body,
            redirects=0,
            redirect_chain=(),
            blocker=None,
        )


@dataclass(frozen=True, slots=True)
class _NewsRepository:
    news_id: UUID
    news: NewsItem

    async def get_by_id(self, news_id: UUID) -> NewsItem | None:
        return self.news if news_id == self.news_id else None


class _InstrumentRepository:
    async def get_news_matches(self, news_id: UUID) -> list[Any]:
        raise AssertionError("instrument repository must not be reached")


class _MarketDataRepository:
    async def get_benchmark_by_code(self, code: str) -> None:
        raise AssertionError("market data repository must not be reached")


class _ReactionRepository:
    async def replace_reactions(self, **kwargs: Any) -> list[Any]:
        raise AssertionError("reaction repository must not be reached")
