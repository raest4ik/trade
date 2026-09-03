from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_accumulation.domain import LiveIssuerSource
from src.free_live_issuer_accumulation.operation import (
    FreeLiveResearchOperation,
    OperationConfig,
    SourcePollStatus,
    build_operation_status,
    verify_operation_seal,
)

BASE_MAIN_SHA = "5" * 40
GIT_SHA = "6" * 40
NOW = datetime(2026, 9, 2, 9, tzinfo=UTC)


def test_poll_once_persists_ready_operation_and_blocked_ml_status(tmp_path: Path) -> None:
    client = _CountingClient(
        {
            "https://issuer.test/rss": _rss_item("AAA", "one"),
            "https://issuer2.test/rss": _rss_item("BBB", "two", domain="issuer2.test"),
        }
    )
    operation = _operation(tmp_path, client, now=NOW)

    report = operation.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)
    status = build_operation_status(
        tmp_path / "operation",
        registry_path=tmp_path / "registry.json",
        historical_ticker_summary_path=tmp_path / "missing-tickers.jsonl",
    )

    assert report["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
    assert report["ML_V2_DATASET_STATUS"] == "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY"
    assert report["newly_discovered_publications"] == 2
    assert report["LIVE_OUTCOMES_READ"] == 0
    assert report["LIVE_TARGETS_COMPUTED"] == 0
    assert report["LIVE_POST_EVENT_PRICE_READS"] == 0
    assert status["collector_process_alive"] is True
    assert status["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
    assert (tmp_path / "operation" / "manifest.json").exists()
    assert (tmp_path / "operation" / "source-health.json").exists()
    assert (tmp_path / "operation" / "live-shadow-corpus.jsonl").exists()
    assert len((tmp_path / "operation" / "live-shadow-corpus.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "operation" / "candidate-source-backlog.json").exists()


def test_poll_interval_skip_keeps_previous_success_healthy(tmp_path: Path) -> None:
    client = _CountingClient(
        {
            "https://issuer.test/rss": _rss_item("AAA", "one"),
            "https://issuer2.test/rss": _rss_item("BBB", "two", domain="issuer2.test"),
        }
    )
    first = _operation(tmp_path, client, now=NOW)
    first.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)
    calls_after_first = len(client.calls)

    second = _operation(tmp_path, client, now=NOW + timedelta(minutes=1))
    report = second.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)

    assert len(client.calls) == calls_after_first
    assert report["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
    assert {row["blocker"] for row in report["source_results"]} == {"POLL_INTERVAL_NOT_ELAPSED"}


def test_restart_dedupe_state_makes_replay_no_new_items(tmp_path: Path) -> None:
    client = _CountingClient(
        {
            "https://issuer.test/rss": _rss_item("AAA", "one"),
            "https://issuer2.test/rss": _rss_item("BBB", "two", domain="issuer2.test"),
        }
    )
    _operation(tmp_path, client, now=NOW).poll_once(
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
    )

    replay = _operation(tmp_path, client, now=NOW + timedelta(hours=2)).poll_once(
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
    )

    assert replay["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
    assert replay["newly_discovered_publications"] == 2
    assert sum(int(row["accepted"]) for row in replay["source_results"]) == 0
    assert replay["duplicates"] == 2
    assert {row["status"] for row in replay["source_results"]} == {
        SourcePollStatus.NO_NEW_ITEMS.value
    }


def test_source_failure_degrades_operation_without_blocking_other_source(tmp_path: Path) -> None:
    client = _CountingClient(
        {
            "https://issuer.test/rss": _failure("https://issuer.test/rss"),
            "https://issuer2.test/rss": _rss_item("BBB", "two", domain="issuer2.test"),
        }
    )
    operation = _operation(tmp_path, client, now=NOW)

    report = operation.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)

    assert report["LIVE_RESEARCH_OPERATION_STATUS"] == "DEGRADED"
    assert report["source_failures"] == 1
    assert report["newly_discovered_publications"] == 1
    by_source = {row["source_id"]: row for row in report["source_results"]}
    assert by_source["AAA_SOURCE_V1"]["status"] == SourcePollStatus.SOURCE_FAILURE.value
    assert by_source["BBB_SOURCE_V1"]["status"] == SourcePollStatus.SUCCESS.value


def test_circuit_breaker_skips_degraded_source_until_cooldown(tmp_path: Path) -> None:
    client = _CountingClient({"https://issuer.test/rss": _failure("https://issuer.test/rss")})
    operation = _operation(
        tmp_path,
        client,
        now=NOW,
        include_second=False,
        failure_threshold=2,
    )
    operation.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)
    _operation(
        tmp_path,
        client,
        now=NOW + timedelta(hours=2),
        include_second=False,
        failure_threshold=2,
    ).poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)
    calls_after_failures = len(client.calls)

    degraded = _operation(
        tmp_path,
        client,
        now=NOW + timedelta(hours=2, minutes=1),
        include_second=False,
        failure_threshold=2,
    ).poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)

    assert len(client.calls) == calls_after_failures
    assert degraded["source_results"][0]["status"] == SourcePollStatus.SOURCE_DEGRADED.value
    assert degraded["source_results"][0]["blocker"] == "SOURCE_DEGRADED_COOLDOWN"


def test_retry_features_and_seal_never_open_targets(tmp_path: Path) -> None:
    client = _CountingClient({"https://issuer.test/rss": _rss_item("AAA", "one")})
    operation = _operation(tmp_path, client, now=NOW, include_second=False)
    operation.poll_once(base_main_sha=BASE_MAIN_SHA, git_sha=GIT_SHA)

    retry = operation.retry_features(published_at=NOW)
    seal = verify_operation_seal(tmp_path / "operation")

    assert retry["LIVE_POST_EVENT_PRICE_READS"] == 0
    assert retry["LIVE_TARGETS_COMPUTED"] == 0
    assert retry["LIVE_OUTCOMES_READ"] == 0
    assert seal["sealed_epoch_verified"] is True
    assert seal["child_artifacts_checked"] == 1


def _operation(
    tmp_path: Path,
    client: _CountingClient,
    *,
    now: datetime,
    include_second: bool = True,
    failure_threshold: int = 3,
) -> FreeLiveResearchOperation:
    registry_path = _registry(
        tmp_path / "registry.json",
        include_second=include_second,
    )
    config = OperationConfig(
        artifact_root=tmp_path / "operation",
        registry_path=registry_path,
        historical_ticker_summary_path=tmp_path / "missing-tickers.jsonl",
        default_interval_minutes=10,
        max_retries=1,
        failure_threshold=failure_threshold,
        cooldown_minutes=30,
        max_items_per_poll=5,
    )

    def client_factory(_source: LiveIssuerSource) -> _CountingClient:
        return client

    return FreeLiveResearchOperation(config, client_factory=client_factory, now=lambda: now)


def _rss_item(ticker: str, suffix: str, *, domain: str = "issuer.test") -> bytes:
    return _rss_items(
        f"""
        <item>
          <title>{ticker} headline {suffix}</title>
          <description>{ticker} description {suffix}</description>
          <link>https://{domain}/news/{suffix}</link>
          <guid>{ticker}-{suffix}</guid>
          <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
        </item>
        """
    )


def _registry(
    path: Path,
    *,
    include_second: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = [_source("https://issuer.test/rss", "issuer.test", "AAA", "AAA_SOURCE_V1")]
    if include_second:
        sources.append(_source("https://issuer2.test/rss", "issuer2.test", "BBB", "BBB_SOURCE_V1"))
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
                "sources": [
                    *sources,
                    {
                        **_source(
                            "https://paid.test/api",
                            "paid.test",
                            "MULTI",
                            "PAID_SOURCE_V1",
                        ),
                        "enabled": False,
                        "source_origin": "COMMERCIAL_PROVIDER",
                        "source_status": "OUT_OF_SCOPE_PAID_SOURCE",
                    },
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _source(url: str, domain: str, ticker: str, source_id: str) -> dict[str, object]:
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


def _rss_items(items: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Issuer RSS</title>
    {items}
  </channel>
</rss>
""".encode()


def _failure(url: str) -> FetchResult:
    return FetchResult(
        request_url=url,
        final_url=url,
        status=500,
        content_type=None,
        body=b"",
        redirects=0,
        redirect_chain=(),
        blocker="HTTP_FAILURE",
    )


class _CountingClient:
    def __init__(self, bodies: dict[str, bytes | FetchResult]) -> None:
        self._bodies = bodies
        self.calls: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
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
