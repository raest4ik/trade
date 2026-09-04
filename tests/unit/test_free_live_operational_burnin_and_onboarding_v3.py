from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_expansion_v2.application import SourceProbeConfig
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    CandidateBacklogEntry,
    assert_recheck_has_new_hypothesis,
    build_burnin_summary,
    distinct_new_legal_issuers,
    diversity_status,
    ready_sources_visible_to_worker,
    registry_delta_from_accepted_sources,
    run_free_live_operational_burnin_and_onboarding_v3,
)

BASE_MAIN_SHA = "1" * 40
GIT_SHA = "2" * 40
NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def test_burnin_summary_requires_multiple_cycles_persisted_state_and_seal() -> None:
    summary = build_burnin_summary(
        [
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "source_results": [{"source_id": "A", "status": "SUCCESS", "accepted": 1}],
            },
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "source_results": [{"source_id": "A", "status": "NO_NEW_ITEMS", "accepted": 0}],
            },
        ],
        status=_status(),
        collector_state={"sources": {"A": {"last_seen_source_identity": "item-1"}}},
        seal={"sealed_epoch_verified": True},
    )

    assert summary["OPERATION"] == "YES"
    assert summary["state_persistence_proven"] is True
    assert summary["repeat_poll_no_duplicate"] is True
    assert summary["pre_event_features_bounded"] is True


def test_isolated_source_failure_degrades_without_hiding_healthy_sources() -> None:
    summary = build_burnin_summary(
        [
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "DEGRADED",
                "source_results": [
                    {"source_id": "A", "status": "SOURCE_FAILURE", "accepted": 0},
                    {"source_id": "B", "status": "SUCCESS", "accepted": 1},
                ],
            },
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "source_results": [{"source_id": "B", "status": "NO_NEW_ITEMS", "accepted": 0}],
            },
        ],
        status=_status(),
        collector_state={"sources": {"B": {"last_seen_source_identity": "item-1"}}},
        seal={"sealed_epoch_verified": True},
    )

    assert summary["one_source_failure_isolated"] is True
    assert summary["OPERATION"] == "YES"


def test_legal_issuer_dedup_collapses_share_classes_and_frozen_cohort() -> None:
    issuers = distinct_new_legal_issuers(
        [
            {"ticker": "SBER", "legal_issuer": "ПАО Сбербанк"},
            {"ticker": "SBERP", "legal_issuer": "ПАО Сбербанк"},
            {"ticker": "ROSN", "legal_issuer": "Rosneft Oil Company"},
            {"ticker": "GAZP", "legal_issuer": "ПАО Газпром"},
        ]
    )

    assert {row["ticker"] for row in issuers} == {"GAZP", "SBER"}
    assert diversity_status(len(issuers)) == "TWO_NEW_FREE_ISSUERS"
    assert diversity_status(3) == "THREE_PLUS_NEW_FREE_ISSUERS"


def test_candidate_backlog_enforces_slow_cadence_and_new_hypothesis() -> None:
    entry = CandidateBacklogEntry(
        issuer="Issuer",
        tickers=("AAA",),
        official_url="https://issuer.test/news",
        last_checked=date(2026, 9, 3),
        previous_blocker="LIVE_CLOCK_WITHOUT_TIMEZONE",
        new_hypothesis="Inspect JSON-LD datePublished instead of visible page clock.",
        current_status="PENDING_RECHECK",
        timestamp_evidence=None,
        identity_evidence="canonical URL",
        next_possible_free_mechanism="official_html_jsonld",
        slow_recheck_after_days=7,
    )

    assert entry.payload()["slow_recheck_after_days"] == 7
    assert_recheck_has_new_hypothesis(
        previous_url="https://issuer.test/news",
        previous_mechanism="html_clock",
        candidate_url="https://issuer.test/news",
        candidate_mechanism="html_clock",
        new_hypothesis="Check embedded JSON LD datePublished value.",
    )
    with pytest.raises(ValueError, match="RECHECK_REQUIRES_NEW_HYPOTHESIS"):
        assert_recheck_has_new_hypothesis(
            previous_url="https://issuer.test/news",
            previous_mechanism="html_clock",
            candidate_url="https://issuer.test/news",
            candidate_mechanism="html_clock",
            new_hypothesis="",
        )


def test_registry_delta_is_worker_visible_but_not_enabled_by_default() -> None:
    accepted = [_accepted("GAZP", "ПАО Газпром"), _accepted("SBER", "ПАО Сбербанк")]

    rows = registry_delta_from_accepted_sources(accepted)

    assert ready_sources_visible_to_worker(rows) is True
    assert {row["enabled"] for row in rows} == {False}


def test_run_v3_writes_required_artifacts_and_keeps_ml_closed(tmp_path: Path) -> None:
    operation_root = _operation_root(tmp_path / "operation")
    registry = _registry(tmp_path / "registry.json")
    output_root = tmp_path / "v3"
    manifest = run_free_live_operational_burnin_and_onboarding_v3(
        output_root=output_root,
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=operation_root,
        source_registry_path=registry,
        historical_ticker_summary_path=tmp_path / "missing.jsonl",
        candidate_configs=(
            _candidate("GAZP", "ПАО Газпром"),
            _candidate("SBER", "ПАО Сбербанк"),
            _candidate("SBERP", "ПАО Сбербанк"),
            _candidate("NVTK", "ПАО НОВАТЭК"),
        ),
        client=_Http(),
        created_at=NOW,
    )

    expected = {
        "manifest.json",
        "burnin-summary.json",
        "poll-cycles.jsonl",
        "source-health.json",
        "candidate-backlog.json",
        "source-probes.jsonl",
        "accepted-sources.json",
        "rejected-sources.jsonl",
        "shadow-stats.json",
        "feature-status.json",
        "safety.json",
        "report.md",
    }
    assert expected.issubset({path.name for path in output_root.iterdir()})
    assert manifest["OPERATION"] == "YES"
    assert manifest["DIVERSITY"] == "YES"
    assert manifest["ML_V2_DATASET_STATUS"] == "NOT_OPENED_BY_V3_ONBOARDING"
    assert manifest["LIVE_OUTCOMES_READ"] == 0
    assert manifest["LIVE_TARGETS_COMPUTED"] == 0
    assert manifest["BROKER_MUTATIONS"] == 0


def _status() -> dict[str, object]:
    return {
        "enabled_sources": ["A"],
        "source_health": [{"source_id": "A", "healthy": True}],
        "duplicates": 1,
        "sealed_violations": 0,
        "timestamp_rejections": 0,
        "feature_status": {"upper_bound_policy": "end_at <= published_at"},
        "outcome_counters": {
            "LIVE_OUTCOMES_READ": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "BROKER_MUTATIONS": 0,
        },
    }


def _accepted(ticker: str, issuer: str) -> dict[str, object]:
    return {
        "source_id": f"{ticker}_SOURCE",
        "ticker": ticker,
        "legal_issuer": issuer,
        "domain": "issuer.test",
        "discovery_url": "https://issuer.test/rss",
        "mechanism": "official_issuer_rss",
        "parser_version": "rss-item-pubdate-explicit-offset-v1",
        "timestamp_field": "rss.channel.item.pubDate",
        "timezone_evidence": "LEVEL_A_EXPLICIT_OFFSET_OR_UTC",
        "identity_mechanism": "rss.guid",
        "real_item_observed": True,
    }


def _candidate(ticker: str, issuer: str) -> SourceProbeConfig:
    return SourceProbeConfig(
        source_id=f"{ticker}_SOURCE",
        ticker=ticker,
        legal_issuer=issuer,
        official_domain="issuer.test",
        url=f"https://issuer.test/{ticker}.rss",
        mechanism="official_issuer_rss",
        parser="rss-item-pubdate-explicit-offset-v1",
        timestamp_field="rss.channel.item.pubDate",
        identity_field="rss.channel.item.guid",
        content_fields=("rss.channel.item.title", "rss.channel.item.description"),
        new_hypothesis="Probe first-party RSS item pubDate explicit offset.",
    )


class _Http:
    def get(self, url: str) -> FetchResult:
        ticker = url.rsplit("/", 1)[-1].split(".", 1)[0]
        body = f"""<rss version="2.0"><channel><item>
<title>{ticker} headline</title>
<link>{url}/1</link>
<guid>{ticker}-1</guid>
<pubDate>Thu, 03 Sep 2026 12:00:00 +0300</pubDate>
<description>Body</description>
</item></channel></rss>""".encode()
        return FetchResult(url, url, 200, "application/rss+xml", body, 0, (), None)


def _operation_root(path: Path) -> Path:
    run = path / "runs" / "20260903T120000Z" / "A_SOURCE"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "BOUNDED_HTTP_REQUESTS": 1,
                "DUPLICATES_ENCOUNTERED": 0,
                "EVENTS_COLLECTED": 1,
                "LIVE_MODEL_PREDICTIONS": 0,
                "LIVE_OUTCOMES_READ": 0,
                "LIVE_POST_EVENT_PRICE_READS": 0,
                "LIVE_TARGETS_COMPUTED": 0,
                "OLD_FUTURE_HOLDOUT_OPENED": False,
                "source_id": "A_SOURCE",
            }
        ),
        encoding="utf-8",
    )
    (run / "source-polls.jsonl").write_text('{"status": "SUCCESS"}\n', encoding="utf-8")
    (run / "live-shadow-corpus.jsonl").write_text(_shadow("A_SOURCE", "AAA"), encoding="utf-8")
    second = path / "runs" / "20260903T130000Z" / "A_SOURCE"
    second.mkdir(parents=True)
    (second / "manifest.json").write_text(
        json.dumps(
            {
                "BOUNDED_HTTP_REQUESTS": 1,
                "DUPLICATES_ENCOUNTERED": 1,
                "EVENTS_COLLECTED": 0,
                "LIVE_MODEL_PREDICTIONS": 0,
                "LIVE_OUTCOMES_READ": 0,
                "LIVE_POST_EVENT_PRICE_READS": 0,
                "LIVE_TARGETS_COMPUTED": 0,
                "OLD_FUTURE_HOLDOUT_OPENED": False,
                "source_id": "A_SOURCE",
            }
        ),
        encoding="utf-8",
    )
    (second / "source-polls.jsonl").write_text('{"status": "NO_NEW_ITEMS"}\n', encoding="utf-8")
    (second / "live-shadow-corpus.jsonl").write_text("", encoding="utf-8")
    (path / "collector-state.json").write_text(
        json.dumps(
            {
                "sources": {
                    "A_SOURCE": {
                        "last_seen_source_identity": "AAA-1",
                        "last_successful_poll_at": "2026-09-03T12:00:00+00:00",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "operation-status.json").write_text(
        """{"LIVE_RESEARCH_OPERATION_STATUS": "READY", "source_results": [], "duplicates": 1}""",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "LIVE_OUTCOMES_READ": 0,
                "LIVE_TARGETS_COMPUTED": 0,
                "LIVE_POST_EVENT_PRICE_READS": 0,
                "BROKER_MUTATIONS": 0,
                "MODEL_TRAINING_PERFORMED": False,
                "MODEL_PREDICTIONS_PERFORMED": False,
            }
        ),
        encoding="utf-8",
    )
    (path / "live-shadow-corpus.jsonl").write_text(_shadow("A_SOURCE", "AAA"), encoding="utf-8")
    return path


def _registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "historical_frozen_issuer_tickers": ["ROSN", "YDEX"],
                "milestone": {
                    "minimum_new_issuer_tickers": 3,
                    "minimum_total_issuer_tickers": 10,
                    "name": "LIVE_DIVERSITY_MILESTONE_V1",
                },
                "source_registry_version": "live-issuer-sources-v1",
                "sources": [
                    {
                        "canonical_domain": "issuer.test",
                        "content_path": ["rss.channel.item.title"],
                        "discovery_type": "official_issuer_rss",
                        "discovery_url": "https://issuer.test/rss",
                        "enabled": True,
                        "expected_publication_frequency": "test",
                        "identity_path": "rss.channel.item.guid",
                        "issuer": "AAA Issuer",
                        "official_domain": "issuer.test",
                        "parser": "rss-item-pubdate-explicit-offset-v1",
                        "parser_type": "rss-item-pubdate-explicit-offset-v1",
                        "polling_policy": {"interval_minutes": 60, "max_items_per_poll": 5},
                        "source_id": "A_SOURCE",
                        "source_origin": "ISSUER_ORIGINATED",
                        "source_status": "LIVE_STRICT_EXACT_READY",
                        "source_version": 1,
                        "stable_identity": "rss_guid",
                        "ticker": "AAA",
                        "ticker_binding": {"binding": "single_issuer_source"},
                        "timestamp_contract": {
                            "evidence_type": "TIMESTAMP_EVIDENCE_TYPE=RFC822_EXPLICIT_OFFSET",
                            "evidence_value": "+0300",
                            "policy": "accept explicit offset",
                        },
                        "timestamp_path": "rss.channel.item.pubDate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _shadow(source_id: str, ticker: str) -> str:
    return (
        json.dumps(
            {
                "TARGET_STATUS": "SEALED",
                "epoch": "LIVE_SHADOW_CORPUS",
                "source_id": source_id,
                "ticker": ticker,
                "semantic_output": {"semantic_unknown": False},
                "pre_event_feature_availability": {"available": True},
            }
        )
        + "\n"
    )
