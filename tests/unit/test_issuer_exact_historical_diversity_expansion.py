from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import apps.cli.expand_issuer_exact_historical_diversity as cli
import src.issuer_exact_historical_diversity_expansion.application as app
from apps.cli.expand_issuer_exact_historical_diversity import build_parser
from src.events.domain.enums import EventAnalysisStatus, EventType
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_dataset_readiness_audit.domain import artifact_sha as readiness_artifact_sha
from src.exact_event_live_official_collection.http_client import FetchResult
from src.issuer_exact_historical_diversity_expansion.application import (
    build_candidate_sources,
    run_issuer_exact_historical_diversity_expansion,
)
from src.issuer_exact_historical_diversity_expansion.domain import (
    ARTIFACT_VERSION,
    DEFAULT_READINESS_AUDIT_ROOT,
    EXPECTED_RULES_V3_FINGERPRINT,
    CandidateStatus,
    FinalDecision,
    parse_local_timestamp,
    parse_verified_exact_timestamp,
    safety_flags,
    validate_selection_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch


def test_cli_defaults_to_issuer_diversity_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.readiness_dir == DEFAULT_READINESS_AUDIT_ROOT
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"
    assert args.universe_dir == "artifacts/tinvest-market-universe-raw-v1"
    assert args.extra_cache_dir == []
    assert args.live_readonly is False
    assert args.base_main_sha == "8" * 40


def test_cli_wires_lazy_readonly_market_client_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "ARTIFACT_SHA": "sha",
            "SELECTED_SOURCES": 1,
            "NEW_EXACT_EVENTS_COLLECTED": 0,
            "NEW_HISTORICAL_EVENTS_COLLECTED": 0,
            "NEW_FEATURE_READY_EVENTS": 0,
            "DIVERSITY_DECISION": "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING",
        }

    class FakeReadOnlyClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_issuer_exact_historical_diversity_expansion", fake_pipeline)
    monkeypatch.setattr(cli, "_git_sha", lambda: "9" * 40)
    monkeypatch.setattr(cli, "load_readonly_token", lambda: "env-readonly-value")
    monkeypatch.setattr(cli, "TInvestReadOnlyClient", FakeReadOnlyClient)

    args = build_parser().parse_args(
        [
            "--base-main-sha",
            "8" * 40,
            "--live-readonly",
            "--extra-cache-dir",
            "cache-a",
            "--universe-dir",
            "universe-a",
        ]
    )

    assert cli.run(args) == 0

    factory = captured["market_client_factory"]
    assert callable(factory)
    factory()
    assert captured["universe_root"] == Path("universe-a")
    assert captured["extra_cache_roots"] == (Path("cache-a"),)
    assert captured["client_kwargs"]["contour"] == cli.TInvestContour.READONLY_PRODUCTION
    assert captured["client_kwargs"]["max_retries"] == 1


def test_safety_flags_forbid_training_trading_reaction_selection_and_future_reads() -> None:
    flags = safety_flags()

    assert flags["RESEARCH_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["BACKTEST_PERFORMED"] is False
    assert flags["SOURCE_SELECTION_USED_MARKET_OUTCOMES"] is False
    assert flags["SOURCE_SELECTION_USED_MODEL_PERFORMANCE"] is False
    assert flags["FUTURE_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_REACTIONS_COMPUTED"] == 0
    assert flags["FUTURE_TARGETS_COMPUTED"] == 0
    assert flags["FEATURE_DEFINITION_CHANGED"] is False
    assert flags["REACTION_METHODOLOGY_CHANGED"] is False
    assert flags["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert flags["MOEX_RISK_PARAMETERS_SELECTED"] is False
    assert flags["EXCHANGE_ORIGINATED_EVENTS_SELECTED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False


def test_candidate_scoring_prioritizes_absent_exact_issuer_and_rejects_policy_blockers(
    tmp_path: Path,
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    manifest = _read_json(readiness_root / "manifest.json")
    before = app.readiness_summary_metrics(readiness_root, manifest)

    rows = [
        app.score_candidate_for_diversity(candidate, before)
        for candidate in build_candidate_sources(before)
    ]
    by_ticker = {row["ticker"]: row for row in rows}

    assert by_ticker["MVID"]["selection_status"] == "SELECTED"
    assert by_ticker["MVID"]["status"] == CandidateStatus.NEW_EXACT_HISTORICAL_CAPABLE
    assert by_ticker["MTSS"]["selection_status"] == "REJECTED"
    assert by_ticker["MTSS"]["status"] == CandidateStatus.DATE_ONLY
    assert by_ticker["MOEX_RISK"]["selection_status"] == "REJECTED"
    assert by_ticker["MOEX_RISK"]["event_origin"] == "EXCHANGE_ORIGINATED"
    assert int(by_ticker["MVID"]["score"]) > int(by_ticker["MTSS"]["score"])


def test_forbidden_source_selection_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="SOURCE_SELECTION_FORBIDDEN_FIELD:target_return"):
        validate_selection_payload({"ticker": "GOOD", "target_return": 0.01})
    with pytest.raises(ValueError, match="SOURCE_SELECTION_FORBIDDEN_FIELD:auc"):
        validate_selection_payload({"model": {"auc": 0.7}})


def test_bare_local_timestamp_does_not_become_strict_exact_without_timezone_evidence() -> None:
    parsed = parse_local_timestamp("30.04.2026 19:45")

    assert parsed == datetime(2026, 4, 30, 19, 45)
    assert parse_verified_exact_timestamp("30.04.2026 19:45", None) is None
    assert parse_verified_exact_timestamp("30.04.2026 19:45", "UTC+03:00") == datetime(
        2026, 4, 30, 16, 45, tzinfo=UTC
    )
    assert parse_local_timestamp("30.04.2026") is None


def test_bare_timestamp_collects_snapshots_but_skips_exact_market_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeHttpClient(),
        analyzer=_FakeAnalyzer(),
    )

    assert manifest["ARTIFACT_SHA"] == app.artifact_sha(manifest)
    assert manifest["SELECTED_SOURCES"] == 1
    assert manifest["SELECTED_SOURCE_IDS"] == ["MVIDEOELDORADO_IR_NEWS_EXACT_V1"]
    assert manifest["COLLECTED_OFFICIAL_PUBLICATIONS"] == 2
    assert manifest["STRICT_EXACT_TIMEZONE_VERIFIED"] is False
    assert manifest["TIMEZONE_VERIFICATION_STATUS"] == "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"
    assert manifest["NEW_EXACT_EVENTS_COLLECTED"] == 0
    assert manifest["STRICT_EXACT_EVENTS"] == 0
    assert manifest["NEW_HISTORICAL_EVENTS_COLLECTED"] == 1
    assert manifest["NEW_FUTURE_METADATA_ONLY_EVENTS"] == 1
    assert manifest["NEW_SEMANTIC_READY_EVENTS"] == 2
    assert manifest["MARKET_ELIGIBLE_EVENTS"] == 0
    assert manifest["MARKET_NETWORK_REQUESTS"] == 0
    assert manifest["NEW_REACTION_READY_EVENTS"] == 0
    assert manifest["NEW_FEATURE_READY_EVENTS"] == 0
    assert manifest["NEW_ISSUER_TICKERS"] == []
    assert manifest["FEATURE_READY_DELTA"] == 0
    assert manifest["DIVERSITY_DECISION"] == FinalDecision.STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING
    assert manifest["EXACT_V3_PRESERVED"] == "YES"
    assert manifest["QWEN_PRESERVED"] == "YES"
    assert manifest["FEATURE_DEFINITION_CHANGED"] is False
    assert manifest["REACTION_METHODOLOGY_CHANGED"] is False
    assert manifest["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["FUTURE_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_TARGETS_COMPUTED"] == 0
    assert manifest["SOURCE_SELECTION_USED_MARKET_OUTCOMES"] is False

    output_root = tmp_path / "output"
    raw = _read_jsonl(output_root / "raw-publication-snapshots.jsonl")
    assert raw[0]["publication_material_available"] is True
    assert raw[0]["publication_material_sha"]
    assert raw[0]["publication_timestamp_raw"] == "30.04.2026 19:45"
    assert raw[0]["publication_timestamp_utc"] is None
    assert raw[0]["publication_timestamp_quality"] == "LOCAL_TIME_WITHOUT_VERIFIED_TIMEZONE"

    metadata = _read_jsonl(output_root / "collected-event-metadata.jsonl")
    assert {row["event_origin"] for row in metadata} == {"ISSUER_ORIGINATED"}
    assert [row["future_holdout"] for row in metadata] == [False, True]
    assert {row["strict_exact_event"] for row in metadata} == {False}

    semantics = _read_jsonl(output_root / "semantic-extraction-results.jsonl")
    assert all(row["semantic_ready"] for row in semantics)
    assert all(row["synthetic_unknown_features_used"] is False for row in semantics)
    assert {row["primary_event_type"] for row in semantics} == {"FINANCIAL_RESULTS"}
    assert all(
        set(row["semantic_features"]) == {"primary_event_type", "event_count", "fact_count"}
        for row in semantics
    )

    market = _read_jsonl(output_root / "market-acquisition-provenance.jsonl")
    assert {row["status"] for row in market} == {"SKIPPED_STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"}
    assert all(row["future_price_lookup"] is False for row in market)

    maturation = _read_jsonl(output_root / "maturation-results.jsonl")
    assert {row["primary_readiness_blocker"] for row in maturation} == {
        "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"
    }

    diversity = _read_json(output_root / "diversity-before-after.json")
    assert diversity["new_absent_before_tickers"] == []
    assert diversity["top_ticker_share_delta"] == "0.000000"


def test_verified_timezone_reuses_pr48_gate_cache_and_frozen_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )
    cache_root = _write_minute_cache(tmp_path / "cache")
    universe_root = _write_universe(tmp_path / "universe")

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeVerifiedTimezoneHttpClient(),
        analyzer=_FakeAnalyzer(),
        extra_cache_roots=(cache_root,),
        universe_root=universe_root,
    )

    assert manifest["STRICT_EXACT_TIMEZONE_VERIFIED"] is True
    assert manifest["STRICT_EXACT_EVENTS"] == 1
    assert manifest["MARKET_ELIGIBLE_EVENTS"] == 1
    assert manifest["MARKET_NETWORK_REQUESTS"] == 0
    assert manifest["NEW_REACTION_READY_EVENTS"] == 1
    assert manifest["NEW_FEATURE_READY_EVENTS"] == 1
    assert manifest["DIVERSITY_DECISION"] == FinalDecision.ISSUER_DIVERSITY_GAIN_MODEST

    eligibility = _read_jsonl(tmp_path / "output" / "market-eligibility.jsonl")
    assert eligibility[0]["pr48_eligibility_gate_reused"] is True
    assert eligibility[0]["market_reaction_eligibility"] == "ELIGIBLE"
    assert eligibility[0]["instrument_uid"] == "uid-mvid"
    identities = _read_jsonl(tmp_path / "output" / "instrument-identity-provenance.jsonl")
    assert {row["ticker"] for row in identities} == {"IMOEX", "MVID"}
    maturation = _read_jsonl(tmp_path / "output" / "maturation-results.jsonl")
    historical = next(row for row in maturation if row["historical_or_future"] == "HISTORICAL")
    assert historical["strict_feature_timestamp_at_or_before_publication"] is True
    assert historical["max_feature_timestamp_utc"] == "2026-04-30T16:45:00+00:00"


def test_bounded_acquisition_deduplicates_days_and_uses_readonly_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )
    universe_root = _write_universe(tmp_path / "universe")
    cache_root = _write_minute_cache(tmp_path / "event-day-cache")
    client = _FakeMarketClient()
    factory_calls = 0

    def factory() -> _FakeMarketClient:
        nonlocal factory_calls
        factory_calls += 1
        return client

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeDuplicateDayVerifiedHttpClient(),
        analyzer=_FakeAnalyzer(),
        market_client_factory=factory,
        extra_cache_roots=(cache_root,),
        universe_root=universe_root,
    )

    assert factory_calls == 1
    assert manifest["MARKET_NETWORK_REQUESTS"] == 14
    assert manifest["UNIQUE_MARKET_DAYS_REQUESTED"] == 7
    assert client.requested == [
        ("uid-imoex", "2026-04-23"),
        ("uid-imoex", "2026-04-24"),
        ("uid-imoex", "2026-04-25"),
        ("uid-imoex", "2026-04-26"),
        ("uid-imoex", "2026-04-27"),
        ("uid-imoex", "2026-04-28"),
        ("uid-imoex", "2026-04-29"),
        ("uid-mvid", "2026-04-23"),
        ("uid-mvid", "2026-04-24"),
        ("uid-mvid", "2026-04-25"),
        ("uid-mvid", "2026-04-26"),
        ("uid-mvid", "2026-04-27"),
        ("uid-mvid", "2026-04-28"),
        ("uid-mvid", "2026-04-29"),
    ]
    acquisition = _read_jsonl(tmp_path / "output" / "market-history-acquisition.jsonl")
    assert all(row["broker_write_surface_used"] is False for row in acquisition)
    assert all(row["token_value_read"] is False for row in acquisition)


def test_market_client_factory_not_called_for_timezone_blocked_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )
    calls = 0

    def factory() -> _FakeMarketClient:
        nonlocal calls
        calls += 1
        return _FakeMarketClient()

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeHttpClient(),
        analyzer=_FakeAnalyzer(),
        market_client_factory=factory,
    )

    assert calls == 0
    assert manifest["MARKET_NETWORK_REQUESTS"] == 0


def test_current_tradable_without_event_date_history_does_not_claim_event_date_trading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )
    universe_root = _write_universe(tmp_path / "universe")

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeVerifiedTimezoneHttpClient(),
        analyzer=_FakeAnalyzer(),
        universe_root=universe_root,
    )

    assert manifest["MARKET_ELIGIBLE_EVENTS"] == 0
    assert manifest["MARKET_NETWORK_REQUESTS"] == 0
    eligibility = _read_jsonl(tmp_path / "output" / "market-eligibility.jsonl")
    historical = next(row for row in eligibility if row["event_validity"] == "VALID_EXACT_EVENT")
    assert historical["api_trade_available"] is True
    assert historical["event_date_trading_confirmed"] is None
    assert historical["market_reaction_eligibility"] == "SECURITY_HISTORY_UNAVAILABLE"


def test_unrelated_iso_timestamp_does_not_verify_publication_timezone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeUnrelatedIsoHttpClient(),
        analyzer=_FakeAnalyzer(),
    )

    assert manifest["STRICT_EXACT_TIMEZONE_VERIFIED"] is False
    assert manifest["STRICT_EXACT_EVENTS"] == 0
    raw = _read_jsonl(tmp_path / "output" / "raw-publication-snapshots.jsonl")
    assert raw[0]["publication_timestamp_raw"] == "30.04.2026 19:45"
    assert raw[0]["publication_timestamp_utc"] is None


def test_publication_datepublished_offset_verifies_timezone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )

    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeDatePublishedHttpClient(),
        analyzer=_FakeAnalyzer(),
    )

    assert manifest["STRICT_EXACT_TIMEZONE_VERIFIED"] is True
    raw = _read_jsonl(tmp_path / "output" / "raw-publication-snapshots.jsonl")
    assert raw[0]["publication_timestamp_raw"] == "30.04.2026 19:45"
    assert raw[0]["publication_timestamp_utc"] == "2026-04-30T16:45:00+00:00"
    timezone = _read_jsonl(tmp_path / "output" / "timezone-evidence.jsonl")
    assert timezone[0]["TIMEZONE_EVIDENCE_SOURCE"] == "STRUCTURED_PUBLICATION_METADATA"


def test_canonical_warmup_window_recovers_early_session_pre_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    monkeypatch.setattr(
        app,
        "EXPECTED_READINESS_AUDIT_SHA",
        _read_json(readiness_root / "manifest.json")["ARTIFACT_SHA"],
    )
    universe_root = _write_universe(tmp_path / "universe")
    event_day_only = _write_early_session_cache(tmp_path / "event-day-only", include_warmup=False)
    with_warmup = _write_early_session_cache(tmp_path / "with-warmup", include_warmup=True)

    event_day_manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "event-day-output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeEarlySessionVerifiedHttpClient(),
        analyzer=_FakeAnalyzer(),
        extra_cache_roots=(event_day_only,),
        universe_root=universe_root,
    )
    warmup_manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=readiness_root,
        output_root=tmp_path / "warmup-output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FakeEarlySessionVerifiedHttpClient(),
        analyzer=_FakeAnalyzer(),
        extra_cache_roots=(with_warmup,),
        universe_root=universe_root,
    )

    assert event_day_manifest["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert event_day_manifest["NEW_FEATURE_READY_EVENTS"] == 0
    assert warmup_manifest["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert warmup_manifest["NEW_FEATURE_READY_EVENTS"] == 1
    maturation = _read_jsonl(tmp_path / "warmup-output" / "maturation-results.jsonl")
    historical = next(row for row in maturation if row["historical_or_future"] == "HISTORICAL")
    features = historical["pre_event_market_features"]
    assert all(features[f"pre_return_{minutes}m"] is not None for minutes in (5, 15, 30, 60))
    assert all(features[f"imoex_pre_return_{minutes}m"] is not None for minutes in (5, 15, 30, 60))


def test_feature_timestamp_equality_is_allowed_and_future_is_rejected() -> None:
    published = datetime(2026, 4, 30, 16, 45, tzinfo=UTC)

    assert app.feature_timestamp_passes_leakage_guard(published, published) is True
    assert (
        app.feature_timestamp_passes_leakage_guard(published + timedelta(microseconds=1), published)
        is False
    )


def test_readiness_manifest_sha_is_enforced(tmp_path: Path) -> None:
    readiness_root = _write_readiness_artifact(tmp_path / "readiness")
    manifest = _read_json(readiness_root / "manifest.json")
    manifest["ARTIFACT_SHA"] = "bad"
    _write_json(readiness_root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="READINESS_AUDIT_SHA_MISMATCH"):
        run_issuer_exact_historical_diversity_expansion(
            readiness_root=readiness_root,
            output_root=tmp_path / "output",
            base_main_sha="8" * 40,
            git_sha="9" * 40,
            http_client=_FakeHttpClient(),
            analyzer=_FakeAnalyzer(),
        )


def test_rules_v3_fingerprint_is_frozen_for_diversity_expansion() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_V3_FINGERPRINT


def test_expansion_module_has_no_model_training_dependency() -> None:
    source = Path(app.__file__).read_text(encoding="utf-8")

    assert "sklearn" not in source
    assert ".fit(" not in source
    assert "predict(" not in source
    assert "accuracy" not in source.lower()
    assert "auc" not in source.lower()
    assert "rmse" not in source.lower()


class _FakeAnalyzer:
    def analyze(self, *, news_id: Any, raw_content: str) -> Any:
        return SimpleNamespace(
            status=EventAnalysisStatus.COMPLETE,
            primary_event_type=EventType.FINANCIAL_RESULTS,
            events=[SimpleNamespace(event_type=EventType.FINANCIAL_RESULTS)],
            financial_facts=[],
        )


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        listing_url = "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/investor-news"
        bodies = {
            listing_url: (
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/100">A</a>'
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/101">B</a>'
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/102">C</a>'
            ),
            f"{listing_url}/detail/100": """
                <html><h1>M.Video Reports 2025 Operating and Financial Results</h1>
                <main>30.04.2026 19:45 M.Video announces results IFRS and reports revenue growth.
                Moscow Exchange ticker: MVID.</main></html>
            """,
            f"{listing_url}/detail/101": """
                <html><h1>M.Video Future Holdout Release</h1>
                <main>12.08.2026 12:00 M.Video announces results IFRS after holdout start.
                Moscow Exchange ticker: MVID.</main></html>
            """,
            f"{listing_url}/detail/102": """
                <html><h1>M.Video Date Only Release</h1>
                <main>30.04.2026 Date-only publication card without exact time.</main></html>
            """,
        }
        body = bodies.get(url, "")
        return FetchResult(
            request_url=url,
            final_url=url,
            status=200 if body else 404,
            content_type="text/html",
            body=body.encode("utf-8"),
            redirects=0,
            redirect_chain=(),
            blocker=None if body else "HTTP_FAILURE",
        )


class _FakeVerifiedTimezoneHttpClient(_FakeHttpClient):
    def get(self, url: str) -> FetchResult:
        result = super().get(url)
        if result.status != 200:
            return result
        body = result.body.decode("utf-8").replace("30.04.2026 19:45", "30.04.2026 19:45 UTC+03:00")
        body = body.replace("12.08.2026 12:00", "12.08.2026 12:00 UTC+03:00")
        return FetchResult(
            request_url=result.request_url,
            final_url=result.final_url,
            status=result.status,
            content_type=result.content_type,
            body=body.encode("utf-8"),
            redirects=result.redirects,
            redirect_chain=result.redirect_chain,
            blocker=result.blocker,
        )


class _FakeDuplicateDayVerifiedHttpClient(_FakeVerifiedTimezoneHttpClient):
    def get(self, url: str) -> FetchResult:
        listing_url = (
            "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/"
            "investor-news"
        )
        if url == listing_url:
            body = (
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/100">A</a>'
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/103">D</a>'
            )
            return FetchResult(url, url, 200, "text/html", body.encode(), 0, (), None)
        if url == f"{listing_url}/detail/103":
            body = """
                <html><h1>M.Video Same Day Release</h1>
                <main>30.04.2026 20:15 UTC+03:00 M.Video announces results IFRS.
                Moscow Exchange ticker: MVID.</main></html>
            """
            return FetchResult(url, url, 200, "text/html", body.encode(), 0, (), None)
        return super().get(url)


class _SingleDetailHttpClient(_FakeHttpClient):
    def __init__(self, detail_body: str) -> None:
        super().__init__()
        self._detail_body = detail_body

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        listing_url = (
            "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/"
            "investor-news"
        )
        if url == listing_url:
            body = (
                '<a href="/en/shareholders-and-investors/news-and-events/'
                'investor-news/detail/100">A</a>'
            )
            return FetchResult(url, url, 200, "text/html", body.encode(), 0, (), None)
        if url == f"{listing_url}/detail/100":
            return FetchResult(url, url, 200, "text/html", self._detail_body.encode(), 0, (), None)
        return FetchResult(url, url, 404, "text/html", b"", 0, (), "HTTP_FAILURE")


class _FakeUnrelatedIsoHttpClient(_SingleDetailHttpClient):
    def __init__(self) -> None:
        super().__init__(
            """
            <html><h1>M.Video Reports 2025 Operating and Financial Results</h1>
            <main>30.04.2026 19:45 M.Video announces results IFRS.</main>
            <script>window.assetBuiltAt="2026-04-30T19:45:00+03:00"</script>
            </html>
            """
        )


class _FakeDatePublishedHttpClient(_SingleDetailHttpClient):
    def __init__(self) -> None:
        super().__init__(
            """
            <html><h1>M.Video Reports 2025 Operating and Financial Results</h1>
            <script type="application/ld+json">
            {"@type":"NewsArticle","datePublished":"2026-04-30T19:45:00+03:00"}
            </script>
            <main>30.04.2026 19:45 M.Video announces results IFRS.</main>
            </html>
            """
        )


class _FakeEarlySessionVerifiedHttpClient(_SingleDetailHttpClient):
    def __init__(self) -> None:
        super().__init__(
            """
            <html><h1>M.Video Early Session Release</h1>
            <main>30.04.2026 10:02 UTC+03:00 M.Video announces results IFRS.</main>
            </html>
            """
        )


class _FakeMarketClient:
    def __init__(self) -> None:
        self.requested: list[tuple[str, str]] = []

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch:
        _ = date_to
        self.requested.append((instrument_uid, date_from.date().isoformat()))
        return TInvestMinuteCandleBatch(tuple(_candles(instrument_uid, date_from)), ())


def _write_readiness_artifact(root: Path) -> Path:
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "exact-dataset-readiness-audit-v1",
        "FEATURE_READY_EVENTS": 10,
        "ISSUER_ORIGINATED_FEATURE_READY": 8,
        "EXCHANGE_ORIGINATED_FEATURE_READY": 2,
        "UNKNOWN_RATE_TOTAL": "0.100000",
        "MOEX_RISK_UNKNOWN_RATE": "1.000000",
        "TOP_TICKER_SHARE": "0.500000",
        "TOP_3_TICKER_SHARE": "0.800000",
        "TOP_5_TICKER_SHARE": "1.000000",
        "TICKER_HHI": "0.340000",
        "EFFECTIVE_TICKER_COUNT": "2.941176",
        "SOURCE_FAMILY_HHI": "0.340000",
        "SOURCE_ID_HHI": "0.340000",
        "EVENT_ORIGIN_HHI": "0.680000",
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
    }
    manifest["ARTIFACT_SHA"] = readiness_artifact_sha(manifest)
    _write_json(root / "manifest.json", manifest)
    _write_jsonl(
        root / "ticker-summary.jsonl",
        [
            {"ticker": "MGNT", "feature_ready": 5, "share": "0.500000"},
            {"ticker": "X5", "feature_ready": 2, "share": "0.200000"},
            {"ticker": "T", "feature_ready": 1, "share": "0.100000"},
            {"ticker": "MOEX", "feature_ready": 2, "share": "0.200000"},
        ],
    )
    _write_jsonl(
        root / "source-family-summary.jsonl",
        [
            {"source_family": "MAGNIT_OFFICIAL_JSON_EXACT", "feature_ready": 5},
            {"source_family": "X5_OFFICIAL_RSS_EXACT", "feature_ready": 2},
            {"source_family": "TBANK_OFFICIAL_EXACT", "feature_ready": 1},
            {
                "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
                "feature_ready": 2,
            },
        ],
    )
    _write_jsonl(
        root / "event-origin-summary.jsonl",
        [
            {"event_origin": "ISSUER_ORIGINATED", "feature_ready_count": 8},
            {"event_origin": "EXCHANGE_ORIGINATED", "feature_ready_count": 2},
        ],
    )
    _write_jsonl(root / "cohort-a-issuer-event-ids.jsonl", [])
    _write_jsonl(
        root / "cohort-c-all-event-ids.jsonl",
        [
            {"event_id": f"mgnt-{index}", "source_id": "MAGNIT_OFFICIAL_JSON_EXACT"}
            for index in range(5)
        ]
        + [{"event_id": f"x5-{index}", "source_id": "X5_OFFICIAL_RSS_EXACT"} for index in range(2)]
        + [{"event_id": "t-0", "source_id": "TBANK_OFFICIAL_EXACT"}]
        + [
            {
                "event_id": f"moex-{index}",
                "source_id": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            }
            for index in range(2)
        ],
    )
    return root


def _write_minute_cache(root: Path) -> Path:
    event_at = datetime(2026, 4, 30, 16, 45, tzinfo=UTC)
    begins = [event_at - timedelta(minutes=61) + timedelta(minutes=index) for index in range(121)]
    for ticker in ("MVID", "IMOEX"):
        rows: list[dict[str, Any]] = []
        for index, begin in enumerate(begins, start=1):
            price = 100 + index if ticker == "MVID" else 200 + index
            rows.append(
                {
                    "source": "TINVEST_API",
                    "instrument_uid": "uid-mvid" if ticker == "MVID" else "uid-imoex",
                    "begin_at": begin.isoformat(),
                    "end_at": (begin + timedelta(minutes=1)).isoformat(),
                    "open": str(price),
                    "high": str(price + 1),
                    "low": str(price - 1),
                    "close": str(price),
                    "volume": 1000 + index,
                    "is_complete": True,
                }
            )
        _write_jsonl(root / ticker / "2026-04-30-day.jsonl", rows)
    return root


def _write_early_session_cache(root: Path, *, include_warmup: bool) -> Path:
    event_at = datetime(2026, 4, 30, 7, 2, tzinfo=UTC)
    event_day_begins = [
        event_at - timedelta(minutes=2) + timedelta(minutes=index) for index in range(63)
    ]
    warmup_begins = [datetime(2026, 4, 29, 23, 58, tzinfo=UTC)]
    for ticker in ("MVID", "IMOEX"):
        rows: list[dict[str, Any]] = []
        begins = [*(warmup_begins if include_warmup else []), *event_day_begins]
        for index, begin in enumerate(begins, start=1):
            price = 100 + index if ticker == "MVID" else 200 + index
            rows.append(
                {
                    "source": "TINVEST_API",
                    "instrument_uid": "uid-mvid" if ticker == "MVID" else "uid-imoex",
                    "begin_at": begin.isoformat(),
                    "end_at": (begin + timedelta(minutes=1)).isoformat(),
                    "open": str(price),
                    "high": str(price + 1),
                    "low": str(price - 1),
                    "close": str(price),
                    "volume": 1000 + index,
                    "is_complete": True,
                }
            )
        event_day_rows = [row for row in rows if str(row["begin_at"]).startswith("2026-04-30")]
        _write_jsonl(root / ticker / "2026-04-30-day.jsonl", event_day_rows)
        if include_warmup:
            warmup_rows = [row for row in rows if str(row["begin_at"]).startswith("2026-04-29")]
            _write_jsonl(root / ticker / "2026-04-29-day.jsonl", warmup_rows)
    return root


def _write_universe(root: Path) -> Path:
    _write_jsonl(
        root / "history-coverage.jsonl",
        [
            {
                "ticker": "MVID",
                "instrument_uid": "uid-mvid",
                "figi": "figi-mvid",
                "class_code": "TQBR",
                "structurally_eligible": True,
                "historical_candle_available": True,
                "api_trade_available_flag": True,
                "buy_available_flag": True,
                "sell_available_flag": True,
                "trading_status": "SECURITY_TRADING_STATUS_NORMAL_TRADING",
                "last_1day_candle_date": "2026-08-10",
            }
        ],
    )
    _write_json(
        root / "instrument-mapping.json",
        {
            "instruments": [
                {
                    "ticker": "IMOEX",
                    "instrument_uid": "uid-imoex",
                    "figi": None,
                    "class_code": "INDX",
                    "instrument_type": "index",
                }
            ]
        },
    )
    return root


def _candles(instrument_uid: str, date_from: datetime) -> list[TInvestMinuteCandle]:
    event_at = datetime.combine(date_from.date(), datetime.min.time(), UTC) + timedelta(
        hours=16, minutes=45
    )
    rows: list[TInvestMinuteCandle] = []
    for index in range(121):
        begin = event_at - timedelta(minutes=61) + timedelta(minutes=index)
        price = Decimal(100 + index)
        rows.append(
            TInvestMinuteCandle(
                instrument_uid=instrument_uid,
                begin_at=begin,
                end_at=begin + timedelta(minutes=1),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
                volume=1000 + index,
                is_complete=True,
            )
        )
    return rows


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
