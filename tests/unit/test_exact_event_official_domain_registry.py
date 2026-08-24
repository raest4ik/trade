from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.cli.enrich_exact_event_official_domain_registry import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_live_source_snapshot.application import (
    build_live_source_snapshot_artifact,
)
from src.exact_event_live_source_snapshot.domain import LiveBlocker
from src.exact_event_live_source_snapshot.http_client import HttpResult
from src.exact_event_official_domain_registry.application import (
    build_official_domain_registry_artifact,
    normalize_host,
    registered_domain,
)
from src.exact_event_official_domain_registry.domain import (
    ARTIFACT_VERSION,
    DOMAIN_DISCOVERY_LIMITS,
    INPUT_DATASET_SHA,
    DomainBlocker,
    sha256_payload,
)


def test_cli_defaults_to_v5_live_report_and_domain_registry_output() -> None:
    args = build_parser().parse_args(["--base-main-sha", "6" * 40])

    assert args.input_dir == "artifacts/exact-event-official-source-discovery-v5"
    assert (
        args.live_source_report
        == "artifacts/exact-event-live-official-source-snapshot-v1/source-report.jsonl"
    )
    assert args.output_dir == "artifacts/exact-event-official-domain-registry-enrichment-v1"


def test_domain_enrichment_accepts_legal_identity_and_runs_second_live(tmp_path: Path) -> None:
    input_root, source_registry, universe, live_report = _write_fixture(tmp_path, ("AAA",))
    seeds = tmp_path / "candidate-domains.jsonl"
    _write_jsonl(
        seeds,
        [
            {
                "ticker": "AAA",
                "domain": "AAA.example",
                "url": "https://aaa.example/about",
                "discovery_origin": "public zero-cost search discovery",
                "evidence_type": "CANDIDATE_WEBSITE",
            }
        ],
    )
    manifest = build_official_domain_registry_artifact(
        input_root=input_root,
        source_registry_path=source_registry,
        universe_path=universe,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="636adf0f796ca052b2ab493438afe58748219ff4",
        git_sha="6" * 40,
        live_source_report_path=live_report,
        candidate_domains_path=seeds,
        client=_FakeHttpClient(
            {
                "https://aaa.example/robots.txt": _ok_robots("https://aaa.example/robots.txt"),
                "https://aaa.example/about": _response(
                    "https://aaa.example/about",
                    "text/html",
                    b"<html>AAA Issuer official investor relations contacts</html>",
                ),
                "https://aaa.example/": _response(
                    "https://aaa.example/",
                    "text/html",
                    b"""
                    <script type="application/ld+json">
                    {"headline":"AAA exact","datePublished":"2026-07-30T11:15:00+03:00",
                     "url":"https://aaa.example/news/1"}
                    </script>
                    """,
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["LIVE_DOMAIN_ENRICHMENT_EXECUTED"] is True
    assert manifest["DOMAIN_CONFIRMED_COUNT"] == 1
    assert manifest["NEWLY_DOMAIN_ENABLED_TICKERS"] == ["AAA"]
    assert manifest["CONFIRMED_DOMAINS_BY_TICKER"] == {"AAA": "aaa.example"}
    assert manifest["SECOND_LIVE_RUN_EXECUTED"] is True
    assert manifest["SECOND_RUN_TICKERS_ATTEMPTED"] == 1
    assert manifest["SECOND_RUN_EXACT_CANDIDATES"] == 1
    assert manifest["DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES"] == 1
    assert manifest["DOWNSTREAM_NEW_EXACT_EVENTS"] == 1
    assert manifest["EXISTING_DOMAIN_ROWS_PRESERVED"] == "PASS"
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["FETCH_TIME_USED_AS_PUBLICATION_TIME"] is False
    assert manifest["TINVEST_READONLY_USED"] is False
    assert manifest["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False

    registry = _read_jsonl(tmp_path / ARTIFACT_VERSION / "official-domain-registry.jsonl")
    assert registry[0]["official_domain"] == "aaa.example"
    assert registry[0]["created_by"] == ARTIFACT_VERSION


def test_fail_closed_rejections_and_blockers(tmp_path: Path) -> None:
    input_root, source_registry, universe, live_report = _write_fixture(
        tmp_path, ("SEA", "MIS", "PAR", "ROB", "RAT")
    )
    seeds = tmp_path / "candidate-domains.jsonl"
    _write_jsonl(
        seeds,
        [
            {
                "ticker": "SEA",
                "domain": "sea.example",
                "evidence_type": "SEARCH_RESULT_ONLY",
            },
            {"ticker": "MIS", "domain": "mis.example", "evidence_type": "CANDIDATE_WEBSITE"},
            {
                "ticker": "PAR",
                "domain": "par.example",
                "evidence_type": "PARENT_SUBSIDIARY_UNPROVEN",
            },
            {"ticker": "ROB", "domain": "rob.example", "evidence_type": "CANDIDATE_WEBSITE"},
            {"ticker": "RAT", "domain": "rat.example", "evidence_type": "CANDIDATE_WEBSITE"},
        ],
    )
    manifest = build_official_domain_registry_artifact(
        input_root=input_root,
        source_registry_path=source_registry,
        universe_path=universe,
        output_root=tmp_path / "blocked",
        base_main_sha="636adf0f796ca052b2ab493438afe58748219ff4",
        git_sha="6" * 40,
        live_source_report_path=live_report,
        candidate_domains_path=seeds,
        client=_FakeHttpClient(
            {
                "https://sea.example/robots.txt": _ok_robots("https://sea.example/robots.txt"),
                "https://sea.example/": _response(
                    "https://sea.example/", "text/html", b"SEA Issuer official site"
                ),
                "https://mis.example/robots.txt": _ok_robots("https://mis.example/robots.txt"),
                "https://mis.example/": _response(
                    "https://mis.example/", "text/html", b"Different legal entity"
                ),
                "https://par.example/robots.txt": _ok_robots("https://par.example/robots.txt"),
                "https://par.example/": _response(
                    "https://par.example/", "text/html", b"PAR Issuer holding parent"
                ),
                "https://rob.example/robots.txt": _response(
                    "https://rob.example/robots.txt",
                    "text/plain",
                    b"User-agent: *\nDisallow: /\n",
                ),
                "https://rat.example/robots.txt": _ok_robots("https://rat.example/robots.txt"),
                "https://rat.example/": HttpResult(
                    "https://rat.example/",
                    "https://rat.example/",
                    429,
                    None,
                    b"",
                    0,
                    DomainBlocker.RATE_LIMITED.value,
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["DOMAIN_CONFIRMED_COUNT"] == 0
    assert manifest["SECOND_LIVE_RUN_EXECUTED"] is False
    assert manifest["DOMAIN_RATE_LIMITED"] == 1
    assert manifest["BLOCKERS_BY_TICKER"]["SEA"] == DomainBlocker.NO_IDENTITY_EVIDENCE.value
    assert manifest["BLOCKERS_BY_TICKER"]["MIS"] == DomainBlocker.LEGAL_ENTITY_MISMATCH.value
    assert manifest["BLOCKERS_BY_TICKER"]["PAR"] == DomainBlocker.PARENT_SUBSIDIARY_AMBIGUITY.value
    assert manifest["BLOCKERS_BY_TICKER"]["ROB"] == DomainBlocker.ROBOTS_BLOCKED.value
    assert manifest["BLOCKERS_BY_TICKER"]["RAT"] == DomainBlocker.RATE_LIMITED.value


def test_registry_hash_deterministic_and_runtime_fetch_time_excluded(tmp_path: Path) -> None:
    first = _build_one_confirmed(tmp_path / "first", datetime(2026, 8, 25, tzinfo=UTC))
    second = _build_one_confirmed(tmp_path / "second", datetime(2026, 8, 26, tzinfo=UTC))

    assert first["DOMAIN_REGISTRY_SHA"] == second["DOMAIN_REGISTRY_SHA"]
    assert first["EVIDENCE_SCHEMA_SHA"] == second["EVIDENCE_SCHEMA_SHA"]
    assert first["DOMAIN_DISCOVERY_LIMITS_SHA"] == sha256_payload(DOMAIN_DISCOVERY_LIMITS)
    assert first["DOMAIN_NETWORK_PROVENANCE_SHA"] != second["DOMAIN_NETWORK_PROVENANCE_SHA"]


def test_domain_normalization_idna_and_live_snapshot_backward_compatibility(
    tmp_path: Path,
) -> None:
    assert normalize_host("https://WWW.Example.RU/investors/") == "www.example.ru"
    assert registered_domain("ir.example.ru") == "example.ru"
    assert normalize_host("https://пример.рф/") == "xn--e1afmkfd.xn--p1ai"

    input_root, source_registry, universe, _live_report = _write_fixture(tmp_path, ("AAA",))
    manifest = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=source_registry,
        universe_path=universe,
        output_root=tmp_path / "old-live",
        base_main_sha="636adf0f796ca052b2ab493438afe58748219ff4",
        git_sha="6" * 40,
        client=_FakeHttpClient({}),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["LIVE_DISCOVERY_EXECUTED"] is False
    assert manifest["LIVE_DISCOVERY_BLOCKER"] == LiveBlocker.NO_OFFICIAL_DOMAIN.value
    assert manifest["LIVE_CANDIDATES_WRITTEN"] == 0


def test_frozen_contracts_and_docs() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA

    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-official-domain-registry-enrichment-v1.md"
    ).read_text(encoding="utf-8")
    assert "DATA_COST_RUB=0" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "TINVEST_READONLY_USED=false" in text
    assert "official domain registry is not a source registry" in text


class _FakeHttpClient:
    def __init__(self, routes: dict[str, HttpResult]) -> None:
        self.routes = routes

    def get(self, url: str) -> HttpResult:
        return self.routes.get(
            url,
            HttpResult(url, url, 404, None, b"", 0, DomainBlocker.HTTP_4XX.value),
        )


def _build_one_confirmed(root: Path, created_at: datetime) -> dict[str, Any]:
    input_root, source_registry, universe, live_report = _write_fixture(root, ("AAA",))
    seeds = root / "candidate-domains.jsonl"
    _write_jsonl(
        seeds,
        [{"ticker": "AAA", "domain": "aaa.example", "evidence_type": "CANDIDATE_WEBSITE"}],
    )
    return build_official_domain_registry_artifact(
        input_root=input_root,
        source_registry_path=source_registry,
        universe_path=universe,
        output_root=root / "artifact",
        base_main_sha="636adf0f796ca052b2ab493438afe58748219ff4",
        git_sha="6" * 40,
        live_source_report_path=live_report,
        candidate_domains_path=seeds,
        client=_FakeHttpClient(
            {
                "https://aaa.example/robots.txt": _ok_robots("https://aaa.example/robots.txt"),
                "https://aaa.example/": _response(
                    "https://aaa.example/", "text/html", b"AAA Issuer official site"
                ),
            }
        ),
        created_at=created_at,
    )


def _write_fixture(
    tmp_path: Path,
    tickers: tuple[str, ...],
) -> tuple[Path, Path, Path, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)
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
    _write_jsonl(input_root / "events.jsonl", [_event_row(ticker) for ticker in tickers])
    _write_jsonl(input_root / "features.jsonl", [_feature_row(ticker) for ticker in tickers])
    _write_jsonl(input_root / "targets.jsonl", [_target_row(ticker) for ticker in tickers])
    _write_jsonl(input_root / "clusters.jsonl", [_cluster_row(ticker) for ticker in tickers])

    source_registry = tmp_path / "source-registry.jsonl"
    _write_jsonl(source_registry, [_source_registry_row(ticker) for ticker in tickers])
    universe = tmp_path / "instrument-mapping.json"
    _write_json(universe, {"instruments": [_instrument(ticker) for ticker in tickers]})
    live_report = tmp_path / "source-report.jsonl"
    _write_jsonl(live_report, [_live_report_row(ticker) for ticker in tickers])
    return input_root, source_registry, universe, live_report


def _event_row(ticker: str) -> dict[str, object]:
    link = f"https://old.example/{ticker}/1"
    return {
        "metadata": {
            "event_id": f"{ticker.lower()}-existing",
            "source_code": f"{ticker}_OLD",
            "source_item_id": link,
            "canonical_url": link,
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": "2026-07-01T07:00:00+00:00",
            "publication_timestamp_raw": "2026-07-01T10:00:00+03:00",
            "publication_date": "2026-07-01",
            "publication_time": "07:00:00",
            "publication_timezone": "UTC",
            "timestamp_source_field": "fixture",
            "timestamp_quality": "EXACT",
            "future_holdout": False,
            "session_state": "DURING_MAIN_SESSION",
            "title_hash": f"{ticker.lower()}-existing",
        },
        "event_features": {"primary_event_type": "OTHER", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": {},
        "target_availability": {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": False,
            "status": "REACTION_READY",
            "missing_reason": None,
        },
        "quality": {"feature_cutoff": "2026-07-01T06:59:00+00:00"},
    }


def _feature_row(ticker: str) -> dict[str, object]:
    return {
        "event_id": f"{ticker.lower()}-existing",
        "feature_cutoff": "2026-07-01T06:59:00+00:00",
        "event_features": {"primary_event_type": "OTHER"},
        "market_features": {},
    }


def _target_row(ticker: str) -> dict[str, object]:
    return {"event_id": f"{ticker.lower()}-existing", "horizons": {}}


def _cluster_row(ticker: str) -> dict[str, object]:
    return {"event_id": f"{ticker.lower()}-existing", "event_cluster_id": f"cluster-{ticker}"}


def _source_registry_row(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": f"uid-{ticker}",
        "official_domain": None,
        "source_url": None,
        "timestamp_capability": "UNKNOWN",
    }


def _instrument(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": f"{ticker} Issuer",
        "class_code": "TQBR",
        "currency": "rub",
        "instrument_uid": f"uid-{ticker}",
    }


def _live_report_row(ticker: str) -> dict[str, object]:
    return {"TICKER": ticker, "ISSUER": f"{ticker} Issuer", "BLOCKER": "NO_OFFICIAL_DOMAIN"}


def _ok_robots(url: str) -> HttpResult:
    return _response(url, "text/plain", b"User-agent: *\nAllow: /\n")


def _response(url: str, content_type: str, body: bytes) -> HttpResult:
    return HttpResult(url, url, 200, content_type, body, 0, None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
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
