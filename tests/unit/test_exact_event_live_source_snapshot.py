from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.cli.acquire_exact_event_live_source_snapshot import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_live_source_snapshot.application import (
    build_live_source_snapshot_artifact,
)
from src.exact_event_live_source_snapshot.domain import (
    ARTIFACT_VERSION,
    INPUT_DATASET_SHA,
    MAX_RESPONSE_BYTES,
    MAX_TICKERS,
    NETWORK_LIMITS,
    STANDARD_PATH_PROBES,
    LiveBlocker,
    sha256_payload,
)
from src.exact_event_live_source_snapshot.http_client import HttpResult


def test_cli_defaults_to_v5_inputs_and_live_snapshot_output() -> None:
    args = build_parser().parse_args(["--base-main-sha", "c" * 40])

    assert args.input_dir == "artifacts/exact-event-official-source-discovery-v5"
    assert (
        args.source_registry
        == "artifacts/exact-event-official-source-discovery-v5/source-registry.jsonl"
    )
    assert args.universe == "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
    assert args.output_dir == "artifacts/exact-event-live-official-source-snapshot-v1"


def test_live_snapshot_writes_v5_compatible_cache_and_runs_downstream(tmp_path: Path) -> None:
    input_root, registry_path, universe_path = _write_fixture(tmp_path)
    manifest = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="c" * 40,
        client=_FakeHttpClient(
            {
                "https://aaa.example/robots.txt": _response(
                    "https://aaa.example/robots.txt", "text/plain", b"User-agent: *\nAllow: /\n"
                ),
                "https://aaa.example/": _response(
                    "https://aaa.example/",
                    "text/html",
                    b"""
                    <html><head>
                    <link rel="alternate" type="application/rss+xml" href="/rss">
                    <script type="application/ld+json">
                    {"@type":"NewsArticle","headline":"AAA exact JSONLD",
                     "datePublished":"2026-07-30T11:15:00+03:00",
                     "url":"https://aaa.example/news/exact-jsonld"}
                    </script></head><body><a href="/investors/news">news</a></body></html>
                    """,
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["LIVE_DISCOVERY_EXECUTED"] is True
    assert manifest["LIVE_DISCOVERY_BLOCKER"] is None
    assert manifest["LIVE_TICKERS_ATTEMPTED"] <= MAX_TICKERS
    assert manifest["LIVE_DOMAINS_CONFIRMED"] == 1
    assert manifest["LIVE_REQUESTS_TOTAL"] > 0
    assert manifest["LIVE_HTTP_2XX"] >= 2
    assert manifest["LIVE_PAGES_PARSED"] >= 1
    assert manifest["LIVE_FEEDS_FOUND"] == 1
    assert manifest["LIVE_JSONLD_SOURCES_FOUND"] == 1
    assert manifest["LIVE_CANDIDATES_WRITTEN"] == 1
    assert manifest["LIVE_EXACT_CANDIDATES"] == 1
    assert manifest["V5_NEW_EXACT_CAPABLE_SOURCES"] == 1
    assert manifest["V5_NEW_EXACT_EVENTS"] == 1
    assert manifest["V5_NEW_EXACT_HISTORICAL"] == 1
    assert manifest["V5_NEW_EXACT_FUTURE_METADATA_ONLY"] == 0
    assert manifest["V5_NEW_EXACT_TICKERS"] == ["AAA"]
    assert manifest["SOURCE_CANDIDATE_DEDUPE"] == "PASS"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_TARGET_ROWS_PRESERVED"] == "PASS"
    assert manifest["DATE_ONLY_COERCIONS"] == 0
    assert manifest["FETCH_TIME_USED_AS_PUBLICATION_TIME"] is False
    assert manifest["TINVEST_READONLY_USED"] is False
    assert manifest["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["TEST_OUTCOME_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False

    candidate = _read_json(
        tmp_path / ARTIFACT_VERSION / "live-source-snapshot-cache" / "AAA" / "candidate-000.json"
    )
    assert candidate["source_url"] == "https://aaa.example/"
    assert candidate["official_source_confirmed"] is True
    assert candidate["timestamp_capability"] == "EXACT"
    assert candidate["items"][0]["published_at"] == "2026-07-30T11:15:00+03:00"


def test_rss_atom_date_only_and_future_metadata_paths(tmp_path: Path) -> None:
    input_root, registry_path, universe_path = _write_fixture(
        tmp_path, tickers=("RSS", "ATOM", "DATE")
    )
    manifest = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "feeds",
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="c" * 40,
        client=_FakeHttpClient(
            {
                "https://rss.example/robots.txt": _ok_robots("https://rss.example/robots.txt"),
                "https://rss.example/rss": _response(
                    "https://rss.example/rss",
                    "application/rss+xml",
                    b"""<rss><channel><item><title>RSS exact</title>
                    <link>https://rss.example/news/1</link>
                    <pubDate>Thu, 13 Aug 2026 11:15:00 +0300</pubDate>
                    </item></channel></rss>""",
                ),
                "https://atom.example/robots.txt": _ok_robots("https://atom.example/robots.txt"),
                "https://atom.example/feed": _response(
                    "https://atom.example/feed",
                    "application/atom+xml",
                    b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
                    <title>Atom exact</title><link href="https://atom.example/news/1"/>
                    <published>2026-07-30T11:15:00+03:00</published></entry></feed>""",
                ),
                "https://date.example/robots.txt": _ok_robots("https://date.example/robots.txt"),
                "https://date.example/": _response(
                    "https://date.example/",
                    "application/ld+json",
                    b'{"headline":"Date only","datePublished":"2026-07-30","url":"https://date.example/n"}',
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["LIVE_EXACT_CANDIDATES"] == 2
    assert manifest["LIVE_DATE_ONLY_CANDIDATES"] == 1
    assert manifest["V5_NEW_EXACT_EVENTS"] == 2
    assert manifest["V5_NEW_EXACT_HISTORICAL"] == 1
    assert manifest["V5_NEW_EXACT_FUTURE_METADATA_ONLY"] == 1


def test_blockers_are_fail_closed_and_do_not_write_candidates(tmp_path: Path) -> None:
    input_root, registry_path, universe_path = _write_fixture(
        tmp_path, tickers=("ROB", "AUTH", "CAP", "PAY", "RATE", "BIG", "BIN", "DNS")
    )
    manifest = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "blocked",
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="c" * 40,
        client=_FakeHttpClient(
            {
                "https://rob.example/robots.txt": _response(
                    "https://rob.example/robots.txt", "text/plain", b"User-agent: *\nDisallow: /\n"
                ),
                "https://auth.example/robots.txt": _ok_robots("https://auth.example/robots.txt"),
                "https://auth.example/": HttpResult(
                    "https://auth.example/",
                    "https://auth.example/",
                    401,
                    None,
                    b"",
                    0,
                    LiveBlocker.AUTH_REQUIRED.value,
                ),
                "https://cap.example/robots.txt": _ok_robots("https://cap.example/robots.txt"),
                "https://cap.example/": HttpResult(
                    "https://cap.example/",
                    "https://cap.example/",
                    403,
                    None,
                    b"",
                    0,
                    LiveBlocker.CAPTCHA_BLOCKED.value,
                ),
                "https://pay.example/robots.txt": _ok_robots("https://pay.example/robots.txt"),
                "https://pay.example/": HttpResult(
                    "https://pay.example/",
                    "https://pay.example/",
                    402,
                    None,
                    b"",
                    0,
                    LiveBlocker.PAYMENT_REQUIRED.value,
                ),
                "https://rate.example/robots.txt": _ok_robots("https://rate.example/robots.txt"),
                "https://rate.example/": HttpResult(
                    "https://rate.example/",
                    "https://rate.example/",
                    429,
                    None,
                    b"",
                    0,
                    LiveBlocker.RATE_LIMITED.value,
                ),
                "https://big.example/robots.txt": _ok_robots("https://big.example/robots.txt"),
                "https://big.example/": HttpResult(
                    "https://big.example/",
                    "https://big.example/",
                    200,
                    "text/html",
                    b"",
                    0,
                    LiveBlocker.RESPONSE_TOO_LARGE.value,
                ),
                "https://bin.example/robots.txt": _ok_robots("https://bin.example/robots.txt"),
                "https://bin.example/": _response(
                    "https://bin.example/", "application/octet-stream", b"\x00\x01"
                ),
                "https://dns.example/robots.txt": HttpResult(
                    "https://dns.example/robots.txt",
                    "https://dns.example/robots.txt",
                    None,
                    None,
                    b"",
                    0,
                    LiveBlocker.DNS_FAILED.value,
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["LIVE_ROBOTS_BLOCKED"] == 1
    assert manifest["LIVE_AUTH_BLOCKED"] == 1
    assert manifest["LIVE_CAPTCHA_BLOCKED"] == 1
    assert manifest["LIVE_RATE_LIMITED"] == 1
    assert manifest["LIVE_TECHNICAL_FAILURES"] >= 3
    assert manifest["LIVE_CANDIDATES_WRITTEN"] == 0
    assert manifest["V5_NEW_EXACT_EVENTS"] == 0
    assert "https://dns.example/" not in {
        row["REQUEST_URL"] for row in manifest["NETWORK_PROVENANCE"]
    }
    assert any(
        row["TICKER"] == "DNS" and row["BLOCKER"] == LiveBlocker.DNS_FAILED.value
        for row in manifest["PER_SOURCE_REPORT"]
    )


def test_sitemap_json_and_snapshot_hash_are_deterministic(tmp_path: Path) -> None:
    input_root, registry_path, universe_path = _write_fixture(tmp_path, tickers=("MAP",))
    routes = {
        "https://map.example/robots.txt": _ok_robots("https://map.example/robots.txt"),
        "https://map.example/": _response(
            "https://map.example/",
            "text/html",
            b'<a href="/sitemap.xml">sitemap</a>',
        ),
        "https://map.example/sitemap.xml": _response(
            "https://map.example/sitemap.xml",
            "application/xml",
            b"<urlset><url><loc>https://map.example/news/feed.json</loc></url></urlset>",
        ),
        "https://map.example/news/feed.json": _response(
            "https://map.example/news/feed.json",
            "application/json",
            b'{"items":[{"title":"JSON exact","published_at":"2026-07-30T11:15:00+03:00",'
            b'"url":"https://map.example/news/1"}]}',
        ),
    }
    first = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "first",
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="c" * 40,
        client=_FakeHttpClient(routes),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )
    second = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "second",
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="d" * 40,
        client=_FakeHttpClient(routes),
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )

    assert first["LIVE_SITEMAPS_FOUND"] == 1
    assert first["LIVE_EXACT_CANDIDATES"] == 1
    assert first["LIVE_SOURCE_SNAPSHOT_SHA"] == second["LIVE_SOURCE_SNAPSHOT_SHA"]
    assert first["CACHE_SCHEMA_SHA"] == second["CACHE_SCHEMA_SHA"]
    assert first["NETWORK_LIMITS_SHA"] == sha256_payload(NETWORK_LIMITS)
    assert first["STANDARD_PATH_PROBES_SHA"] == sha256_payload(STANDARD_PATH_PROBES)


def test_network_unavailable_does_not_claim_live_discovery_executed(tmp_path: Path) -> None:
    input_root, registry_path, universe_path = _write_fixture(tmp_path, tickers=("DNS",))
    manifest = build_live_source_snapshot_artifact(
        input_root=input_root,
        source_registry_path=registry_path,
        universe_path=universe_path,
        output_root=tmp_path / "network-unavailable",
        base_main_sha="c1e07932b9176e4e1111827320123860fa5f748a",
        git_sha="c" * 40,
        client=_FakeHttpClient(
            {
                "https://dns.example/robots.txt": HttpResult(
                    "https://dns.example/robots.txt",
                    "https://dns.example/robots.txt",
                    None,
                    None,
                    b"",
                    0,
                    LiveBlocker.DNS_FAILED.value,
                ),
            }
        ),
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert manifest["LIVE_DISCOVERY_EXECUTED"] is False
    assert manifest["LIVE_DISCOVERY_BLOCKER"] == LiveBlocker.NETWORK_UNAVAILABLE.value
    assert manifest["SOURCE_ABSENCE_CONCLUSION_ALLOWED"] is False
    assert manifest["LIVE_REQUESTS_TOTAL"] == 1


def test_frozen_contracts_and_docs() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert MAX_RESPONSE_BYTES == 1_000_000

    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-live-official-source-snapshot-v1.md"
    ).read_text(encoding="utf-8")
    assert "TINVEST_READONLY_USED=false" in text
    assert "FETCH_TIME_USED_AS_PUBLICATION_TIME=false" in text
    assert "no market candles" in text
    assert "no sparse label family" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text


class _FakeHttpClient:
    def __init__(self, routes: dict[str, HttpResult]) -> None:
        self.routes = routes

    def get(self, url: str) -> HttpResult:
        return self.routes.get(
            url,
            HttpResult(url, url, 404, None, b"", 0, LiveBlocker.HTTP_4XX.value),
        )


def _write_fixture(
    tmp_path: Path, *, tickers: tuple[str, ...] = ("AAA",)
) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir()
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
    _write_jsonl(input_root / "events.jsonl", _existing_events(tickers))
    _write_jsonl(input_root / "features.jsonl", _existing_features(tickers))
    _write_jsonl(input_root / "targets.jsonl", _existing_targets(tickers))
    _write_jsonl(input_root / "clusters.jsonl", _existing_clusters(tickers))

    registry_path = tmp_path / "source-registry.jsonl"
    _write_jsonl(registry_path, [_registry_row(ticker) for ticker in tickers])

    universe_path = tmp_path / "instrument-mapping.json"
    _write_json(
        universe_path,
        {"instruments": [_instrument(ticker) for ticker in tickers]},
    )
    return input_root, registry_path, universe_path


def _existing_events(tickers: tuple[str, ...]) -> list[dict[str, object]]:
    return [_event_row(f"{ticker.lower()}-existing", ticker) for ticker in tickers]


def _existing_features(tickers: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "event_id": f"{ticker.lower()}-existing",
            "feature_cutoff": "2026-07-01T06:59:00+00:00",
            "event_features": {"primary_event_type": "OTHER"},
            "market_features": {"pre_return_5m": "0.001"},
        }
        for ticker in tickers
    ]


def _existing_targets(tickers: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"event_id": f"{ticker.lower()}-existing", "horizons": {}} for ticker in tickers]


def _existing_clusters(tickers: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {"event_id": f"{ticker.lower()}-existing", "event_cluster_id": f"cluster-{ticker}"}
        for ticker in tickers
    ]


def _event_row(event_id: str, ticker: str) -> dict[str, object]:
    link = f"https://{ticker.lower()}.example/existing"
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": f"{ticker}_EXISTING",
            "source_item_id": link,
            "canonical_url": link,
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": "2026-07-01T07:00:00+00:00",
            "publication_timestamp_raw": "Wed, 01 Jul 2026 10:00:00 +0300",
            "publication_date": "2026-07-01",
            "publication_time": "07:00:00",
            "publication_timezone": "UTC",
            "timestamp_source_field": "official exact fixture",
            "timestamp_quality": "EXACT",
            "future_holdout": False,
            "session_state": "DURING_MAIN_SESSION",
            "title_hash": event_id,
        },
        "event_features": {"primary_event_type": "OTHER", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": {"pre_return_5m": "0.001"},
        "target_availability": {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": True,
            "status": "REACTION_READY",
            "missing_reason": None,
        },
        "quality": {
            "feature_cutoff": "2026-07-01T06:59:00+00:00",
            "reaction_starts_after_or_at_publication": True,
            "security_benchmark_same_window": True,
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
        },
    }


def _registry_row(ticker: str) -> dict[str, object]:
    domain = f"{ticker.lower()}.example"
    seed = "/rss" if ticker == "RSS" else "/feed" if ticker == "ATOM" else "/"
    return {
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": f"uid-{ticker}",
        "official_domain": domain,
        "source_url": f"https://{domain}{seed}",
        "source_family": f"{ticker}_EXISTING",
        "timestamp_capability": "UNKNOWN",
        "archive": False,
        "source_found": True,
        "provenance": "synthetic official registry row",
    }


def _instrument(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": f"{ticker} Issuer",
        "class_code": "TQBR",
        "currency": "rub",
        "instrument_uid": f"uid-{ticker}",
    }


def _ok_robots(url: str) -> HttpResult:
    return _response(url, "text/plain", b"User-agent: *\nAllow: /\n")


def _response(url: str, content_type: str, body: bytes) -> HttpResult:
    return HttpResult(url, url, 200, content_type, body, 0, None)


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
