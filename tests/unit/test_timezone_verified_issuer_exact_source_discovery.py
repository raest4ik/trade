from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.cli.discover_timezone_verified_issuer_exact_sources import build_parser
from src.exact_event_live_official_collection.http_client import FetchResult
from src.timezone_verified_issuer_exact_source_discovery import application as app
from src.timezone_verified_issuer_exact_source_discovery.domain import (
    ARTIFACT_VERSION,
    MAX_DOMAINS_TO_AUDIT,
    CandidateSource,
    FinalDecision,
    SourceStatus,
    artifact_sha,
    safety_flags,
)


def test_cli_defaults_to_timezone_verified_discovery_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.readiness_dir == "artifacts/exact-dataset-readiness-audit-v1"
    assert args.issuer_diversity_dir == "artifacts/issuer-exact-historical-diversity-expansion-v1"
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"
    assert args.base_main_sha == "8" * 40


def test_safety_flags_forbid_market_model_test_backtest_and_trading() -> None:
    flags = safety_flags()

    assert flags["RESEARCH_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["TINVEST_REQUESTS"] == 0
    assert flags["MARKET_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_REACTIONS_COMPUTED"] == 0
    assert flags["FUTURE_TARGETS_COMPUTED"] == 0
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["FEATURE_DEFINITION_CHANGED"] is False
    assert flags["REACTION_METHODOLOGY_CHANGED"] is False
    assert flags["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["BACKTEST_PERFORMED"] is False
    assert flags["REAL_TRADING_ALLOWED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False


def test_rfc_pubdate_iso_datepublished_and_z_timestamp_qualify(tmp_path: Path) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = (
        _candidate("AAA", "aaa.example", "https://aaa.example/rss.xml", "RSS"),
        _candidate("BBB", "bbb.example", "https://bbb.example/news/", "HTML"),
        _candidate("CCC", "ccc.example", "https://ccc.example/feed.xml", "RSS"),
    )
    client = _FixtureHttpClient(
        {
            "https://aaa.example/rss.xml": _rss(
                "AAA",
                [
                    "Thu, 30 Apr 2026 19:45:00 +0300",
                    "Wed, 29 Apr 2026 09:15:00 +0300",
                    "Tue, 28 Apr 2026 08:05:00 +0300",
                ],
            ),
            "https://bbb.example/news/": _listing(
                "https://bbb.example/news/1",
                "https://bbb.example/news/2",
                "https://bbb.example/news/3",
            ),
            "https://bbb.example/news/1": _html_date_published("2026-04-30T19:45:00+03:00"),
            "https://bbb.example/news/2": _html_date_published("2026-04-29T10:15:00+03:00"),
            "https://bbb.example/news/3": _html_date_published("2026-04-28T08:05:00+03:00"),
            "https://ccc.example/feed.xml": _rss(
                "CCC",
                [
                    "2026-04-30T16:45:00Z",
                    "2026-04-29T07:15:00Z",
                    "2026-04-28T05:05:00Z",
                ],
                field="datePublished",
            ),
        }
    )

    manifest = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )

    assert manifest["STRICT_EXACT_HISTORICAL_READY"] == 3
    assert manifest["VERIFIED_HISTORICAL_ITEMS"] == 9
    assert manifest["NEW_STRICT_EXACT_ISSUER_TICKERS"] == ["AAA", "BBB", "CCC"]
    assert manifest["FINAL_DECISION"] == FinalDecision.NEW_HISTORICAL_STRICT_EXACT_SOURCES_FOUND
    evidence = _read_jsonl(tmp_path / "output" / "historical-item-evidence.jsonl")
    assert {row["timezone_evidence_type"] for row in evidence} == {
        "RFC822_OFFSET",
        "ISO_OFFSET",
        "UTC_Z",
    }
    assert manifest["ARTIFACT_SHA"] == artifact_sha(manifest)


def test_bad_timestamp_sources_fail_closed_without_timezone(tmp_path: Path) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = (
        _candidate("BARE", "bare.example", "https://bare.example/rss.xml", "RSS"),
        _candidate("DATE", "date.example", "https://date.example/rss.xml", "RSS"),
        _candidate("MOD", "modified.example", "https://modified.example/news/", "HTML"),
        _candidate("ANL", "analytics.example", "https://analytics.example/news/", "HTML"),
    )
    client = _FixtureHttpClient(
        {
            "https://bare.example/rss.xml": _rss(
                "BARE",
                ["2026-04-30T19:45:00", "2026-04-29T10:00:00", "2026-04-28T09:00:00"],
                field="datePublished",
            ),
            "https://date.example/rss.xml": _rss(
                "DATE", ["2026-04-30", "2026-04-29", "2026-04-28"], field="datePublished"
            ),
            "https://modified.example/news/": (
                '<html><head><meta property="article:modified_time" '
                'content="2026-04-30T19:45:00+03:00"></head>'
                "<body><h1>Modified only</h1>30 April 2026</body></html>"
            ),
            "https://analytics.example/news/": (
                "<html><body><h1>Analytics only</h1>"
                "30.04.2026 19:45 issuer publication text"
                '<script>window.analyticsAt="2026-04-30T19:45:00+03:00"</script>'
                "</body></html>"
            ),
        }
    )

    manifest = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )

    audit = _audit_by_ticker(tmp_path / "output")
    assert audit["BARE"]["status"] == SourceStatus.CLOCK_TIME_WITHOUT_TIMEZONE
    assert audit["DATE"]["status"] == SourceStatus.DATE_ONLY
    assert audit["MOD"]["timezone_evidence_available"] is False
    assert audit["ANL"]["status"] == SourceStatus.CLOCK_TIME_WITHOUT_TIMEZONE
    assert manifest["STRICT_EXACT_HISTORICAL_READY"] == 0
    assert manifest["CLOCK_TIME_WITHOUT_TIMEZONE"] == 2
    assert manifest["DATE_ONLY"] == 2
    timezone = _read_jsonl(tmp_path / "output" / "timezone-evidence.jsonl")
    analytics = next(row for row in timezone if row["ticker"] == "ANL")
    assert analytics["unrelated_timezone_timestamp_seen"] is True
    assert analytics["accepted_as_publication_timezone"] is False


def test_publication_specific_metadata_is_required(tmp_path: Path) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = (
        _candidate("GOOD", "good.example", "https://good.example/news/", "HTML"),
        _candidate("BAD", "bad.example", "https://bad.example/news/", "HTML"),
    )
    client = _FixtureHttpClient(
        {
            "https://good.example/news/": (
                '<html><head><meta property="article:published_time" '
                'content="2026-04-30T19:45:00+03:00"></head>'
                "<body><h1>Good</h1>publication body</body></html>"
            ),
            "https://bad.example/news/": (
                '<html><head><meta property="article:modified_time" '
                'content="2026-04-30T19:45:00+03:00"></head>'
                "<body><h1>Bad</h1>publication body</body></html>"
            ),
        }
    )

    manifest = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )

    audit = _audit_by_ticker(tmp_path / "output")
    assert audit["GOOD"]["status"] == SourceStatus.STRICT_EXACT_LIVE_ONLY
    assert audit["GOOD"]["timezone_evidence_field"] == "article:published_time"
    assert audit["BAD"]["timezone_evidence_available"] is False
    assert manifest["STRICT_EXACT_HISTORICAL_READY"] == 0


def test_policy_rejects_third_party_and_exchange_origin(tmp_path: Path) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = (
        _candidate("THRD", "issuer.example", "https://third-party.example/news", "HTML"),
        CandidateSource(
            ticker="MOEXRISK",
            issuer="Moscow Exchange risk feed",
            official_domain="www.moex.com",
            source_url="https://www.moex.com/export/news.aspx",
            source_mechanism="RSS",
            event_origin="EXCHANGE_ORIGINATED",
        ),
    )

    manifest = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=_FixtureHttpClient({}),
        candidates=candidates,
    )

    audit = _audit_by_ticker(tmp_path / "output")
    assert audit["THRD"]["status"] == SourceStatus.POLICY_BLOCKED
    assert audit["THRD"]["primary_blocker"] == "THIRD_PARTY_CANONICAL_URL"
    assert audit["MOEXRISK"]["status"] == SourceStatus.POLICY_BLOCKED
    assert audit["MOEXRISK"]["primary_blocker"] == "EVENT_ORIGIN_NOT_ISSUER"
    assert manifest["POLICY_BLOCKED"] == 2


def test_no_tinvest_or_outcome_fields_used_and_cap_ordering_is_deterministic(
    tmp_path: Path,
) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = tuple(
        _candidate(
            f"T{index:02d}", f"d{index:02d}.example", f"https://d{index:02d}.example/rss.xml", "RSS"
        )
        for index in range(20)
    )
    client = _FixtureHttpClient({})

    left = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )
    right = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="a" * 40,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )

    assert left["DOMAINS_AUDITED"] == MAX_DOMAINS_TO_AUDIT
    assert left["SOURCES_AUDITED"] == MAX_DOMAINS_TO_AUDIT
    assert left["CANDIDATE_SOURCES_SHA"] == right["CANDIDATE_SOURCES_SHA"]
    assert left["AUDITED_SOURCES_SHA"] == right["AUDITED_SOURCES_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert left["DETERMINISTIC_REPLAY"] == "PASS"
    assert left["TINVEST_REQUESTS"] == 0
    assert left["MARKET_PRICE_LOOKUPS"] == 0
    assert left["SOURCE_SELECTION_USED_MARKET_OUTCOMES"] is False
    assert left["SOURCE_SELECTION_USED_EVENT_ANALYZER"] is False
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "TInvestReadOnlyClient" not in source
    assert "fetch_minute_candles" not in source
    assert "target_return" not in source
    assert ".fit(" not in source


def test_replay_artifact_is_deterministic_with_fixture_network(tmp_path: Path) -> None:
    readiness, issuer = _write_inputs(tmp_path)
    candidates = (_candidate("AAA", "aaa.example", "https://aaa.example/rss.xml", "RSS"),)
    client = _FixtureHttpClient(
        {
            "https://aaa.example/rss.xml": _rss(
                "AAA",
                [
                    "Thu, 30 Apr 2026 19:45:00 +0300",
                    "Wed, 29 Apr 2026 09:15:00 +0300",
                    "Tue, 28 Apr 2026 08:05:00 +0300",
                ],
            )
        }
    )

    left = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )
    right = app.run_timezone_verified_issuer_exact_source_discovery(
        readiness_root=readiness,
        issuer_diversity_root=issuer,
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="a" * 40,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        http_client=client,
        candidates=candidates,
    )

    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert _read_jsonl(tmp_path / "left" / "audited-sources.jsonl") == _read_jsonl(
        tmp_path / "right" / "audited-sources.jsonl"
    )


class _FixtureHttpClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        body = self.responses.get(url)
        if body is None:
            return FetchResult(url, url, 404, "text/html", b"", 0, (), "HTTP_FAILURE")
        content_type = "application/rss+xml" if body.lstrip().startswith("<?xml") else "text/html"
        return FetchResult(url, url, 200, content_type, body.encode("utf-8"), 0, (), None)


def _candidate(ticker: str, domain: str, url: str, mechanism: str) -> CandidateSource:
    return CandidateSource(
        ticker=ticker,
        issuer=f"{ticker} Issuer",
        official_domain=domain,
        source_url=url,
        source_mechanism=mechanism,
        source_family=f"{ticker}_OFFICIAL_TZ_SOURCE",
    )


def _rss(ticker: str, timestamps: list[str], *, field: str = "pubDate") -> str:
    items = "\n".join(
        f"""
        <item>
          <title>{ticker} item {index}</title>
          <link>https://{ticker.lower()}.example/news/{index}</link>
          <{field}>{timestamp}</{field}>
          <description>{ticker} official publication material {index}</description>
        </item>
        """
        for index, timestamp in enumerate(timestamps, start=1)
    )
    return f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>'


def _html_date_published(timestamp: str) -> str:
    return (
        '<html><head><script type="application/ld+json">'
        f'{{"@type":"NewsArticle","datePublished":"{timestamp}"}}'
        "</script></head><body><h1>Issuer release</h1>material body</body></html>"
    )


def _listing(*links: str) -> str:
    return (
        "<html><body>" + "".join(f'<a href="{link}">item</a>' for link in links) + "</body></html>"
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    readiness = tmp_path / "readiness"
    issuer = tmp_path / "issuer"
    _write_json(readiness / "manifest.json", {"ARTIFACT_SHA": "readiness-sha"})
    _write_jsonl(
        readiness / "source-family-summary.jsonl",
        [{"source_family": "EXISTING_SOURCE", "feature_ready": 1}],
    )
    _write_json(
        issuer / "manifest.json",
        {"ARTIFACT_SHA": "issuer-sha", "FINAL_DECISION": "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"},
    )
    return readiness, issuer


def _audit_by_ticker(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["ticker"]): row for row in _read_jsonl(root / "audited-sources.jsonl")}


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
