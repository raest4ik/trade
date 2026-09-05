from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_expansion_v2.application import SourceProbeConfig
from src.moex_target_source_discovery_v5.application import (
    build_candidate_universe,
    build_probe_plan,
    canonical_registry_from_mapping,
    run_moex_target_source_discovery_v5,
)

BASE_MAIN_SHA = "a" * 40
GIT_SHA = "b" * 40
NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


def test_only_canonical_moex_targets_enter_discovery(tmp_path: Path) -> None:
    client = _Client(
        {
            "https://issuer-a.example/feed": _rss("ABIO", "issuer-a.example"),
            "https://foreign.example/feed": _rss("MSFT", "foreign.example"),
        }
    )

    manifest = run_moex_target_source_discovery_v5(
        output_root=tmp_path / "v5",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "operation"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "mapping.json",
            [_mapping("ABIO"), _mapping("IMOEX", exchange="imoex_index")],
        ),
        source_configs=[
            _config("ABIO", "https://issuer-a.example/feed", "issuer-a.example"),
            _config("MSFT", "https://foreign.example/feed", "foreign.example"),
        ],
        client=client,
        created_at=NOW,
    )

    assert manifest["DISTINCT_ISSUERS_PROBED"] == 1
    assert client.calls == ["https://issuer-a.example/feed"]


def test_frozen_and_existing_live_issuers_are_excluded() -> None:
    mapping = [
        _mapping("ROSN"),
        _mapping("YDEX"),
        _mapping("ABIO"),
        _mapping("IMOEX", exchange="imoex_index"),
    ]
    registry = canonical_registry_from_mapping(mapping)
    candidates, excluded = build_candidate_universe(
        mapping_rows=mapping,
        canonical_registry=registry,
        previous_v4_rejections={},
    )

    assert [candidate.ticker for candidate in candidates] == ["ABIO"]
    excluded_by_ticker = {row["source_ticker"]: row["skip_reason"] for row in excluded}
    assert excluded_by_ticker["ROSN"] == "FROZEN_HISTORICAL_ISSUER"
    assert excluded_by_ticker["YDEX"] == "FROZEN_HISTORICAL_ISSUER"


def test_share_class_collapse_before_network() -> None:
    mapping = [
        _mapping("SNGS", name="Сургутнефтегаз"),
        _mapping("SNGSP", name="Сургутнефтегаз - привилегированные акции"),
        _mapping("IMOEX", exchange="imoex_index"),
    ]
    registry = canonical_registry_from_mapping(mapping)
    candidates, excluded = build_candidate_universe(
        mapping_rows=mapping,
        canonical_registry=registry,
        previous_v4_rejections={},
    )

    assert [candidate.ticker for candidate in candidates] == ["SNGS"]
    assert any(
        row["source_ticker"] == "SNGSP"
        and row["skip_reason"] == "DUPLICATE_LEGAL_ISSUER_SHARE_CLASS_COLLAPSE"
        for row in excluded
    )


def test_previous_v4_source_cannot_be_reprobed_without_new_hypothesis() -> None:
    mapping = [_mapping("SBER"), _mapping("IMOEX", exchange="imoex_index")]
    registry = canonical_registry_from_mapping(mapping)
    candidates, _excluded = build_candidate_universe(
        mapping_rows=mapping,
        canonical_registry=registry,
        previous_v4_rejections={},
    )
    config = _config(
        "SBER",
        "https://www.sberbank.ru/ru/press_center/all",
        "www.sberbank.ru",
        mechanism="same_mechanism",
        hypothesis="",
    )

    probes, rows, deferred = build_probe_plan(
        candidates,
        configs_by_ticker={"SBER": config},
        previous_v4_rejections={
            "SBER": {
                "url": "https://www.sberbank.ru/ru/press_center/all",
                "new_hypothesis": "same_mechanism",
                "blocker": "TIMEOUT",
            }
        },
    )

    assert probes == []
    assert rows[0]["current_status"] == "RECHECK_DEFERRED_NO_NEW_HYPOTHESIS"
    assert deferred[0]["blocker"] == "RECHECK_DEFERRED_NO_NEW_HYPOTHESIS"


def test_alternate_mechanism_with_new_hypothesis_is_allowed() -> None:
    mapping = [_mapping("SBER"), _mapping("IMOEX", exchange="imoex_index")]
    registry = canonical_registry_from_mapping(mapping)
    candidates, _excluded = build_candidate_universe(
        mapping_rows=mapping,
        canonical_registry=registry,
        previous_v4_rejections={},
    )
    config = _config(
        "SBER",
        "https://www.sberbank.com/ru/investor-relations/news",
        "www.sberbank.com",
        mechanism="alternate_official_investor_feed",
        hypothesis="alternate official investor-relations endpoint",
    )

    probes, rows, deferred = build_probe_plan(
        candidates,
        configs_by_ticker={"SBER": config},
        previous_v4_rejections={
            "SBER": {
                "url": "https://www.sberbank.ru/ru/press_center/all",
                "new_hypothesis": "official_ru_press_center_jsonld_probe",
                "blocker": "TIMEOUT",
            }
        },
    )

    assert [probe.ticker for probe in probes] == ["SBER"]
    assert rows[0]["network_probe_allowed"] is True
    assert deferred == []


def test_foreign_source_excluded_before_probe(tmp_path: Path) -> None:
    client = _Client({"https://news.microsoft.com/feed/": _rss("MSFT", "news.microsoft.com")})

    manifest = run_moex_target_source_discovery_v5(
        output_root=tmp_path / "v5",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "operation"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "mapping.json", [_mapping("IMOEX", exchange="imoex_index")]
        ),
        source_configs=[_config("MSFT", "https://news.microsoft.com/feed/", "news.microsoft.com")],
        client=client,
        created_at=NOW,
    )

    assert manifest["DISTINCT_ISSUERS_PROBED"] == 0
    assert client.calls == []


def test_timestamp_without_timezone_rejected(tmp_path: Path) -> None:
    manifest = _run_with_responses(
        tmp_path,
        {"AAA": _rss("AAA", "issuer-a.example", timestamp="2026-09-05T12:00:00")},
        tickers=("AAA",),
    )

    assert manifest["FREE_OFFICIAL_SOURCES_READY"] == 0
    assert manifest["BLOCKERS_BY_CATEGORY"] == {"LIVE_CLOCK_WITHOUT_TIMEZONE": 1}


def test_exact_offset_accepted_and_stable_identity_required(tmp_path: Path) -> None:
    accepted = _run_with_responses(
        tmp_path / "accepted",
        {"AAA": _rss("AAA", "issuer-a.example", timestamp="2026-09-05T12:00:00+03:00")},
        tickers=("AAA",),
    )
    rejected = _run_with_responses(
        tmp_path / "identity",
        {"BBB": _rss("BBB", "issuer-b.example", guid="", link="")},
        tickers=("BBB",),
    )

    assert accepted["FREE_OFFICIAL_SOURCES_READY"] == 1
    assert accepted["NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS"] == 1
    assert rejected["FREE_OFFICIAL_SOURCES_READY"] == 0
    assert rejected["BLOCKERS_BY_CATEGORY"] == {"STABLE_IDENTITY_REQUIRED": 1}


def test_source_ready_without_target_eligibility_does_not_count(tmp_path: Path) -> None:
    manifest = run_moex_target_source_discovery_v5(
        output_root=tmp_path / "v5",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "operation"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "mapping.json",
            [_mapping("AAA", exchange="nasdaq"), _mapping("IMOEX", exchange="imoex_index")],
        ),
        source_configs=[_config("AAA", "https://issuer-a.example/feed", "issuer-a.example")],
        client=_Client({"https://issuer-a.example/feed": _rss("AAA", "issuer-a.example")}),
        created_at=NOW,
    )

    assert manifest["DISTINCT_ISSUERS_PROBED"] == 0
    assert manifest["NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS"] == 0


def test_target_eligibility_without_source_readiness_does_not_count(tmp_path: Path) -> None:
    manifest = _run_with_responses(
        tmp_path,
        {"AAA": _technical_failure("https://issuer-a.example/feed")},
        tickers=("AAA",),
    )

    assert manifest["CANONICAL_TARGET_ISSUERS_CONSIDERED"] == 1
    assert manifest["FREE_OFFICIAL_SOURCES_READY"] == 0
    assert manifest["NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS"] == 0


def test_two_and_three_distinct_eligible_issuers_drive_diversity(tmp_path: Path) -> None:
    two = _run_with_responses(
        tmp_path / "two",
        {
            "AAA": _rss("AAA", "issuer-a.example"),
            "BBB": _rss("BBB", "issuer-b.example"),
        },
        tickers=("AAA", "BBB"),
    )
    three = _run_with_responses(
        tmp_path / "three",
        {
            "AAA": _rss("AAA", "issuer-a.example"),
            "BBB": _rss("BBB", "issuer-b.example"),
            "CCC": _rss("CCC", "issuer-c.example"),
        },
        tickers=("AAA", "BBB", "CCC"),
    )

    assert two["TARGET_DIVERSITY"] == "NO"
    assert two["FINAL_DIVERSITY_STATUS"] == "TWO_NEW_TARGET_ELIGIBLE_ISSUERS"
    assert three["TARGET_DIVERSITY"] == "YES"
    assert three["FINAL_DIVERSITY_STATUS"] == "THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS"


def test_one_candidate_failure_does_not_abort_rest(tmp_path: Path) -> None:
    manifest = _run_with_responses(
        tmp_path,
        {
            "AAA": _technical_failure("https://issuer-a.example/feed"),
            "BBB": _rss("BBB", "issuer-b.example"),
        },
        tickers=("AAA", "BBB"),
    )

    assert manifest["DISTINCT_ISSUERS_PROBED"] == 2
    assert manifest["FREE_OFFICIAL_SOURCES_READY"] == 1
    assert manifest["BLOCKERS_BY_CATEGORY"] == {"TIMEOUT": 1}


def _run_with_responses(
    tmp_path: Path, responses_by_ticker: dict[str, FetchResult], *, tickers: tuple[str, ...]
) -> dict[str, object]:
    urls = {ticker: f"https://issuer-{ticker.lower()}.example/feed" for ticker in tickers}
    responses = {urls[ticker]: response for ticker, response in responses_by_ticker.items()}
    return run_moex_target_source_discovery_v5(
        output_root=tmp_path / "v5",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "operation"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "mapping.json",
            [*[_mapping(ticker) for ticker in tickers], _mapping("IMOEX", exchange="imoex_index")],
        ),
        source_configs=[
            _config(ticker, urls[ticker], f"issuer-{ticker.lower()}.example") for ticker in tickers
        ],
        client=_Client(responses),
        created_at=NOW,
    )


def _mapping(
    ticker: str,
    *,
    name: str | None = None,
    exchange: str = "moex_mrng_evng_e_wknd_dlr",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "figi": f"FIGI-{ticker}",
        "instrument_uid": f"UID-{ticker}",
        "class_code": "TQBR",
        "exchange": exchange,
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
        "first_1day_candle_date": "2020-01-01",
        "name": name or f"Issuer {ticker}",
    }


def _instrument_mapping(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"instruments": rows}), encoding="utf-8")
    return path


def _config(
    ticker: str,
    url: str,
    domain: str,
    *,
    mechanism: str = "official_issuer_rss",
    hypothesis: str = "new official first-party endpoint",
) -> SourceProbeConfig:
    return SourceProbeConfig(
        source_id=f"{ticker}_V5_TEST",
        ticker=ticker,
        legal_issuer=f"Issuer {ticker}",
        official_domain=domain,
        url=url,
        mechanism=mechanism,
        parser="rss-item-pubdate-explicit-offset-v1",
        timestamp_field="rss.channel.item.pubDate",
        identity_field="rss.channel.item.guid || rss.channel.item.link",
        content_fields=("rss.channel.item.title", "rss.channel.item.description"),
        new_hypothesis=hypothesis,
    )


def _rss(
    ticker: str,
    domain: str,
    *,
    timestamp: str = "Tue, 25 Aug 2026 11:44:16 +0300",
    guid: str | None = None,
    link: str | None = None,
) -> FetchResult:
    guid_value = ticker if guid is None else guid
    link_value = f"https://{domain}/news/{ticker.lower()}" if link is None else link
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>{ticker} headline</title>
      <description>{ticker} description</description>
      <link>{link_value}</link>
      <guid>{guid_value}</guid>
      <pubDate>{timestamp}</pubDate>
    </item>
  </channel>
</rss>
""".encode()
    return FetchResult(
        request_url=f"https://{domain}/feed",
        final_url=f"https://{domain}/feed",
        status=200,
        content_type="application/rss+xml",
        body=body,
        redirects=0,
        redirect_chain=(),
        blocker=None,
    )


def _technical_failure(url: str) -> FetchResult:
    return FetchResult(url, url, None, None, b"", 0, (), "TIMEOUT")


def _live_operation_root(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "operation-status.json").write_text(
        json.dumps(
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "source_results": [],
                "duplicates": 1,
                "timestamp_rejections": 0,
                "sealed_violations": 0,
            }
        ),
        encoding="utf-8",
    )
    (path / "collector-state.json").write_text(
        json.dumps({"sources": {"ROSN": {"last_successful_poll_at": NOW.isoformat()}}}),
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
    return path


class _Client:
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        return self.responses[url]
