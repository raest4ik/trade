from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_expansion_v2.application import SourceProbeConfig
from src.moex_target_source_expansion_v4.application import (
    build_target_candidate_universe,
    build_v4_candidate_hypotheses,
    prove_source_isolation_application_path,
    run_moex_target_source_expansion_v4,
)

BASE_MAIN_SHA = "a" * 40
GIT_SHA = "b" * 40
NOW = datetime(2026, 9, 5, 9, tzinfo=UTC)
LKOH_V4_URL = "https://www.lukoil.com/PressCenter/Pressreleases?tags=38CXjmeic02Sgectxn85Pg%2C1%3B"


def test_target_eligibility_happens_before_source_probe(tmp_path: Path) -> None:
    client = _CountingClient({})

    manifest = run_moex_target_source_expansion_v4(
        output_root=tmp_path / "v4",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "live-operation"),
        instrument_mapping_path=_instrument_mapping(tmp_path / "mapping.json", tickers=("IMOEX",)),
        client=client,
        created_at=NOW,
    )

    assert manifest["CANDIDATES_ACTUALLY_PROBED"] == 0
    assert client.calls == []


def test_non_target_source_never_probed_for_diversity() -> None:
    candidates, skipped = build_target_candidate_universe(
        instrument_mapping_rows=[_mapping("IMOEX", exchange="imoex_index"), _mapping("MSFT")]
    )
    probe_configs, _rows = build_v4_candidate_hypotheses(candidates)

    assert "MSFT" not in {candidate.ticker for candidate in candidates}
    assert "MSFT" not in {config.ticker for config in probe_configs}
    assert all(row["source_ticker"] != "MSFT" for row in skipped)


def test_frozen_issuers_missing_mapping_and_unsupported_venue_are_skipped() -> None:
    candidates, skipped = build_target_candidate_universe(
        instrument_mapping_rows=[
            _mapping("ROSN"),
            _mapping("GAZP", exchange="nasdaq"),
            _mapping("SBER"),
            _mapping("IMOEX", exchange="imoex_index"),
        ]
    )

    skipped_by_ticker = {row["source_ticker"]: row["skip_reason"] for row in skipped}
    assert skipped_by_ticker["ROSN"] == "ALREADY_IN_FROZEN_HISTORICAL_TARGET_UNIVERSE"
    assert skipped_by_ticker["GAZP"] == "UNSUPPORTED_EXCHANGE_OR_TRADING_VENUE"
    assert skipped_by_ticker["LKOH"] == "CANONICAL_INSTRUMENT_MAPPING_MISSING"
    assert [candidate.ticker for candidate in candidates] == ["SBER"]


def test_share_classes_collapse_before_network_probe() -> None:
    candidates, skipped = build_target_candidate_universe(
        instrument_mapping_rows=[
            _mapping("SBER"),
            _mapping("SBERP"),
            _mapping("GAZP"),
            _mapping("IMOEX", exchange="imoex_index"),
        ]
    )

    assert [candidate.ticker for candidate in candidates] == ["SBER", "GAZP"]
    assert any(
        row["source_ticker"] == "SBERP"
        and row["skip_reason"] == "DUPLICATE_LEGAL_ISSUER_SHARE_CLASS_COLLAPSE"
        for row in skipped
    )


def test_feature_incompatible_candidate_does_not_count() -> None:
    candidates, skipped = build_target_candidate_universe(
        instrument_mapping_rows=[_mapping("SBER")]
    )

    assert candidates == []
    assert any(
        row["source_ticker"] == "SBER" and row["skip_reason"] == "APPROVED_BENCHMARK_PATH_MISSING"
        for row in skipped
    )


def test_two_and_three_ready_target_eligible_issuers_drive_diversity(tmp_path: Path) -> None:
    two_client = _CountingClient(
        {
            "https://www.gazprom.com/investors/disclosure/irreleases/": _rss("GAZP"),
            LKOH_V4_URL: _rss("LKOH"),
        }
    )
    two = run_moex_target_source_expansion_v4(
        output_root=tmp_path / "two",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "live-two"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "two-mapping.json", tickers=("GAZP", "LKOH", "IMOEX")
        ),
        client=two_client,
        created_at=NOW,
    )

    three_client = _CountingClient(
        {
            "https://www.gazprom.com/investors/disclosure/irreleases/": _rss("GAZP"),
            LKOH_V4_URL: _rss("LKOH"),
            "https://www.novatek.ru/en/press/releases/index.php": _rss("NVTK"),
        }
    )
    three = run_moex_target_source_expansion_v4(
        output_root=tmp_path / "three",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "live-three"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "three-mapping.json", tickers=("GAZP", "LKOH", "NVTK", "IMOEX")
        ),
        client=three_client,
        created_at=NOW,
    )

    assert two["TARGET_ELIGIBLE_DIVERSITY"] == "NO"
    assert two["FINAL_DIVERSITY_STATUS"] == "TWO_NEW_TARGET_ELIGIBLE_ISSUERS"
    assert three["TARGET_ELIGIBLE_DIVERSITY"] == "YES"
    assert three["FINAL_DIVERSITY_STATUS"] == "THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS"


def test_source_ready_but_target_ineligible_does_not_count(tmp_path: Path) -> None:
    config = SourceProbeConfig(
        source_id="MSFT_DISCOVERY_ONLY",
        ticker="MSFT",
        legal_issuer="Microsoft Corporation",
        official_domain="news.microsoft.com",
        url="https://news.microsoft.com/feed/",
        mechanism="official_issuer_rss",
        parser="rss-item-pubdate-explicit-offset-v1",
        timestamp_field="rss.channel.item.pubDate",
        identity_field="rss.channel.item.guid || rss.channel.item.link",
        content_fields=("rss.channel.item.title", "rss.channel.item.description"),
        new_hypothesis="Discovery-only source does not enter MOEX target diversity.",
    )
    client = _CountingClient(
        {"https://news.microsoft.com/feed/": _rss("MSFT", "news.microsoft.com")}
    )
    manifest = run_moex_target_source_expansion_v4(
        output_root=tmp_path / "v4",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "live-operation"),
        instrument_mapping_path=_instrument_mapping(tmp_path / "mapping.json", tickers=("IMOEX",)),
        client=client,
        network_check=False,
        created_at=NOW,
    )

    assert config.ticker == "MSFT"
    assert manifest["NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT"] == 0
    assert manifest["TARGET_ELIGIBLE_DIVERSITY"] == "NO"


def test_application_level_source_isolation_failure_continuation_and_recovery(
    tmp_path: Path,
) -> None:
    proof = prove_source_isolation_application_path(
        output_root=tmp_path / "proof",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        created_at=NOW,
    )

    assert proof["SOURCE_ISOLATION_APPLICATION_PROOF"] is True
    assert proof["failed_source"]["status"] == "SOURCE_FAILURE"
    assert proof["healthy_source_continued"]["status"] == "SUCCESS"
    assert proof["recovered_source"]["status"] == "SUCCESS"
    assert proof["state_b_persisted"] is True
    assert proof["burnin"]["OPERATIONAL_BURN_IN"] == "PASS"


def test_source_isolation_and_safety_zero_are_required_for_burnin_pass(tmp_path: Path) -> None:
    manifest = run_moex_target_source_expansion_v4(
        output_root=tmp_path / "v4",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "live-operation"),
        instrument_mapping_path=_instrument_mapping(tmp_path / "mapping.json", tickers=("IMOEX",)),
        client=_CountingClient({}),
        created_at=NOW,
    )

    assert manifest["SOURCE_ISOLATION_APPLICATION_PROOF"] is True
    assert manifest["OPERATIONAL_BURN_IN"] == "PASS"
    assert manifest["LIVE_OUTCOMES_READ"] == 0
    assert manifest["LIVE_TARGETS_COMPUTED"] == 0
    assert manifest["LIVE_POST_EVENT_PRICE_READS"] == 0
    assert manifest["BROKER_MUTATIONS"] == 0


def _mapping(
    ticker: str,
    *,
    exchange: str = "moex_mrng_evng_e_wknd_dlr",
    first_1day_candle_date: str = "2020-01-01",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "figi": f"FIGI-{ticker}",
        "instrument_uid": f"UID-{ticker}",
        "class_code": "TQBR",
        "exchange": exchange,
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
        "first_1day_candle_date": first_1day_candle_date,
        "name": ticker,
    }


def _instrument_mapping(path: Path, *, tickers: tuple[str, ...]) -> Path:
    rows = [
        _mapping(
            ticker, exchange="imoex_index" if ticker == "IMOEX" else "moex_mrng_evng_e_wknd_dlr"
        )
        for ticker in tickers
    ]
    path.write_text(json.dumps({"instruments": rows}), encoding="utf-8")
    return path


def _rss(ticker: str, domain: str = "issuer.test") -> FetchResult:
    body = f"""<!doctype html>
<html>
  <head>
    <script type="application/ld+json">
      {{
        "@type": "NewsArticle",
        "headline": "{ticker} headline",
        "description": "{ticker} description",
        "url": "https://{domain}/news/{ticker.lower()}",
        "datePublished": "2026-08-25T11:44:16+03:00"
      }}
    </script>
  </head>
</html>
""".encode()
    return FetchResult(
        request_url=f"https://{domain}/rss",
        final_url=f"https://{domain}/rss",
        status=200,
        content_type="application/rss+xml",
        body=body,
        redirects=0,
        redirect_chain=(),
        blocker=None,
    )


def _live_operation_root(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "operation-status.json").write_text(
        json.dumps(
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "source_results": [],
                "duplicates": 1,
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


class _CountingClient:
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        return self.responses[url]
