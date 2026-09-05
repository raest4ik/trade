from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    evaluate_target_eligibility,
)
from src.moex_issuer_controlled_channel_discovery_v6.application import (
    ChannelCandidate,
    OwnershipProof,
    accepted_source_payload,
    discover_issuer_controlled_channels,
    probe_platform_channel,
    run_moex_issuer_controlled_channel_discovery_v6,
)
from src.moex_target_source_discovery_v5.application import default_v5_source_configs

BASE_MAIN_SHA = "a" * 40
GIT_SHA = "b" * 40
NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


def test_unofficial_telegram_channel_rejected() -> None:
    proof = OwnershipProof(
        ticker="ALRS",
        legal_issuer="АЛРОСА",
        platform="telegram",
        channel_id="alrosa_official",
        channel_url="https://t.me/alrosa_official",
        official_url="https://www.alrosa.ru/press-center/",
        official_domain="www.alrosa.ru",
        proof_level="NONE",
        proof_ready=False,
        blocker="ISSUER_CONTROL_UNPROVEN",
    )

    probe = probe_platform_channel(proof, client=_Client({}))

    assert probe.source_ready is False
    assert probe.blocker == "ISSUER_CONTROL_UNPROVEN"


def test_official_site_outbound_link_proves_control() -> None:
    candidate = _candidate("ALRS")
    client = _Client(
        {
            candidate.official_url: _html(
                candidate.official_url, "<a href='https://t.me/alrosa_official'>"
            )
        }
    )

    proofs, rejected = discover_issuer_controlled_channels(candidate, client=client)

    assert rejected == []
    assert proofs[0].proof_ready is True
    assert proofs[0].proof_level == "LEVEL_A_OFFICIAL_SITE_OUTBOUND_LINK"


def test_matching_name_without_official_link_rejected() -> None:
    candidate = _candidate("ALRS")
    client = _Client(
        {candidate.official_url: _html(candidate.official_url, "Telegram: ALROSA official")}
    )

    proofs, rejected = discover_issuer_controlled_channels(candidate, client=client)

    assert proofs == []
    assert rejected[0]["blocker"] == "NO_OFFICIAL_CHANNEL_REFERENCE"


def test_public_message_exact_utc_timestamp_accepted() -> None:
    proof = _proof("ALRS", "alrosa_official")
    client = _Client(
        {
            "https://t.me/s/alrosa_official": _telegram(
                "alrosa_official", "2026-09-05T09:00:00+00:00"
            )
        }
    )

    probe = probe_platform_channel(proof, client=client)

    assert probe.source_ready is True
    assert probe.timestamp_ready is True
    assert probe.identity_ready is True


def test_relative_timestamp_rejected() -> None:
    proof = _proof("ALRS", "alrosa_official")
    client = _Client(
        {
            "https://t.me/s/alrosa_official": _html(
                "https://t.me/s/alrosa_official", "<span>today 14:30</span>"
            )
        }
    )

    probe = probe_platform_channel(proof, client=client)

    assert probe.source_ready is False
    assert probe.blocker == "TIMESTAMP_UNVERIFIED"


def test_stable_message_id_required() -> None:
    proof = _proof("ALRS", "alrosa_official")
    page = '<time datetime="2026-09-05T09:00:00+00:00"></time>'
    client = _Client(
        {"https://t.me/s/alrosa_official": _html("https://t.me/s/alrosa_official", page)}
    )

    probe = probe_platform_channel(proof, client=client)

    assert probe.source_ready is False
    assert probe.blocker == "STABLE_IDENTITY_UNVERIFIED"


def test_auth_required_channel_rejected() -> None:
    proof = _proof("ALRS", "alrosa_official")
    client = _Client(
        {
            "https://t.me/s/alrosa_official": FetchResult(
                "https://t.me/s/alrosa_official", None, None, None, b"", 0, (), "AUTH_REQUIRED"
            )
        }
    )

    probe = probe_platform_channel(proof, client=client)

    assert probe.source_ready is False
    assert probe.blocker == "AUTH_REQUIRED"


def test_private_channel_rejected() -> None:
    proof = _proof("ALRS", "alrosa_official")
    private_page = '<div class="tgme_channel_join_telegram">Join Telegram to view</div>'
    client = _Client(
        {"https://t.me/s/alrosa_official": _html("https://t.me/s/alrosa_official", private_page)}
    )

    probe = probe_platform_channel(proof, client=client)

    assert probe.source_ready is False
    assert probe.blocker == "PRIVATE_CHANNEL"


def test_foreign_non_target_issuer_excluded_before_probe(tmp_path: Path) -> None:
    manifest = _run(
        tmp_path,
        tickers=("ALRS",),
        v5_tickers=("MSFT",),
        telegram_timestamps={"ALRS": "2026-09-05T09:00:00+00:00"},
    )

    assert manifest["CANONICAL_ISSUERS_INSPECTED"] == 0
    assert manifest["SOURCE_READY_CHANNELS"] == 0


def test_historical_post_cannot_enter_live_shadow_epoch(tmp_path: Path) -> None:
    manifest = _run(
        tmp_path,
        tickers=("ALRS",),
        telegram_timestamps={"ALRS": "2026-08-10T09:00:00+00:00"},
    )
    accepted = json.loads((tmp_path / "v6" / "accepted-sources.json").read_text(encoding="utf-8"))

    assert manifest["SOURCE_READY_CHANNELS"] == 1
    assert manifest["LIVE_SHADOW_POSTS_ELIGIBLE"] == 0
    assert accepted["sources"][0]["live_shadow_candidates"] == []


def test_qualifying_post_after_live_epoch_start_may_enter_shadow(tmp_path: Path) -> None:
    manifest = _run(
        tmp_path,
        tickers=("ALRS",),
        telegram_timestamps={"ALRS": "2026-08-11T09:00:00+00:00"},
    )

    assert manifest["SOURCE_READY_CHANNELS"] == 1
    assert manifest["LIVE_SHADOW_POSTS_ELIGIBLE"] == 1


def test_source_ready_but_feature_incompatible_does_not_count() -> None:
    probe = probe_platform_channel(
        _proof("ALRS", "alrosa_official"),
        client=_Client(
            {
                "https://t.me/s/alrosa_official": _telegram(
                    "alrosa_official", "2026-09-05T09:00:00+00:00"
                )
            }
        ),
    )
    eligibility = evaluate_target_eligibility(
        [accepted_source_payload(probe)],
        instrument_mapping_rows=[_mapping("ALRS")],
    )

    assert probe.source_ready is True
    assert eligibility[0].feature_pipeline_compatible is False
    assert eligibility[0].counts_toward_ml_diversity is False


def test_two_and_three_eligible_issuers_drive_diversity(tmp_path: Path) -> None:
    two = _run(
        tmp_path / "two",
        tickers=("ALRS", "MTSS"),
        telegram_timestamps={
            "ALRS": "2026-09-05T09:00:00+00:00",
            "MTSS": "2026-09-05T09:01:00+00:00",
        },
    )
    three = _run(
        tmp_path / "three",
        tickers=("ALRS", "MTSS", "PHOR"),
        telegram_timestamps={
            "ALRS": "2026-09-05T09:00:00+00:00",
            "MTSS": "2026-09-05T09:01:00+00:00",
            "PHOR": "2026-09-05T09:02:00+00:00",
        },
    )

    assert two["TARGET_DIVERSITY"] == "NO"
    assert two["FINAL_DIVERSITY_STATUS"] == "TWO_NEW_TARGET_ELIGIBLE_ISSUERS"
    assert three["TARGET_DIVERSITY"] == "YES"
    assert three["FINAL_DIVERSITY_STATUS"] == "THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS"


def _run(
    tmp_path: Path,
    *,
    tickers: tuple[str, ...],
    telegram_timestamps: dict[str, str],
    v5_tickers: tuple[str, ...] | None = None,
) -> dict[str, object]:
    configs = default_v5_source_configs()
    responses: dict[str, FetchResult] = {}
    for ticker in tickers:
        config = configs[ticker]
        channel = f"{ticker.lower()}_official"
        responses[config.url] = _html(config.url, f"<a href='https://t.me/{channel}'>channel</a>")
        responses[f"https://t.me/s/{channel}"] = _telegram(channel, telegram_timestamps[ticker])
    return run_moex_issuer_controlled_channel_discovery_v6(
        output_root=tmp_path / "v6",
        base_main_sha=BASE_MAIN_SHA,
        git_sha=GIT_SHA,
        operation_root=_live_operation_root(tmp_path / "operation"),
        instrument_mapping_path=_instrument_mapping(
            tmp_path / "mapping.json",
            [*[_mapping(ticker) for ticker in tickers], _mapping("IMOEX", exchange="imoex_index")],
        ),
        v5_root=_v5_root(tmp_path / "v5", v5_tickers or tickers),
        client=_Client(responses),
        created_at=NOW,
    )


def _candidate(ticker: str) -> ChannelCandidate:
    config = default_v5_source_configs()[ticker]
    return ChannelCandidate(
        ticker=ticker,
        legal_issuer=config.legal_issuer,
        official_url=config.url,
        official_domain=config.official_domain,
        instrument={},
    )


def _proof(ticker: str, channel: str) -> OwnershipProof:
    candidate = _candidate(ticker)
    return OwnershipProof(
        ticker=ticker,
        legal_issuer=candidate.legal_issuer,
        platform="telegram",
        channel_id=channel,
        channel_url=f"https://t.me/{channel}",
        official_url=candidate.official_url,
        official_domain=candidate.official_domain,
        proof_level="LEVEL_A_OFFICIAL_SITE_OUTBOUND_LINK",
        proof_ready=True,
        blocker=None,
    )


def _telegram(channel: str, timestamp: str) -> FetchResult:
    message_id = "101"
    page = f"""
<div class="tgme_widget_message" data-post="{channel}/{message_id}">
  <a class="tgme_widget_message_date" href="https://t.me/{channel}/{message_id}">
    <time datetime="{timestamp}"></time>
  </a>
  <div class="tgme_widget_message_text">Issuer publication</div>
</div>
"""
    return _html(f"https://t.me/s/{channel}", page)


def _html(url: str, body: str) -> FetchResult:
    return FetchResult(url, url, 200, "text/html", body.encode(), 0, (), None)


def _mapping(
    ticker: str,
    *,
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
        "name": f"Issuer {ticker}",
    }


def _instrument_mapping(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"instruments": rows}), encoding="utf-8")
    return path


def _v5_root(path: Path, tickers: tuple[str, ...]) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"CANONICAL_TARGET_TICKERS_CONSIDERED": list(tickers)}),
        encoding="utf-8",
    )
    return path


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
