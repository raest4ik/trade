from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import (
    EXPECTED_RULES_FINGERPRINT,
    FROZEN_V1_DATASET_SHA,
    FROZEN_V1_FEATURE_SCHEMA_SHA,
    FROZEN_V1_PROVENANCE_SHA,
    FROZEN_V1_SOURCE_REGISTRY_SHA,
    build_source_registry,
    daily_target,
    leakage_pass,
    market_context,
)
from src.event_market_dataset.domain import (
    DATASET_VERSION,
    PREDICTIVE_UNIT,
    QWEN_PROMPT_SHA,
    QWEN_SCHEMA_SHA,
    REACTION_DAILY,
    SOURCE_REGISTRY_VERSION,
    AcquiredEvent,
    EventMarketRow,
    SourceRegistryStatus,
    deduplicate_events,
    readiness,
    require_unambiguous_ticker,
)
from src.event_market_dataset.sources import (
    PARSER_VERSION,
    ArchiveSourceConfig,
    acquire_archive,
    parse_official_archive,
)
from src.events.domain.v3 import rules_v3_fingerprint
from src.news.domain.enums import PublicationTimestampQuality
from src.shared.config.settings import (
    DEFAULT_AI_RANDOM_SEED,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_THINK,
)


def test_source_registry_is_deterministic_official_and_zero_cost(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "instruments": [
                    _instrument("YDEX", "Yandex", "uid-yandex"),
                    _instrument("NVTK", "NOVATEK", "uid-novatek"),
                    _instrument("ROSN", "Rosneft", "uid-rosneft"),
                    _instrument("LKOH", "LUKOIL", "uid-lukoil"),
                    _instrument("GMKN", "Nornickel", "uid-nornickel"),
                    _instrument("TATN", "Tatneft", "uid-tatneft"),
                    _instrument("ALRS", "ALROSA", "uid-alrosa"),
                    _instrument("PHOR", "PhosAgro", "uid-phosagro"),
                    _instrument("PLZL", "Polyus", "uid-polyus"),
                    _instrument("IRAO", "Inter RAO", "uid-interrao"),
                    _instrument("MGNT", "Magnit", "uid-magnit"),
                    _instrument("GAZP", "Gazprom", "uid-gazprom"),
                    _instrument("ZZZZ", "Unknown issuer", "uid-unknown"),
                    _instrument("IMOEX", "Benchmark", "uid-index"),
                ]
            }
        ),
        encoding="utf-8",
    )
    first = build_source_registry(mapping, checked_on=date(2026, 8, 13))
    second = build_source_registry(mapping, checked_on=date(2026, 8, 13))
    assert [item.payload() for item in first] == [item.payload() for item in second]
    ready = [item for item in first if item.status == SourceRegistryStatus.SOURCE_READY]
    assert {item.ticker for item in ready} == {
        "ALRS",
        "GMKN",
        "IRAO",
        "LKOH",
        "MGNT",
        "NVTK",
        "PHOR",
        "PLZL",
        "ROSN",
        "TATN",
        "YDEX",
    }
    assert all(
        item.official_source_url is not None and item.official_source_url.startswith("https://")
        for item in ready
    )
    assert all(not item.payment_required and not item.authentication_required for item in ready)
    unknown = next(item for item in first if item.ticker == "ZZZZ")
    assert unknown.official_source_url is None
    assert "no URL was guessed" in unknown.reason
    assert all(item.ticker != "IMOEX" for item in first)
    yandex = next(item for item in first if item.ticker == "YDEX")
    assert "2024-07-24" in str(yandex.historical_range)
    gazprom = next(item for item in first if item.ticker == "GAZP")
    assert gazprom.status == SourceRegistryStatus.BLOCKED_BY_ACCESS
    assert all(item.parser_version == PARSER_VERSION for item in ready)
    assert all(
        item.payload()["source_registry_version"] == SOURCE_REGISTRY_VERSION for item in first
    )


def test_v2_versions_and_research_readiness_are_explicit() -> None:
    assert DATASET_VERSION == "event-market-predictive-dataset-v2"
    assert SOURCE_REGISTRY_VERSION == "event-source-registry-v2"
    assert readiness(746, 3)["event_model_data_status"] == "NOT_READY"
    assert readiness(500, 10)["event_model_data_status"] == "READY_FOR_BASELINE_EXPERIMENT"
    assert readiness(500, 4)["event_diversity_status"] == "VERY_LOW_EVENT_DIVERSITY"
    assert FROZEN_V1_DATASET_SHA == (
        "21053ddf973f08837b719c28119ae20af8563b6caa88fe603cbef6a6e114e4f5"
    )
    assert FROZEN_V1_SOURCE_REGISTRY_SHA == (
        "3cd7a8a268be5c6146ce9793971940787de8cd0fb4ed493ca5d63f097658f13d"
    )
    assert FROZEN_V1_PROVENANCE_SHA == (
        "73e9c2cfe9a1709979a78a4b9f674f3b963ae79e288747e85a32c7560146b838"
    )
    assert FROZEN_V1_FEATURE_SCHEMA_SHA == (
        "4be00e812ba4e23a5245c0d132057bbb5d2e4fc1c6b50ea40a85ae476cfe34cc"
    )


@pytest.mark.parametrize(
    ("profile", "payload", "expected_title"),
    [
        (
            "TATNEFT",
            '<div class="material"><span class="material__date">9 августа 2024</span>'
            '<a href="/press-tsentr/press-relizi/more/9140/?lang=ru" '
            'class="material__title">Tatneft release</a></div>',
            "Tatneft release",
        ),
        (
            "PHOSAGRO",
            '<a href="/press/company/release/" class="press__card link">'
            '<h4>PhosAgro release</h4><div class="press__card-date">10 июля 2026</div></a>',
            "PhosAgro release",
        ),
        (
            "POLYUS",
            '<article class="news-item"><a href="/ru/media/press-releases/release/">'
            'Polyus release</a><time datetime="2026-07-27">27 июля</time></article>',
            "Polyus release",
        ),
        (
            "INTERRAO",
            '<div class="news-item"><div class="date">12 мая 2026</div>'
            '<div class="name"><a href="detail.php?ID=18945">Inter RAO release</a></div>',
            "Inter RAO release",
        ),
    ],
)
def test_configured_official_html_profiles_are_reusable(
    profile: str, payload: str, expected_title: str
) -> None:
    parsed = parse_official_archive(
        payload,
        base_url="https://issuer.example/archive/",
        profile=profile,
    )
    assert len(parsed) == 1
    assert parsed[0].title == expected_title
    assert parsed[0].publication_date <= date(2026, 12, 31)


def test_public_application_state_profiles_do_not_impute_time() -> None:
    nornickel = (
        '<script>App = {"items":[{"name":"Nornickel release","code":"release",'
        '"detailPageUrl":"/news/release/","activeFrom":1785218400}]};</script>'
    )
    magnit = (
        '<script>App = {"items":[{"date":1786088700,"name":"Magnit release",'
        '"link":"/ru/media/press-releases/release/"}]};</script>'
    )
    for profile, payload in (("NORNICKEL_APP", nornickel), ("MAGNIT_APP", magnit)):
        parsed = parse_official_archive(
            payload,
            base_url="https://issuer.example/",
            profile=profile,
        )
        assert len(parsed) == 1
        assert isinstance(parsed[0].publication_date, date)


def test_lukoil_and_alrosa_public_json_profiles() -> None:
    lukoil = (
        '<script type="application/json" class="pressreleases-data">'
        '{"Items":[{"FriendlyUrl":"release","Name":"LUKOIL release",'
        '"PublicationDate":"2026-07-21T00:00:00"}]}</script>'
    )
    alrosa = (
        '<ar-news :news="[{&quot;id&quot;:&quot;1&quot;,&quot;caption&quot;:'
        "&quot;ALROSA release&quot;,&quot;date&quot;:&quot;27 дек 2025&quot;,"
        '&quot;url&quot;:&quot;/press-center/news/2025/release/&quot;}]"></ar-news>'
    )
    assert parse_official_archive(lukoil, base_url="https://lukoil.ru/", profile="LUKOIL")[
        0
    ].publication_date == date(2026, 7, 21)
    assert parse_official_archive(alrosa, base_url="https://www.alrosa.ru/", profile="ALROSA")[
        0
    ].publication_date == date(2025, 12, 27)


def test_novatek_parser_canonicalizes_stable_release_id() -> None:
    from src.event_market_dataset.sources import parse_novatek_archive

    payload = (
        '<div class="date">24 July 2026</div>'
        '<a href="/en/press/releases/index.php?id_4=7822&from_4=1&amp;mode_5=pdrm">'
        "NOVATEK Announces Financial Results</a>"
    )
    assert parse_novatek_archive(payload, base_url="https://www.novatek.ru/en/press/releases/") == [
        (
            "https://www.novatek.ru/en/press/releases/index.php?id_4=7822",
            "https://www.novatek.ru/en/press/releases/index.php?id_4=7822",
            "NOVATEK Announces Financial Results",
            date(2026, 7, 24),
        )
    ]


@pytest.mark.asyncio
async def test_archive_acquisition_is_bounded_official_and_uses_no_auth() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            request=request,
            text=(
                '<a class="press-release-item__link" href="/press-releases/one">'
                '<span class="date">12 августа 2026</span>'
                '<span class="press-release-item__title">Release one</span></a>'
            ),
        )

    config = _archive_config()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await acquire_archive(
            config,
            date_from=date(2026, 1, 1),
            date_to=date(2026, 12, 31),
            limit=1,
            client=client,
        )
    assert len(events) == 1
    assert len(calls) == 1
    assert calls[0].url.host == "ir.yandex.ru"
    assert "authorization" not in calls[0].headers
    assert events[0].timestamp_quality == PublicationTimestampQuality.DATE_ONLY
    assert events[0].published_at is None


def test_deduplication_covers_existing_url_source_and_story() -> None:
    existing = _event("existing", "https://issuer.example/releases/1", "Same title")
    same_url = _event("new-url", "https://issuer.example/releases/1/", "Same title")
    same_source = _event("existing", "https://issuer.example/releases/2", "Same title")
    same_story = _event("story", "https://issuer.example/releases/3", "Same title")
    selected, dropped = deduplicate_events(
        [same_url, same_source, same_story], existing_events=[existing]
    )
    assert selected == []
    assert {str(item["reason"]) for item in dropped} == {
        "DUPLICATE_CANONICAL_URL",
        "DUPLICATE_SOURCE_RECORD",
        "SAME_EVENT_REPEATED",
    }


def test_changed_title_for_same_source_is_diagnosed_as_update() -> None:
    existing = _event("one", "https://issuer.example/one", "Original")
    updated = _event("one", "https://issuer.example/one", "Updated")
    selected, dropped = deduplicate_events([updated], existing_events=[existing])
    assert selected == []
    assert dropped[0]["reason"] == "UPDATED_PUBLICATION"


def test_timestamp_quality_does_not_impute_date_only_intraday() -> None:
    event = _event("one", "https://issuer.example/one", "Release")
    assert event.timestamp_quality == PublicationTimestampQuality.DATE_ONLY
    assert event.published_at is None
    with pytest.raises(ValueError, match="non-EXACT"):
        AcquiredEvent.create(
            source_code="OFFICIAL",
            source_item_id="bad",
            source_url="https://issuer.example/bad",
            ticker="YDEX",
            issuer_name="Yandex",
            instrument_uid="uid",
            figi="figi",
            title="Release",
            publication_date=date(2026, 8, 12),
            published_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
        )


def test_ticker_ambiguity_fails_closed() -> None:
    assert require_unambiguous_ticker(("YDEX",)) == "YDEX"
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        require_unambiguous_ticker(("YDEX", "YNDX"))
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        require_unambiguous_ticker(())


def test_date_only_market_context_is_strictly_before_publication_date() -> None:
    rows = {
        "YDEX": [
            (date(2026, 8, 10), datetime(2026, 8, 9, 23, 59, 59, tzinfo=UTC), {"x": 1.0}),
            (date(2026, 8, 11), datetime(2026, 8, 10, 23, 59, 59, tzinfo=UTC), {"x": 2.0}),
            (date(2026, 8, 12), datetime(2026, 8, 11, 23, 59, 59, tzinfo=UTC), {"x": 3.0}),
        ]
    }
    cutoff, values = market_context(rows, "YDEX", date(2026, 8, 11)) or (None, {})
    assert cutoff == datetime(2026, 8, 10, 23, 59, 59, tzinfo=UTC)
    assert values == {"x": 2.0}


def test_exact_and_date_only_cutoff_guards() -> None:
    exact = AcquiredEvent.create(
        source_code="OFFICIAL",
        source_item_id="exact",
        source_url="https://issuer.example/exact",
        ticker="YDEX",
        issuer_name="Yandex",
        instrument_uid="uid",
        figi="figi",
        title="Release",
        publication_date=date(2026, 8, 12),
        published_at=datetime(2026, 8, 12, 10, tzinfo=UTC),
        timestamp_quality=PublicationTimestampQuality.EXACT,
    )
    assert exact.published_at is not None
    with pytest.raises(ValueError, match="precede publication"):
        EventMarketRow(exact, "EXACT_INTRADAY", exact.published_at, {}, {}, {}).feature_payload()
    date_only = _event("date", "https://issuer.example/date", "Date release")
    with pytest.raises(ValueError, match="precede publication date"):
        EventMarketRow(
            date_only,
            REACTION_DAILY,
            datetime(2026, 8, 12, tzinfo=UTC),
            {},
            {},
            {},
        ).feature_payload()


def test_date_safe_reaction_uses_sessions_strictly_around_event() -> None:
    event = _event("one", "https://issuer.example/one", "Release")
    security = [_candle("2026-08-11", 100), _candle("2026-08-12", 900), _candle("2026-08-13", 110)]
    benchmark = [_candle("2026-08-11", 200), _candle("2026-08-12", 1), _candle("2026-08-13", 202)]
    target = daily_target(event, security, benchmark)
    assert target is not None
    assert target["baseline_session_date"] == "2026-08-11"
    assert target["target_session_date"] == "2026-08-13"
    assert target["security_return"] == "0.1"
    assert target["benchmark_return"] == "0.01"


def test_targets_are_separate_and_predictive_unit_is_event() -> None:
    event = _event("one", "https://issuer.example/one", "Release")
    row = EventMarketRow(
        event,
        REACTION_DAILY,
        datetime(2026, 8, 11, 23, 59, 59, tzinfo=UTC),
        {"primary_event_type": "OTHER"},
        {"return_1d": 0.01},
        {"qwen_used": False},
    ).feature_payload()
    metadata = cast("dict[str, Any]", row["metadata"])
    market_features = cast("dict[str, object]", row["market_features"])
    assert metadata["predictive_unit"] == PREDICTIVE_UNIT == "EVENT"
    assert metadata["market_only_daily_rows_as_event_examples"] is False
    assert "labels" not in row and "targets" not in row
    assert leakage_pass(row)
    market_features["abnormal_return"] = 1.0
    assert not leakage_pass(row)


def test_frozen_nlp_and_market_research_guards() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    assert DEFAULT_OLLAMA_THINK is False
    assert DEFAULT_AI_RANDOM_SEED == 0


def test_generated_contract_has_no_order_or_secret_capability() -> None:
    root = Path("src/event_market_dataset")
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = ("place_order", "post_order", "authorization", "tinvest_readonly_token")
    assert not any(value in text.lower() for value in forbidden)
    assert 'observed_market_test_used": False' in text
    assert 'future_holdout_evaluated": False' in text
    application = Path("src/event_market_dataset/application.py").read_text(encoding="utf-8")
    assert '"source_selection_used_future_returns": False' in application
    assert '"events_downsampled_by_reaction": False' in application


def _instrument(ticker: str, name: str, uid: str) -> dict[str, str]:
    return {"ticker": ticker, "name": name, "instrument_uid": uid, "figi": f"figi-{ticker}"}


def _archive_config() -> ArchiveSourceConfig:
    return ArchiveSourceConfig(
        source_code="YANDEX_IR_ARCHIVE_DATE_ONLY",
        source_name="Yandex IR",
        official_owner="Yandex",
        ticker="YDEX",
        issuer_name="Yandex",
        instrument_uid="uid",
        figi="figi",
        url_template="https://ir.yandex.ru/press-releases?year={page}",
        page_values=(2026,),
        source_type="ISSUER_ARCHIVE",
        collection_method="bounded year page",
        historical_range="2026",
        live_supported=False,
        parser_profile="YANDEX",
    )


def _event(source_id: str, url: str, title: str) -> AcquiredEvent:
    return AcquiredEvent.create(
        source_code="OFFICIAL",
        source_item_id=source_id,
        source_url=url,
        ticker="YDEX",
        issuer_name="Yandex",
        instrument_uid="uid",
        figi="figi",
        title=title,
        publication_date=date(2026, 8, 12),
        published_at=None,
        timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
    )


def _candle(day: str, close: int) -> dict[str, object]:
    return {"trade_date": day, "close": close}
