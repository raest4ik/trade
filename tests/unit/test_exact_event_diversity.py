from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_diversity.application import (
    assert_holdout_guard,
    assert_rows_preserved,
    assert_v1_preserved,
    build_diversity_source_registry,
    exclude_v1_duplicates,
)
from src.exact_event_diversity.domain import (
    FROZEN_V1_COUNTS,
    FROZEN_V1_HASHES,
    concentration,
    exact_model_data_status,
    feature_ready_gap,
    parse_explicit_utc,
)
from src.exact_event_diversity.sources import (
    OfficialSourceProfile,
    acquire_embedded_app_state,
    acquire_moex_rss,
    acquire_tbank_public_news,
    acquire_vk_next_state,
    acquire_x5_wordpress,
)


def test_exact_timestamp_requires_explicit_timezone() -> None:
    assert parse_explicit_utc("2026-08-10T10:15:30+03:00") == datetime(
        2026, 8, 10, 7, 15, 30, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="TIMESTAMP_TIMEZONE_UNRESOLVED"):
        parse_explicit_utc("2026-08-10T10:15:30")


@pytest.mark.asyncio
async def test_x5_official_wordpress_adapter_is_bounded_and_exact(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            request=request,
            json=[
                {
                    "id": 7,
                    "date_gmt": "2026-08-10T07:15:30",
                    "link": "https://www.x5.ru/ru/news/release/",
                    "slug": "release",
                    "title": {"rendered": "X5 &amp; release"},
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await acquire_x5_wordpress(
            _profile("X5", "https://www.x5.ru/wp-json/wp/v2/news", "www.x5.ru"),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=10,
            cache_dir=tmp_path,
            client=client,
        )
    assert len(rows) == 1
    assert rows[0].publication_timestamp_utc == datetime(2026, 8, 10, 7, 15, 30, tzinfo=UTC)
    assert rows[0].publication_timestamp_raw == "2026-08-10T07:15:30"
    assert calls[0].url.params["lang"] == "ru"
    assert "authorization" not in calls[0].headers


@pytest.mark.asyncio
async def test_tbank_public_page_api_adapter_uses_public_exact_field(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "response": {
                    "items": [
                        {
                            "id": "42",
                            "slug": "official-release",
                            "publishedAt": "2026-08-09T12:30:00.000Z",
                            "title": "T-Bank release",
                            "textshort": "Public newsroom item",
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await acquire_tbank_public_news(
            _profile(
                "T",
                "https://cfg.tbank.ru/about/public/api/news/platform/v1/getArticles",
                "cfg.tbank.ru",
            ),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=10,
            cache_dir=tmp_path,
            client=client,
        )
    assert rows[0].source_item_id == "42"
    assert rows[0].publication_timestamp_utc == datetime(2026, 8, 9, 12, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_vk_public_next_state_adapter_parses_exact_timestamp(tmp_path: Path) -> None:
    payload = {
        "props": {
            "pageProps": {
                "publications": [
                    {
                        "id": 11,
                        "pub_date": "2026-08-08T09:40:00Z",
                        "public_url": "press/releases/release-11/",
                        "title": "VK release",
                    }
                ]
            }
        }
    }
    page = f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text=page)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await acquire_vk_next_state(
            _profile("VKCO", "https://vk.company/ru/press/releases/", "vk.company"),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=10,
            cache_dir=tmp_path,
            client=client,
        )
    assert rows[0].canonical_url == "https://vk.company/ru/press/releases/release-11"
    assert rows[0].publication_timestamp_utc == datetime(2026, 8, 8, 9, 40, tzinfo=UTC)


@pytest.mark.asyncio
async def test_novabev_app_state_rejects_source_local_midnight(tmp_path: Path) -> None:
    exact = int(datetime(2026, 8, 7, 12, 5, tzinfo=UTC).timestamp())
    midnight_moscow = int(datetime(2026, 8, 6, 21, 0, tzinfo=UTC).timestamp())
    page = json.dumps(
        {
            "news": {
                "items": [
                    {"name": "Exact", "detailPageUrl": "/news/exact/", "activeFrom": exact},
                    {
                        "name": "Date placeholder",
                        "detailPageUrl": "/news/date-only/",
                        "activeFrom": midnight_moscow,
                    },
                ]
            }
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text=f"App = {page};")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await acquire_embedded_app_state(
            _profile("BELU", "https://novabev.com/en/investors/news/", "novabev.com"),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=10,
            cache_dir=tmp_path,
            client=client,
        )
    assert [row.title for row in rows] == ["Exact"]


@pytest.mark.asyncio
async def test_moex_rss_filters_after_xml_entity_decoding(tmp_path: Path) -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<rss><channel><item>
<title>Группа компаний &quot;Самолет&quot; разместила выпуск</title>
<description>SMLT official notice</description>
<link>https://www.moex.com/n9001</link>
<pubDate>Fri, 07 Aug 2026 15:20:00 +0300</pubDate>
</item></channel></rss>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=xml.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await acquire_moex_rss(
            _profile("SMLT", "https://www.moex.com/export/news.aspx?cat=100", "www.moex.com"),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=10,
            cache_dir=tmp_path,
            required_phrases=('Группа компаний "Самолет',),
            client=client,
        )
    assert rows[0].publication_timestamp_utc == datetime(2026, 8, 7, 12, 20, tzinfo=UTC)


@pytest.mark.asyncio
async def test_private_or_non_allowlisted_endpoint_is_rejected_before_request(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="OFFICIAL_SOURCE_URL_REJECTED"):
            await acquire_x5_wordpress(
                _profile("X5", "https://private.example/internal", "www.x5.ru"),
                date_from=date(2026, 8, 1),
                date_to=date(2026, 8, 10),
                item_limit=10,
                cache_dir=tmp_path,
                client=client,
            )
    assert calls == 0


@pytest.mark.asyncio
async def test_acquisition_limits_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MUST_BE_BOUNDED"):
        await acquire_x5_wordpress(
            _profile("X5", "https://www.x5.ru/wp-json/wp/v2/news", "www.x5.ru"),
            date_from=date(2026, 8, 1),
            date_to=date(2026, 8, 10),
            item_limit=201,
            cache_dir=tmp_path,
        )


def test_v1_frozen_contract_and_prefix_are_preserved() -> None:
    assert FROZEN_V1_COUNTS == {
        "exact_timestamp_events": 449,
        "exact_reaction_ready": 342,
        "exact_feature_ready": 239,
        "exact_unique_tickers": 4,
        "exact_unique_issuers": 4,
    }
    assert len(FROZEN_V1_HASHES) == 6
    old_events: list[dict[str, object]] = [{"metadata": {"event_id": "old"}}]
    old_features: list[dict[str, object]] = [{"event_id": "old"}]
    assert_v1_preserved(
        old_events,
        [*old_events, {"metadata": {"event_id": "new"}}],
        old_features,
        [*old_features, {"event_id": "new"}],
    )
    with pytest.raises(ValueError, match="EXACT_V1_NOT_PRESERVED"):
        assert_v1_preserved(old_events, [], old_features, [])
    assert_rows_preserved(old_events, old_events, artifact="events")


def test_diversity_registry_keeps_315_instruments_and_adds_verified_sources(
    tmp_path: Path,
) -> None:
    new_tickers = ["X5", "VKCO", "T", "BELU", "MOEX", "SMLT", "VTBR"]
    tickers = [*new_tickers, *(f"T{index:03d}" for index in range(308))]
    instruments = [
        {"ticker": ticker, "name": f"Issuer {ticker}", "instrument_uid": f"uid-{ticker}"}
        for ticker in tickers
    ]
    mapping = tmp_path / "mapping.json"
    previous = tmp_path / "registry.jsonl"
    mapping.write_text(
        json.dumps(
            {
                "instruments": [
                    *instruments,
                    {"ticker": "IMOEX", "name": "Benchmark", "instrument_uid": "uid-imoex"},
                ]
            }
        ),
        encoding="utf-8",
    )
    previous.write_text(
        "".join(
            json.dumps(
                {
                    "ticker": item["ticker"],
                    "official_domain": None,
                    "source_url": None,
                    "source_family": None,
                    "parser_version": None,
                    "timestamp_capability": "UNKNOWN",
                    "timestamp_field_source": None,
                    "timezone_semantics": "UNKNOWN",
                    "historical_archive_start": None,
                    "historical_archive_end": None,
                    "incremental_supported": False,
                    "public_access": False,
                    "payment_required": False,
                    "auth_required": False,
                    "source_policy_status": "UNKNOWN_FAIL_CLOSED",
                    "collector_status": "NO_OFFICIAL_NEWS_ARCHIVE",
                    "reason": "No verified source",
                }
            )
            + "\n"
            for item in instruments
        ),
        encoding="utf-8",
    )
    registry = build_diversity_source_registry(mapping, previous)
    assert len(registry) == 315
    selected = {row.ticker: row for row in registry if row.ticker in new_tickers}
    assert set(selected) == set(new_tickers)
    assert all(row.public_access and not row.auth_required for row in selected.values())
    assert all(
        row.source_url and row.source_url.startswith("https://") for row in selected.values()
    )


def test_v1_duplicate_source_identity_and_url_are_excluded() -> None:
    first = _event("source-1", "https://issuer.example/one")
    second = _event("source-2", "https://issuer.example/two")
    old: list[dict[str, object]] = [
        {
            "metadata": {
                "source_code": first.source_code,
                "source_item_id": first.source_item_id,
                "canonical_url": first.canonical_url,
            }
        }
    ]
    selected, excluded = exclude_v1_duplicates([first, second], old)
    assert selected == [second]
    assert excluded[0]["reason"] == "EXACT_V1_DUPLICATE_PRESERVED"


def test_feature_ready_gap_reconciles_exact_reasons() -> None:
    rows = [
        _gap_row({}, {}),
        _gap_row({"primary_event_type": "OTHER"}, {}),
        _gap_row(
            {"primary_event_type": "OTHER"},
            {"pre_return_5m": None, "imoex_pre_return_5m": 0.1},
        ),
    ]
    result = feature_ready_gap(rows)
    assert result["count"] == 3
    reasons = cast("dict[str, int]", result["reasons"])
    assert sum(reasons.values()) == 3
    assert reasons == {
        "cluster_exclusion": 0,
        "market_history_warmup": 1,
        "missing_event_features": 1,
        "missing_pre_event_market_context": 1,
        "other": 0,
        "source_policy_issue": 0,
        "ticker_mapping": 0,
    }


def test_concentration_and_readiness_require_diverse_feature_rows() -> None:
    diagnostic = concentration(Counter({"MGNT": 400, "YDEX": 22, "ROSN": 20, "GMKN": 7}))
    assert abs(cast("float", diagnostic["top_share"]) - 400 / 449) < 1e-12
    assert cast("float", diagnostic["effective_count"]) > 1
    assert (
        exact_model_data_status(
            feature_ready=250,
            feature_ready_by_ticker=Counter({f"T{index}": 25 for index in range(10)}),
        )
        == "READY_FOR_EXACT_BASELINE_EXPERIMENT"
    )
    assert (
        exact_model_data_status(
            feature_ready=250,
            feature_ready_by_ticker=Counter({"MGNT": 250}),
        )
        == "NOT_READY_FOR_EXACT_MODEL"
    )


def test_future_holdout_guard_rejects_targets_and_visible_outcomes() -> None:
    future: dict[str, object] = {
        "metadata": {"event_id": "future", "future_holdout": True},
        "target_availability": {"research_outcomes_visible": False},
    }
    assert_holdout_guard([future], [])
    with pytest.raises(ValueError, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        assert_holdout_guard([future], [{"event_id": "future"}])
    future["target_availability"] = {"research_outcomes_visible": True}
    with pytest.raises(ValueError, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        assert_holdout_guard([future], [])


def test_frozen_nlp_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA


def test_diversity_package_has_no_model_training_or_trading_capability() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("src/exact_event_diversity").glob("*.py")
    ).lower()
    forbidden = (
        "logisticregression",
        "histgradientboosting",
        "ridgeclassifier",
        ".fit(",
        ".predict(",
        "place_order",
        "postorder",
        "buy_order",
        "sell_order",
    )
    assert not any(value in text for value in forbidden)
    assert '"model_trained": false' in text
    assert '"orders_submitted": false' in text


def _profile(ticker: str, source_url: str, allowed_host: str) -> OfficialSourceProfile:
    return OfficialSourceProfile(
        source_code=f"{ticker}_OFFICIAL_EXACT",
        ticker=ticker,
        issuer=f"Issuer {ticker}",
        instrument_uid=f"uid-{ticker}",
        source_url=source_url,
        allowed_host=allowed_host,
        timestamp_field="official.publication_timestamp",
    )


def _event(source_item_id: str, url: str) -> ExactEvent:
    return ExactEvent.create(
        source_code="OFFICIAL_EXACT",
        source_item_id=source_item_id,
        canonical_url=url,
        ticker="TEST",
        issuer="Issuer",
        instrument_uid="uid",
        title="Release",
        publication_timestamp_raw="2026-08-10T10:00:00Z",
        publication_timestamp_utc=datetime(2026, 8, 10, 10, tzinfo=UTC),
        timestamp_source_field="publishedAt",
    )


def _gap_row(
    event_features: dict[str, object], market_features: dict[str, object]
) -> dict[str, object]:
    return {
        "metadata": {
            "instrument_uid": "uid",
            "event_cluster_id": "cluster",
            "storage_policy": "METADATA_TITLE_HASH_ONLY",
        },
        "event_features": event_features,
        "pre_event_market_features": market_features,
        "target_availability": {"reaction_ready": True, "feature_ready": False},
    }
