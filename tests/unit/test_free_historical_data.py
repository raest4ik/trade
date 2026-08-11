from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from apps.cli.collect_live_news import build_parser as build_live_parser
from src.free_historical_data.domain import (
    DATA_BUDGET,
    MAX_PILOT_ITEMS,
    AcquisitionBounds,
    DiscoveryIdentity,
    FreeSourceAudit,
    FreeSourceStatus,
    readiness_for_feature_rows,
    select_pilot_candidates,
    source_volume_summary,
)
from src.free_historical_data.providers import (
    BoundedHttpClient,
    JsonLdArticleParser,
    PaginatedIssuerArchiveNewsSource,
    PublicJsonNewsSource,
    SitemapArchiveNewsSource,
    parse_sitemap,
)
from src.free_historical_data.registry import (
    compliant_exact_audits,
    free_source_audits,
    source_volumes,
)
from src.free_historical_data.reporting import (
    CumulativeCorpusState,
    PilotState,
    write_zero_cost_reports,
)
from src.historical_news.domain.entities import HistoricalSourceItem
from src.historical_news.domain.enums import ContentStoragePolicy
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL

FROM = datetime(2020, 1, 1, tzinfo=UTC)
TO = datetime(2027, 1, 1, tzinfo=UTC)


def test_data_budget_is_permanently_zero_and_paid_sources_are_rejected() -> None:
    assert DATA_BUDGET == "ZERO"
    paid = [audit for audit in free_source_audits() if not audit.free]
    assert {audit.source_code for audit in paid} == {
        "CBONDS_NEWS_API",
        "INTERFAX_DISCLOSURE_API",
        "MOEX_CORPORATE_INFORMATION",
    }
    assert all(audit.status == FreeSourceStatus.REJECTED_PAID for audit in paid)


def test_source_audit_covers_issuer_universe_and_sberp_uses_sber() -> None:
    audits = free_source_audits()
    tickers = {ticker for audit in audits for ticker in audit.tickers}
    assert {"SBER", "SBERP", "GAZP", "LKOH", "ROSN", "NVTK", "YDEX", "T", "VTBR", "GMKN"} <= tickers
    sber = next(audit for audit in audits if audit.source_code == "SBER_IR_ARCHIVE")
    assert sber.tickers == ("SBER", "SBERP")
    assert sber.issuer_owned is True


def test_only_two_sources_are_accepted_exact() -> None:
    accepted = compliant_exact_audits()
    assert {audit.source_code for audit in accepted} == {
        "ROSNEFT_PRESS_RELEASES_RSS",
        "YANDEX_IR_PRESS_RELEASES_RSS",
    }
    assert all(audit.free and audit.automation_allowed for audit in accepted)


def test_exact_source_requires_source_clock_and_verified_timezone() -> None:
    base = next(
        audit for audit in free_source_audits() if audit.status == FreeSourceStatus.COMPLIANT_EXACT
    )
    payload = {field: getattr(base, field) for field in base.__dataclass_fields__}
    payload["timezone_verified"] = False
    with pytest.raises(ValueError, match="timezone"):
        FreeSourceAudit(**payload).validate()


def test_date_only_source_cannot_be_promoted_to_exact() -> None:
    base = next(
        audit for audit in free_source_audits() if audit.source_code == "LUKOIL_PRESS_ARCHIVE"
    )
    payload = {field: getattr(base, field) for field in base.__dataclass_fields__}
    payload.update(
        status=FreeSourceStatus.COMPLIANT_EXACT,
        automation_allowed=True,
        storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
        blocking_reason=None,
    )
    with pytest.raises(ValueError, match="date and clock"):
        FreeSourceAudit(**payload).validate()


def test_acquisition_is_bounded_and_low_concurrency() -> None:
    good = AcquisitionBounds(FROM, TO, "issuer", 200, True, concurrency=2)
    good.validate()
    with pytest.raises(ValueError, match="pilot limit"):
        AcquisitionBounds(FROM, TO, "issuer", MAX_PILOT_ITEMS + 1, True).validate()
    with pytest.raises(ValueError, match="concurrency"):
        AcquisitionBounds(FROM, TO, "issuer", 1, True, concurrency=3).validate()


def test_first_seen_is_separate_and_cannot_replace_publication_time() -> None:
    identity = DiscoveryIdentity(
        source_code="ISSUER",
        source_item_id="one",
        source_url="https://issuer.invalid/one",
        published_at=FROM,
        first_seen_at=TO,
        content_hash=None,
    )
    identity.validate()
    with pytest.raises(ValueError, match="first_seen_at"):
        DiscoveryIdentity(
            source_code="ISSUER",
            source_item_id="one",
            source_url="https://issuer.invalid/one",
            published_at=FROM,
            first_seen_at=FROM,
            content_hash=None,
        ).validate()


def test_stable_identity_deduplicates_and_selection_ignores_predictions_and_returns() -> None:
    common: dict[str, Any] = {
        "source_code": "ISSUER",
        "source_item_id": "one",
        "published_at": "2026-01-01T10:00:00+03:00",
    }
    duplicate = {**common, "rules_prediction": "DIVIDEND", "future_return": 99}
    alternate = {**common, "rules_prediction": "OTHER", "future_return": -99}
    selected_a = select_pilot_candidates([duplicate], existing_stable_keys=set(), limit=10)
    selected_b = select_pilot_candidates([alternate], existing_stable_keys=set(), limit=10)
    assert [(row["source_code"], row["source_item_id"]) for row in selected_a] == [
        (row["source_code"], row["source_item_id"]) for row in selected_b
    ]
    assert (
        len(select_pilot_candidates([duplicate, alternate], existing_stable_keys=set(), limit=10))
        == 1
    )


def test_source_volume_separates_verified_from_estimated() -> None:
    summary = source_volume_summary(source_volumes())
    assert summary["verified"] == {
        "available_items": 40,
        "exact_items": 40,
        "date_only_items": 0,
    }
    assert summary["estimated_additional"]["available_items"] == 8056
    assert summary["eligible"]["exact_items"] == 40


def test_sitemap_parser_distinguishes_index_and_urls() -> None:
    index = b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://issuer.invalid/news.xml</loc></sitemap></sitemapindex>'
    urlset = b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://issuer.invalid/news/1</loc></url></urlset>'
    assert parse_sitemap(index) == (["https://issuer.invalid/news.xml"], [])
    assert parse_sitemap(urlset) == ([], ["https://issuer.invalid/news/1"])


async def test_sitemap_provider_is_bounded_and_uses_jsonld_publication_time() -> None:
    responses = {
        "/sitemap.xml": b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://issuer.invalid/news/1</loc></url><url><loc>https://issuer.invalid/other/2</loc></url></urlset>',
        "/news/1": b'<script type="application/ld+json">{"@type":"NewsArticle","headline":"Issuer release","datePublished":"2026-01-02T10:30:00+03:00","description":"excerpt","url":"https://issuer.invalid/news/1"}</script>',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=responses[request.url.path], request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = _transport(client)
        provider = SitemapArchiveNewsSource(
            sitemap_urls=("https://issuer.invalid/sitemap.xml",),
            article_parser=JsonLdArticleParser(
                storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
                source_timezone=None,
            ),
            transport=transport,
            news_url_filter=lambda url: "/news/" in url,
            max_sitemaps=1,
            max_article_requests=1,
        )
        page = await provider.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    assert len(page.items) == 1
    assert page.items[0].original_timestamp_text == "2026-01-02T10:30:00+03:00"
    assert page.items[0].fetched_at != datetime(2026, 1, 2, 7, 30, tzinfo=UTC)
    assert transport.request_count == 2


async def test_paginated_provider_stops_at_configured_page_bound() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"page", request=request)

    def parse_page(
        page_url: str, payload: bytes, fetched_at: datetime
    ) -> tuple[list[HistoricalSourceItem], bool]:
        assert payload == b"page"
        page_number = page_url.rsplit("=", 1)[-1]
        return [_item(f"page-{page_number}", fetched_at)], True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = PaginatedIssuerArchiveNewsSource(
            page_url=lambda page: f"https://issuer.invalid/archive?page={page}",
            page_parser=parse_page,
            transport=_transport(client),
            max_pages=2,
        )
        first = await provider.fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=None, limit=10
        )
        second = await provider.fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=first.next_cursor, limit=10
        )
    assert first.next_cursor == "2"
    assert second.next_cursor is None


async def test_public_json_provider_uses_mocked_http_and_pagination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [request.url.params["page"]]}, request=request)

    def parse_json(payload: bytes, fetched_at: datetime) -> tuple[list[HistoricalSourceItem], bool]:
        data = json.loads(payload)
        return [_item(f"json-{data['items'][0]}", fetched_at)], True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = PublicJsonNewsSource(
            page_url=lambda page, limit: f"https://issuer.invalid/api?page={page}&limit={limit}",
            page_parser=parse_json,
            transport=_transport(client),
            max_pages=1,
        )
        page = await provider.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=5)
    assert page.items[0].source_item_id == "json-1"
    assert page.next_cursor is None


async def test_transport_enforces_https_allowlist_retries_and_response_limit() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = _transport(client, max_retries=1)
        assert await transport.get("https://issuer.invalid/data") == b"ok"
        with pytest.raises(Exception, match="allowlist"):
            await transport.get("https://other.invalid/data")
    assert attempts == 2


def test_live_cli_only_exposes_compliant_sources_and_caps_limit() -> None:
    parser = build_live_parser()
    parsed = parser.parse_args(
        [
            "--source-code",
            "ROSNEFT_PRESS_RELEASES_RSS",
            "--from",
            "2026-01-01",
            "--to",
            "2026-02-01",
            "--limit",
            "200",
            "--dry-run",
        ]
    )
    assert parsed.limit == 200
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source-code",
                "GAZPROM_PRESS_ARCHIVE",
                "--from",
                "2026-01-01",
                "--to",
                "2026-02-01",
            ]
        )


def test_readiness_count_thresholds_do_not_override_diversity_gates() -> None:
    assert (
        readiness_for_feature_rows(21, ticker_count=2, source_count=2, month_count=4)["status"]
        == "NOT_READY"
    )
    result = readiness_for_feature_rows(1000, ticker_count=1, source_count=1, month_count=1)
    assert result["status"] == "PILOT_ONLY"
    assert result["market_regime_diversity"] == "UNKNOWN"


def test_reports_are_machine_readable_and_pilot_is_zero_when_no_new_source(tmp_path: Path) -> None:
    paths = write_zero_cost_reports(
        tmp_path,
        audits=free_source_audits(),
        volumes=source_volumes(),
        cumulative=CumulativeCorpusState(
            real=40,
            real_exact=40,
            matched=35,
            ambiguous=0,
            unmatched=5,
            reaction_ready=26,
            feature_ready=21,
            ticker_distribution={"ROSN": 19, "YDEX": 16},
            source_distribution={
                "ROSNEFT_PRESS_RELEASES_RSS": 20,
                "YANDEX_IR_PRESS_RELEASES_RSS": 20,
            },
            date_from="2025-06-20T12:30:00Z",
            date_to="2026-08-11T08:00:00Z",
            month_count=7,
        ),
        pilot=PilotState(),
        discovered_items=[],
    )
    pilot = json.loads(paths["pilot_manifest"].read_text(encoding="utf-8"))
    growth = json.loads(paths["growth_plan"].read_text(encoding="utf-8"))
    assert pilot["new_real_imported"] == 0
    assert pilot["uses_future_returns"] is False
    assert paths["discovered_items"].read_text(encoding="utf-8") == ""
    assert growth["estimated_time_to_100"] == "UNKNOWN"


def test_frozen_nlp_identity_is_unchanged() -> None:
    assert (
        EXPECTED_RULES_FINGERPRINT
        == "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
    )
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"


def test_provider_unit_tests_use_no_live_http() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "httpx.MockTransport" in source
    assert 'await transport.get("https://issuer.invalid/data")' in source


def test_acquisition_modules_contain_no_credentials_or_private_api_shortcuts() -> None:
    root = Path(__file__).parents[2]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "src" / "free_historical_data").glob("*.py"))
    ).lower()
    assert "api_key=" not in source
    assert "bearer " not in source
    assert "private api" not in source
    assert "captcha bypass" not in source


def _transport(
    client: httpx.AsyncClient,
    *,
    max_retries: int = 0,
) -> BoundedHttpClient:
    return BoundedHttpClient(
        allowed_hosts=("issuer.invalid",),
        timeout_seconds=1,
        max_retries=max_retries,
        min_request_interval_seconds=0.1,
        concurrency=1,
        max_response_bytes=100_000,
        user_agent="unit-test",
        client=client,
        sleep=False,
    )


def _item(source_item_id: str, fetched_at: datetime) -> HistoricalSourceItem:
    return HistoricalSourceItem(
        source_item_id=source_item_id,
        source_url=f"https://issuer.invalid/news/{source_item_id}",
        title=f"Release {source_item_id}",
        published_at_text="2026-01-02T10:30:00+03:00",
        source_timezone=None,
        content="excerpt",
        content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
        content_is_excerpt=True,
        original_timestamp_text="2026-01-02T10:30:00+03:00",
        corrects_source_item_id=None,
        fetched_at=fetched_at,
    )
