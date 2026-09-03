from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_accumulation.domain import (
    SealedLiveEpochOutcomeReadError,
    guard_sealed_live_epoch_post_event_price_read,
)
from src.free_live_issuer_expansion_v2.application import (
    FeatureReadinessBlocker,
    LiveFeatureMarketProvider,
    SourceProbeConfig,
    SourceStatus,
    StaticFeatureMarketProviderFactory,
    TimestampLevel,
    diagnose_shadow_event_feature_readiness,
    discover_html_alternates,
    probe_candidate_source,
    run_free_live_issuer_source_expansion_v2,
    seed_identity,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch

PUBLISHED = datetime(2026, 8, 25, 8, 44, tzinfo=UTC)


def test_sealed_guard_allows_safe_pre_event_read_and_rejects_post_event() -> None:
    guard_sealed_live_epoch_post_event_price_read(
        epoch="LIVE_SHADOW_CORPUS",
        published_at=PUBLISHED,
        query_end_at=PUBLISHED,
        context="feature",
    )
    guard_sealed_live_epoch_post_event_price_read(
        epoch="LIVE_SHADOW_CORPUS",
        published_at=PUBLISHED,
        query_end_at=PUBLISHED - timedelta(minutes=1),
        context="feature",
    )
    with pytest.raises(
        SealedLiveEpochOutcomeReadError, match="SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT"
    ):
        guard_sealed_live_epoch_post_event_price_read(
            epoch="LIVE_SHADOW_CORPUS",
            published_at=PUBLISHED,
            query_end_at=PUBLISHED + timedelta(microseconds=1),
            context="feature",
        )


def test_pre_event_feature_readiness_uses_canonical_builder_and_upper_bound() -> None:
    manifest = asyncio.run(_diagnose(_client()))

    assert manifest["feature_ready_after"] is True
    assert manifest["feature_builder_invoked"] is True
    assert manifest["exact_blocker"] is None
    assert manifest["max_market_timestamp_read"] <= PUBLISHED.isoformat()
    assert manifest["feature_builder_result"]["relative_returns"]["5"] is not None


def test_unfinished_candle_containing_publication_is_excluded() -> None:
    future = _candle("uid-ROSN", PUBLISHED, close="130")
    manifest = asyncio.run(_diagnose(_client(extra_security=(future,))))

    assert manifest["feature_ready_after"] is True
    assert manifest["timestamp_violation"] is False
    assert manifest["max_market_timestamp_read"] < future.end_at.isoformat()


def test_benchmark_future_candle_rejected_by_canonical_builder_when_not_prefiltered() -> None:
    future = _candle("uid-IMOEX", PUBLISHED, close="3000")
    manifest = asyncio.run(_diagnose(_client(extra_benchmark=(future,), prefilter=False)))

    assert manifest["feature_ready_after"] is False
    assert manifest["timestamp_violation"] is True
    assert manifest["exact_blocker"] == FeatureReadinessBlocker.FEATURE_CONTRACT_FAILURE.value


def test_missing_benchmark_and_market_data_are_explicit_blockers() -> None:
    missing_benchmark = asyncio.run(_diagnose(_client(benchmark=())))
    missing_security = asyncio.run(_diagnose(_client(security=())))

    assert missing_benchmark["exact_blocker"] == FeatureReadinessBlocker.BENCHMARK_DATA_UNAVAILABLE
    assert missing_security["exact_blocker"] == FeatureReadinessBlocker.MARKET_DATA_UNAVAILABLE


def test_source_discovery_accepts_rss_atom_jsonld_and_first_party_json() -> None:
    rss = probe_candidate_source(
        _source("RSS", "https://issuer.test/rss", parser="rss-item-pubdate-explicit-offset-v1"),
        client=_Http({"https://issuer.test/rss": _rss()}),
    )
    atom = probe_candidate_source(
        _source(
            "ATOM", "https://issuer.test/atom", parser="atom-entry-published-explicit-offset-v1"
        ),
        client=_Http({"https://issuer.test/atom": _atom()}),
    )
    jsonld = probe_candidate_source(
        _source("JSONLD", "https://issuer.test/news", parser="html-alternate-jsonld-v1"),
        client=_Http({"https://issuer.test/news": _jsonld("2026-08-25T11:44:16+03:00")}),
    )
    endpoint = probe_candidate_source(
        _source(
            "JSON",
            "https://issuer.test/api/news",
            parser="json-endpoint-datepublished-v1",
            timestamp_field="datePublished",
            identity_field="id",
        ),
        client=_Http({"https://issuer.test/api/news": _json_endpoint()}),
    )

    assert rss.status == SourceStatus.LIVE_STRICT_EXACT_READY
    assert atom.status == SourceStatus.LIVE_STRICT_EXACT_READY
    assert jsonld.status == SourceStatus.LIVE_STRICT_EXACT_READY
    assert endpoint.status == SourceStatus.LIVE_STRICT_EXACT_READY
    assert {
        rss.timestamp_level,
        atom.timestamp_level,
        jsonld.timestamp_level,
        endpoint.timestamp_level,
    } == {TimestampLevel.LEVEL_A}


def test_html_alternate_discovery_resolves_rss_and_atom_links() -> None:
    alternates = discover_html_alternates(
        """
        <html><head>
          <link rel="alternate" type="application/rss+xml" href="/rss.xml">
          <link rel="alternate" type="application/atom+xml" href="https://issuer.test/atom.xml">
        </head></html>
        """,
        "https://issuer.test/news",
    )

    assert alternates == ("https://issuer.test/rss.xml", "https://issuer.test/atom.xml")


def test_jsonld_date_modified_cannot_substitute_publication_and_naive_rejected() -> None:
    modified = probe_candidate_source(
        _source("MOD", "https://issuer.test/news", parser="html-alternate-jsonld-v1"),
        client=_Http(
            {"https://issuer.test/news": _jsonld(None, date_modified="2026-08-25T11:44:16+03:00")}
        ),
    )
    naive = probe_candidate_source(
        _source("NAIVE", "https://issuer.test/news", parser="html-alternate-jsonld-v1"),
        client=_Http({"https://issuer.test/news": _jsonld("2026-08-25T11:44:16")}),
    )

    assert modified.status == SourceStatus.LIVE_TIMESTAMP_UNVERIFIED
    assert modified.blocker == "DATE_MODIFIED_CANNOT_SUBSTITUTE_PUBLICATION_TIME"
    assert naive.status == SourceStatus.LIVE_CLOCK_WITHOUT_TIMEZONE
    assert naive.timestamp_level == TimestampLevel.LEVEL_C


def test_date_only_rejected_and_documented_timezone_accepted() -> None:
    date_only = probe_candidate_source(
        _source("DATE", "https://issuer.test/news", parser="html-alternate-jsonld-v1"),
        client=_Http({"https://issuer.test/news": _jsonld("2026-08-25")}),
    )
    documented = probe_candidate_source(
        _source(
            "DOCUTC",
            "https://issuer.test/news",
            parser="html-alternate-jsonld-v1",
            document_timezone="UTC",
        ),
        client=_Http({"https://issuer.test/news": _jsonld("2026-08-25T11:44:16")}),
    )

    assert date_only.status == SourceStatus.LIVE_DATE_ONLY
    assert date_only.timestamp_level == TimestampLevel.LEVEL_D
    assert documented.status == SourceStatus.LIVE_STRICT_EXACT_READY
    assert documented.timestamp_level == TimestampLevel.LEVEL_B


def test_unofficial_endpoint_stable_identity_paid_fallback_and_bounded_retry() -> None:
    unofficial = probe_candidate_source(
        _source("BAD", "https://mirror.test/rss", parser="rss-item-pubdate-explicit-offset-v1"),
        client=_Http({}),
    )
    no_id = probe_candidate_source(
        _source("NOID", "https://issuer.test/rss", parser="rss-item-pubdate-explicit-offset-v1"),
        client=_Http({"https://issuer.test/rss": _rss(guid="", link="")}),
    )
    paid = probe_candidate_source(
        _source(
            "PAID",
            "https://issuer.test/rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            paid_source=True,
        ),
        client=_Http({}),
    )
    retry = probe_candidate_source(
        _source("RETRY", "https://issuer.test/rss", parser="rss-item-pubdate-explicit-offset-v1"),
        client=_Http(
            {
                "https://issuer.test/rss": [
                    FetchResult(
                        "https://issuer.test/rss",
                        "https://issuer.test/rss",
                        500,
                        None,
                        b"",
                        0,
                        (),
                        "HTTP_FAILURE",
                    ),
                    _rss(),
                ]
            }
        ),
    )

    assert unofficial.status == SourceStatus.LIVE_NOT_ISSUER_ORIGINATED
    assert no_id.status == SourceStatus.LIVE_NO_STABLE_ID
    assert paid.status == SourceStatus.OUT_OF_SCOPE_PAID_SOURCE
    assert retry.request_attempts == 2
    assert retry.status == SourceStatus.LIVE_STRICT_EXACT_READY


def test_duplicate_share_classes_and_same_legal_issuer_count_once(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text(json.dumps(_event()) + "\n", encoding="utf-8")
    manifest = asyncio.run(
        run_free_live_issuer_source_expansion_v2(
            output_root=tmp_path / "out",
            base_main_sha="5" * 40,
            git_sha="6" * 40,
            shadow_corpus_path=shadow,
            provider_factory=_factory(_client()),
            client=_Http(
                {
                    "https://www.gazprom.com/press/": _jsonld("2026-08-25T11:44:16+03:00"),
                    "https://www.lukoil.com/PressCenter/Pressreleases": _jsonld(
                        "2026-08-25T11:44:16+03:00"
                    ),
                    "https://www.novatek.ru/en/press/releases/": _jsonld(
                        "2026-08-25T11:44:16+03:00"
                    ),
                    "https://www.sberbank.com/news-and-media/press-releases": _jsonld(
                        "2026-08-25T11:44:16+03:00"
                    ),
                    "https://www.vtb.com/o-banke/press-centr/novosti-i-press-relizy/": _jsonld(
                        "2026-08-25T11:44:16+03:00"
                    ),
                }
            ),
            created_at=PUBLISHED,
        )
    )

    assert manifest["FEATURE_PIPELINE"] == "YES"
    assert manifest["SOURCE_DIVERSITY"] == "YES"
    assert "ПАО Сбербанк" in manifest["NEW_DISTINCT_LEGAL_ISSUERS"]
    assert manifest["NEW_DISTINCT_LEGAL_ISSUER_COUNT"] == len(
        set(manifest["NEW_DISTINCT_LEGAL_ISSUERS"])
    )
    assert manifest["PAID_SOURCE_FALLBACK_CONSIDERED"] is False


def test_existing_source_reaudit_records_new_hypothesis() -> None:
    config = next(item for item in _configs() if item.prior_rejection_source_id)

    assert config.new_hypothesis
    assert config.prior_rejection_source_id is not None


async def _diagnose(client: _MinuteClient) -> dict[str, Any]:
    return await diagnose_shadow_event_feature_readiness(
        _event(), provider_factory=_factory(client)
    )


def _factory(client: _MinuteClient) -> StaticFeatureMarketProviderFactory:
    identity = seed_identity("ROSN")
    assert identity is not None
    return StaticFeatureMarketProviderFactory(
        {
            "ROSN": LiveFeatureMarketProvider(
                instrument=identity,
                benchmark_uid="uid-IMOEX",
                client=client,
            )
        }
    )


def _event() -> dict[str, Any]:
    return {
        "event_id": "event-1",
        "ticker": "ROSN",
        "published_at": PUBLISHED.isoformat(),
        "source_id": "ROSN_SOURCE",
        "pre_event_feature_availability": {"available": False},
        "semantic_output": {"semantic_unknown": False},
    }


def _source(
    suffix: str,
    url: str,
    *,
    parser: str,
    timestamp_field: str = "datePublished",
    identity_field: str = "url",
    paid_source: bool = False,
    document_timezone: str | None = None,
) -> SourceProbeConfig:
    return SourceProbeConfig(
        source_id=f"AAA_{suffix}",
        ticker="AAA",
        legal_issuer="AAA Issuer",
        official_domain="issuer.test",
        url=url,
        mechanism="official_test",
        parser=parser,
        timestamp_field=timestamp_field,
        identity_field=identity_field,
        content_fields=("headline", "description"),
        new_hypothesis="test",
        paid_source=paid_source,
        document_timezone=document_timezone,
    )


@dataclass
class _MinuteClient:
    security: tuple[TInvestMinuteCandle, ...]
    benchmark: tuple[TInvestMinuteCandle, ...]
    prefilter: bool = True

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch:
        rows = self.benchmark if instrument_uid == "uid-IMOEX" else self.security
        if self.prefilter:
            rows = tuple(row for row in rows if date_from <= row.begin_at and row.end_at <= date_to)
        return TInvestMinuteCandleBatch(rows, ())


class _Http:
    def __init__(self, bodies: dict[str, bytes | FetchResult | list[bytes | FetchResult]]) -> None:
        self.bodies = bodies

    def get(self, url: str) -> FetchResult:
        value = self.bodies[url]
        if isinstance(value, list):
            body = value.pop(0)
        else:
            body = value
        if isinstance(body, FetchResult):
            return body
        content_type = "application/json" if body.lstrip().startswith(b"{") else "text/html"
        if body.lstrip().startswith(b"<rss"):
            content_type = "application/rss+xml"
        if body.lstrip().startswith(b"<feed"):
            content_type = "application/atom+xml"
        return FetchResult(url, url, 200, content_type, body, 0, (), None)


def _client(
    *,
    security: tuple[TInvestMinuteCandle, ...] | None = None,
    benchmark: tuple[TInvestMinuteCandle, ...] | None = None,
    extra_security: tuple[TInvestMinuteCandle, ...] = (),
    extra_benchmark: tuple[TInvestMinuteCandle, ...] = (),
    prefilter: bool = True,
) -> _MinuteClient:
    return _MinuteClient(
        security if security is not None else (*_candles("uid-ROSN", "100"), *extra_security),
        benchmark if benchmark is not None else (*_candles("uid-IMOEX", "2000"), *extra_benchmark),
        prefilter=prefilter,
    )


def _candles(uid: str, base: str) -> tuple[TInvestMinuteCandle, ...]:
    return tuple(
        _candle(uid, PUBLISHED - timedelta(minutes=120 - index), close=str(Decimal(base) + index))
        for index in range(120)
    )


def _candle(uid: str, begin_at: datetime, *, close: str) -> TInvestMinuteCandle:
    value = Decimal(close)
    return TInvestMinuteCandle(
        instrument_uid=uid,
        begin_at=begin_at,
        end_at=begin_at + timedelta(minutes=1),
        open=value,
        high=value,
        low=value,
        close=value,
        volume=100,
        is_complete=True,
    )


def _rss(guid: str = "guid-1", link: str = "https://issuer.test/news/1") -> bytes:
    guid_tag = f"<guid>{guid}</guid>" if guid else ""
    link_tag = f"<link>{link}</link>" if link else ""
    return f"""<rss version="2.0"><channel><item>
<title>News</title>{link_tag}{guid_tag}
<pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
<description>Body</description>
</item></channel></rss>""".encode()


def _atom() -> bytes:
    return b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>atom-1</id><title>Atom</title><link href="https://issuer.test/a/1"/>
<published>2026-08-25T11:44:16+03:00</published><summary>Body</summary>
</entry></feed>"""


def _jsonld(date_published: str | None, *, date_modified: str | None = None) -> bytes:
    payload = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "url": "https://issuer.test/news/1",
        "headline": "Headline",
        "description": "Body",
    }
    if date_published is not None:
        payload["datePublished"] = date_published
    if date_modified is not None:
        payload["dateModified"] = date_modified
    return (
        '<html><head><script type="application/ld+json">'
        + json.dumps(payload)
        + "</script></head></html>"
    ).encode()


def _json_endpoint() -> bytes:
    return json.dumps(
        {
            "items": [
                {
                    "id": "item-1",
                    "url": "https://issuer.test/news/1",
                    "datePublished": "2026-08-25T11:44:16+03:00",
                    "headline": "Headline",
                    "description": "Body",
                }
            ]
        }
    ).encode()


def _configs() -> tuple[SourceProbeConfig, ...]:
    from src.free_live_issuer_expansion_v2.application import default_candidate_sources

    return default_candidate_sources()
