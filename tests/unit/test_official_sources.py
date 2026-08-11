from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from src.corpus_quality.domain import (
    PublicationTimeRecord,
    cumulative_funnel,
    diversity_warnings,
    select_annotation_batch,
)
from src.historical_news.domain.entities import HistoricalSourceItem
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.domain.time import parse_publication_timestamp
from src.historical_news.infrastructure.issuer_feed import IssuerFeedNewsSource
from src.news.domain.enums import PublicationTimestampQuality
from src.official_sources.domain import (
    ANNOTATION_BATCH_VERSION,
    ControlledImport,
    OfficialSourceConfig,
    OfficialSourceStatus,
    audit_payload,
)
from src.official_sources.registry import (
    YANDEX_FEED_URL,
    official_source_configs,
    reaction_ready_configs,
)
from src.official_sources.reporting import (
    FROZEN_BATCH_002_SHA256,
    represented_ticker_distribution,
)
from src.reaction_ready_corpus.domain import (
    CorpusProvenance,
    classify_provenance,
    plan_market_windows,
)

PUBLISHED = datetime(2026, 8, 1, 8, 0, tzinfo=UTC)


def test_source_audit_covers_nine_issuers_without_separate_sberp_source() -> None:
    configs = official_source_configs()
    assert len(configs) == 9
    assert {ticker for item in configs for ticker in item.tickers} == {
        "SBER",
        "SBERP",
        "GAZP",
        "LKOH",
        "ROSN",
        "NVTK",
        "YDEX",
        "T",
        "VTBR",
        "GMKN",
    }


def test_lukoil_rss_is_located_but_not_accepted_without_observed_items() -> None:
    config = _config("AUDIT_LKOH_RSS")
    assert config.status == OfficialSourceStatus.UNSTABLE_SOURCE
    assert "Get RSS link" in config.timestamp_semantics
    assert config.timestamp_quality == PublicationTimestampQuality.UNKNOWN


def test_novatek_remains_nlp_only_date_only() -> None:
    config = _config("AUDIT_NVTK")
    assert config.status == OfficialSourceStatus.NLP_ONLY_DATE_ONLY
    assert config.timestamp_quality == PublicationTimestampQuality.DATE_ONLY


def test_yandex_is_reaction_ready_with_exact_offset_timestamp() -> None:
    config = _config("YANDEX_IR_PRESS_RELEASES_RSS")
    config.validate()
    assert config.status == OfficialSourceStatus.REACTION_READY
    assert config.timestamp_quality == PublicationTimestampQuality.EXACT
    assert config.feed_url == YANDEX_FEED_URL
    assert "+0300" in config.timestamp_semantics


def test_reaction_ready_gate_rejects_date_only_source() -> None:
    yandex = _config("YANDEX_IR_PRESS_RELEASES_RSS")
    invalid = replace(yandex, timestamp_quality=PublicationTimestampQuality.DATE_ONLY)
    with pytest.raises(ValueError, match="EXACT"):
        invalid.validate()


def test_reaction_ready_gate_requires_stable_https_feed() -> None:
    yandex = _config("YANDEX_IR_PRESS_RELEASES_RSS")
    with pytest.raises(ValueError, match="HTTPS"):
        replace(yandex, feed_url="http://ir.yandex.ru/press-releases/news.rss").validate()


def test_controlled_import_is_bounded_to_one_hundred() -> None:
    command = ControlledImport(
        "YANDEX_IR_PRESS_RELEASES_RSS", PUBLISHED, PUBLISHED + timedelta(days=1), 10
    )
    command.validate()
    replace(command, limit=100).validate()
    with pytest.raises(ValueError, match="between 1 and 100"):
        replace(command, limit=101).validate()


def test_controlled_import_has_no_rule_ai_or_return_filters() -> None:
    assert {item.name for item in fields(ControlledImport)} == {
        "source_code",
        "date_from",
        "date_to",
        "limit",
        "source_order",
    }


def test_yandex_uses_generic_rss_adapter_configuration() -> None:
    config = _config("YANDEX_IR_PRESS_RELEASES_RSS")
    assert config.source_kind == HistoricalNewsSourceKind.ISSUER_RSS
    assert config.storage_policy == ContentStoragePolicy.EXCERPT_ALLOWED
    assert not (Path(__file__).parents[2] / "src" / "historical_news" / "yandex.py").exists()


async def test_generic_rss_parser_uses_link_as_stable_identity_when_guid_is_absent() -> None:
    items = await _parsed_items()
    assert len(items) == 1
    assert items[0].source_item_id == "https://ir.yandex.ru/press-releases?id=one"
    assert items[0].source_url == items[0].source_item_id


async def test_generic_rss_parser_preserves_exact_timezone_semantics() -> None:
    item = (await _parsed_items())[0]
    parsed = parse_publication_timestamp(
        item.published_at_text, source_timezone=item.source_timezone
    )
    assert parsed.quality == PublicationTimestampQuality.EXACT
    assert parsed.published_at == datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
    assert item.original_timestamp_text.endswith("+0300")


async def test_generic_rss_parser_marks_description_as_excerpt() -> None:
    item = (await _parsed_items())[0]
    assert item.content_is_excerpt is True
    assert "issuer excerpt" in (item.content or "")


def test_market_windows_split_long_source_range_into_bounded_clusters() -> None:
    windows = plan_market_windows(
        [
            ("YDEX", PUBLISHED),
            ("YDEX", PUBLISHED + timedelta(days=5)),
            ("YDEX", PUBLISHED + timedelta(days=20)),
        ]
    )
    assert len(windows) == 3
    assert all(item.date_to - item.date_from <= timedelta(days=14) for item in windows)


def test_cumulative_real_funnel_counts_two_sources() -> None:
    records = [_record(1, "ROSN", "ROSNEFT_PRESS_RELEASES_RSS"), _record(2)]
    assert [item["count"] for item in cumulative_funnel(records)] == [2] * 9


def test_ticker_diversity_warning_clears_at_balanced_two_tickers() -> None:
    warnings = diversity_warnings(
        {"ROSN": 10, "YDEX": 20},
        {"UNKNOWN": 15, "OTHER": 15},
    )
    assert "LOW_TICKER_DIVERSITY" not in warnings


def test_unmatched_records_do_not_create_a_ticker_in_coverage() -> None:
    unmatched = replace(_record(3, ticker="UNMATCHED"), matched=False)
    assert represented_ticker_distribution([_record(1, "ROSN"), _record(2), unmatched]) == {
        "ROSN": 1,
        "YDEX": 1,
    }


def test_event_and_unknown_warnings_remain_independent() -> None:
    warnings = diversity_warnings(
        {"ROSN": 15, "YDEX": 15},
        {"UNKNOWN": 20, "OTHER": 10},
    )
    assert "HIGH_UNKNOWN_EVENT_RATE" in warnings
    assert "LOW_EVENT_DIVERSITY" not in warnings


def test_batch_003_selection_is_deterministic_and_draft() -> None:
    records = [_record(index) for index in range(30)]
    first = select_annotation_batch(
        records,
        batch_version=ANNOTATION_BATCH_VERSION,
        record_prefix="batch-003",
    )
    second = select_annotation_batch(
        list(reversed(records)),
        batch_version=ANNOTATION_BATCH_VERSION,
        record_prefix="batch-003",
    )
    assert first == second
    assert len(first) == 30
    assert all(item["annotation_status"] == "DRAFT" for item in first)
    assert all(item["assignment_status"] == "UNASSIGNED" for item in first)
    assert all(item["is_gold"] is False for item in first)


def test_batch_003_selection_contains_no_future_return_fields() -> None:
    row = select_annotation_batch(
        [_record(1)],
        batch_version=ANNOTATION_BATCH_VERSION,
        record_prefix="batch-003",
    )[0]
    assert not set(row).intersection({"return", "abnormal_return", "volume", "labels"})


def test_synthetic_and_seed_sources_are_not_real() -> None:
    assert classify_provenance("SYNTHETIC_TEST") == CorpusProvenance.SYNTHETIC
    assert classify_provenance("BATCH_001", "seed-dataset") == CorpusProvenance.SEED
    assert classify_provenance("YANDEX_IR_PRESS_RELEASES_RSS") == CorpusProvenance.REAL


def test_source_audit_payload_is_idempotent_and_selection_safe() -> None:
    first = audit_payload(official_source_configs())
    second = audit_payload(official_source_configs())
    assert first == second
    assert first["uses_rule_or_ai_output_for_selection"] is False
    assert first["uses_future_market_data_for_selection"] is False


def test_frozen_batch_002_checksum_is_unchanged() -> None:
    assert FROZEN_BATCH_002_SHA256 == (
        "358ea17184a6328283147e4c423db6d825147dfeed3add789a9bc2aef86c3159"
    )


def test_unit_tests_do_not_perform_live_http() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "MockTransport" in source
    assert "transport=httpx.MockTransport" in source


def test_exactly_two_sources_are_currently_reaction_ready() -> None:
    assert {item.source_code for item in reaction_ready_configs()} == {
        "ROSNEFT_PRESS_RELEASES_RSS",
        "YANDEX_IR_PRESS_RELEASES_RSS",
    }


def _config(source_code: str) -> OfficialSourceConfig:
    return next(item for item in official_source_configs() if item.source_code == source_code)


def _record(
    index: int,
    ticker: str = "YDEX",
    source: str = "YANDEX_IR_PRESS_RELEASES_RSS",
) -> PublicationTimeRecord:
    return PublicationTimeRecord(
        news_id=UUID(int=index + 1),
        ticker=ticker,
        source_code=source,
        source_item_id=f"item-{index}",
        source_url=f"https://ir.yandex.ru/press-releases?id={index}",
        title=f"Issuer release {index}",
        content="<p>Issuer-owned public RSS excerpt for deterministic review.</p>",
        published_at=PUBLISHED + timedelta(minutes=index),
        timestamp_quality=PublicationTimestampQuality.EXACT,
        storage_policy="EXCERPT_ALLOWED",
        content_is_excerpt=True,
        rules_primary_event="UNKNOWN",
        rules_event_count=0,
        rules_fact_count=0,
        analysis_status="NO_EVENT_FOUND",
        analysis_warnings=(),
        matched=True,
        market_data_ready=True,
        reaction_ready=True,
        feature_ready=True,
        valid_label_horizons=(1, 5, 15, 30, 60),
    )


def _rss_payload() -> bytes:
    return b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Yandex</title><item>
<title>Yandex release</title>
<link>https://ir.yandex.ru/press-releases?id=one</link>
<pubDate>Tue, 11 Aug 2026 11:00:00 +0300</pubDate>
<description><![CDATA[<p>issuer excerpt</p>]]></description>
</item></channel></rss>"""


async def _parsed_items() -> list[HistoricalSourceItem]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == YANDEX_FEED_URL
        return httpx.Response(200, content=_rss_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        source = IssuerFeedNewsSource(
            feed_url=YANDEX_FEED_URL,
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
            source_timezone=None,
            timeout_seconds=1,
            max_retries=0,
            max_items=10,
            user_agent="unit-test",
            client=client,
            sleep=False,
        )
        page = await source.fetch_items(
            from_datetime=datetime(2026, 8, 1, tzinfo=UTC),
            to_datetime=datetime(2026, 8, 12, tzinfo=UTC),
            cursor=None,
            limit=10,
        )
        return page.items
