from __future__ import annotations

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urljoin, urlparse
from uuid import UUID, uuid5

from src.exact_event_corpus.domain import SessionState
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.free_live_issuer_accumulation.domain import (
    SealedLiveEpochOutcomeReadError,
    guard_sealed_live_epoch_post_event_price_read,
    live_accumulation_safety_flags,
    parse_publication_timestamp,
    sha256_payload,
    sha256_text,
)
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS
from src.ml_features.application.point_in_time import (
    PointInTimeFeatureBuilder,
    PointInTimeMarketFeatures,
    PointInTimeViolationError,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch

ARTIFACT_VERSION = "free-live-issuer-source-expansion-v2"
SOURCE_REGISTRY_VERSION = "live-issuer-sources-v2"
LOOKBACK_MINUTES = 120
BENCHMARK_TICKER = "IMOEX"
HISTORICAL_ISSUER_TICKERS = ("GMKN", "MGNT", "ROSN", "T", "VKCO", "X5", "YDEX")
LIVE_READY_BASELINE_TICKERS = ("ROSN", "YDEX")
FEATURE_BLOCKERS = (
    "FEATURE_PIPELINE_NOT_WIRED",
    "INSTRUMENT_MAPPING_MISSING",
    "MARKET_DATA_UNAVAILABLE",
    "BENCHMARK_DATA_UNAVAILABLE",
    "INSUFFICIENT_PRE_EVENT_LOOKBACK",
    "OUTSIDE_SUPPORTED_SESSION",
    "SEALED_GUARD_TOO_BROAD",
    "FEATURE_CONTRACT_FAILURE",
    "ENVIRONMENT_UNAVAILABLE",
    "OTHER_EXPLICIT_BLOCKER",
)
_EVENT_NAMESPACE = UUID("4f72ca69-5449-469f-abda-b7b903fd4c95")
_HTML_LINK_RE = re.compile(
    r"<link\b(?=[^>]*\brel=[\"'][^\"']*\balternate\b[^\"']*[\"'])(?=[^>]*\btype=[\"']"
    r"(?P<type>application/(?:rss|atom)\+xml)[\"'])(?=[^>]*\bhref=[\"'](?P<href>[^\"']+)[\"'])",
    re.IGNORECASE,
)
_JSONLD_RE = re.compile(
    r"<script\b[^>]*type=[\"']application/ld\+json[\"'][^>]*>(?P<body>.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)


class FeatureReadinessBlocker(StrEnum):
    FEATURE_PIPELINE_NOT_WIRED = "FEATURE_PIPELINE_NOT_WIRED"
    INSTRUMENT_MAPPING_MISSING = "INSTRUMENT_MAPPING_MISSING"
    MARKET_DATA_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"
    BENCHMARK_DATA_UNAVAILABLE = "BENCHMARK_DATA_UNAVAILABLE"
    INSUFFICIENT_PRE_EVENT_LOOKBACK = "INSUFFICIENT_PRE_EVENT_LOOKBACK"
    OUTSIDE_SUPPORTED_SESSION = "OUTSIDE_SUPPORTED_SESSION"
    SEALED_GUARD_TOO_BROAD = "SEALED_GUARD_TOO_BROAD"
    FEATURE_CONTRACT_FAILURE = "FEATURE_CONTRACT_FAILURE"
    ENVIRONMENT_UNAVAILABLE = "ENVIRONMENT_UNAVAILABLE"
    OTHER_EXPLICIT_BLOCKER = "OTHER_EXPLICIT_BLOCKER"


class SourceStatus(StrEnum):
    SOURCE_CONTRACT_READY = "SOURCE_CONTRACT_READY"
    LIVE_STRICT_EXACT_READY = "LIVE_STRICT_EXACT_READY"
    LIVE_TIMESTAMP_UNVERIFIED = "LIVE_TIMESTAMP_UNVERIFIED"
    LIVE_DATE_ONLY = "LIVE_DATE_ONLY"
    LIVE_CLOCK_WITHOUT_TIMEZONE = "LIVE_CLOCK_WITHOUT_TIMEZONE"
    LIVE_TECHNICAL_BLOCKER = "LIVE_TECHNICAL_BLOCKER"
    LIVE_NO_STABLE_ID = "LIVE_NO_STABLE_ID"
    LIVE_NOT_ISSUER_ORIGINATED = "LIVE_NOT_ISSUER_ORIGINATED"
    OUT_OF_SCOPE_PAID_SOURCE = "OUT_OF_SCOPE_PAID_SOURCE"


class TimestampLevel(StrEnum):
    LEVEL_A = "LEVEL_A_EXPLICIT_OFFSET_OR_UTC"
    LEVEL_B = "LEVEL_B_DOCUMENTED_TIMEZONE"
    LEVEL_C = "LEVEL_C_CLOCK_WITHOUT_TIMEZONE"
    LEVEL_D = "LEVEL_D_DATE_ONLY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class InstrumentIdentity:
    ticker: str
    legal_issuer: str
    instrument_uid: str
    figi: str | None
    primary_board: str


@dataclass(frozen=True, slots=True)
class FeatureCandle:
    end_at: datetime
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class LiveFeatureMarketProvider:
    instrument: InstrumentIdentity
    benchmark_uid: str
    client: MinuteCandleClient

    async def security_candles(
        self, *, published_at: datetime, lookback_minutes: int
    ) -> TInvestMinuteCandleBatch:
        upper_bound = _last_complete_observation_cutoff(published_at)
        return await self.client.fetch_minute_candles_audited(
            instrument_uid=self.instrument.instrument_uid,
            date_from=upper_bound - timedelta(minutes=lookback_minutes),
            date_to=upper_bound,
        )

    async def benchmark_candles(
        self, *, published_at: datetime, lookback_minutes: int
    ) -> TInvestMinuteCandleBatch:
        upper_bound = _last_complete_observation_cutoff(published_at)
        return await self.client.fetch_minute_candles_audited(
            instrument_uid=self.benchmark_uid,
            date_from=upper_bound - timedelta(minutes=lookback_minutes),
            date_to=upper_bound,
        )


class MinuteCandleClient(Protocol):
    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


class FeatureMarketProviderFactory(Protocol):
    def for_event(self, ticker: str) -> LiveFeatureMarketProvider | None: ...


@dataclass(frozen=True, slots=True)
class StaticFeatureMarketProviderFactory:
    providers: dict[str, LiveFeatureMarketProvider]

    def for_event(self, ticker: str) -> LiveFeatureMarketProvider | None:
        return self.providers.get(ticker)


def provider_factory_from_tinvest_mapping(
    *,
    mapping_path: Path,
    client: MinuteCandleClient,
    tickers: tuple[str, ...] = LIVE_READY_BASELINE_TICKERS,
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> StaticFeatureMarketProviderFactory:
    mapping = _read_json(mapping_path)
    raw_instruments = mapping.get("instruments")
    if not isinstance(raw_instruments, list):
        raise ValueError("TINVEST_MAPPING_INSTRUMENTS_MISSING")
    instrument_rows = cast("list[object]", raw_instruments)
    instruments: dict[str, dict[str, Any]] = {}
    for raw_row in instrument_rows:
        if not isinstance(raw_row, dict):
            continue
        row = cast("dict[str, Any]", raw_row)
        ticker_value = row.get("ticker")
        if isinstance(ticker_value, str) and ticker_value:
            instruments[ticker_value.upper()] = row
    benchmark = instruments.get(benchmark_ticker)
    if benchmark is None:
        raise ValueError("TINVEST_MAPPING_BENCHMARK_MISSING")
    providers: dict[str, LiveFeatureMarketProvider] = {}
    for ticker in tickers:
        row = instruments.get(ticker)
        seed = next((item for item in SEED_INSTRUMENTS if item.ticker == ticker), None)
        if row is None or seed is None:
            continue
        identity = InstrumentIdentity(
            ticker=ticker,
            legal_issuer=seed.issuer_name,
            instrument_uid=str(row["instrument_uid"]),
            figi=_optional_string(row.get("figi")),
            primary_board=str(row.get("class_code") or seed.primary_board),
        )
        providers[ticker] = LiveFeatureMarketProvider(
            instrument=identity,
            benchmark_uid=str(benchmark["instrument_uid"]),
            client=client,
        )
    return StaticFeatureMarketProviderFactory(providers)


@dataclass(frozen=True, slots=True)
class SourceProbeConfig:
    source_id: str
    ticker: str
    legal_issuer: str
    official_domain: str
    url: str
    mechanism: str
    parser: str
    timestamp_field: str
    identity_field: str
    content_fields: tuple[str, ...]
    new_hypothesis: str
    issuer_originated: bool = True
    paid_source: bool = False
    enabled: bool = False
    document_timezone: str | None = None
    prior_rejection_source_id: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedSourceItem:
    source_item_id: str
    canonical_url: str
    published_at: datetime
    published_raw: str
    title: str
    content: str
    timestamp_level: TimestampLevel


@dataclass(frozen=True, slots=True)
class SourceProbeResult:
    config: SourceProbeConfig
    status: SourceStatus
    timestamp_level: TimestampLevel
    real_item_observed: bool
    items_observed: tuple[ParsedSourceItem, ...]
    blocker: str | None
    response: FetchResult | None
    request_attempts: int
    alternate_links: tuple[str, ...] = ()


def seed_identity(ticker: str) -> InstrumentIdentity | None:
    normalized = ticker.strip().upper()
    seed = next((item for item in SEED_INSTRUMENTS if item.ticker == normalized), None)
    if seed is None:
        return None
    return InstrumentIdentity(
        ticker=seed.ticker,
        legal_issuer=seed.issuer_name,
        instrument_uid=f"seed:{seed.ticker}",
        figi=None,
        primary_board=seed.primary_board,
    )


async def diagnose_shadow_event_feature_readiness(
    event: dict[str, Any],
    *,
    provider_factory: FeatureMarketProviderFactory | None,
    lookback_minutes: int = LOOKBACK_MINUTES,
    before_pipeline_not_wired: bool = True,
) -> dict[str, Any]:
    published = _parse_datetime(event["published_at"])
    ticker = str(event["ticker"]).upper()
    required_start = published - timedelta(minutes=lookback_minutes)
    diagnosis: dict[str, Any] = {
        "event_id": str(event["event_id"]),
        "ticker": ticker,
        "published_at": published.isoformat(),
        "source_id": str(event["source_id"]),
        "instrument_resolved": False,
        "instrument_uid": None,
        "figi": None,
        "market_session_status": "UNKNOWN",
        "pre_event_candle_availability": {
            "available": False,
            "rows": 0,
            "last_observation_end_at": None,
        },
        "benchmark_candle_availability": {
            "available": False,
            "rows": 0,
            "last_observation_end_at": None,
        },
        "required_lookback_start": required_start.isoformat(),
        "hard_upper_bound": published.isoformat(),
        "market_provider": None,
        "feature_builder_invoked": False,
        "feature_builder_result": None,
        "feature_ready_before": bool(
            event.get("pre_event_feature_availability", {}).get("available", False)
        ),
        "feature_ready_after": False,
        "max_market_timestamp_read": None,
        "timestamp_violation": False,
        "exact_blocker": FeatureReadinessBlocker.FEATURE_PIPELINE_NOT_WIRED.value
        if before_pipeline_not_wired
        else FeatureReadinessBlocker.OTHER_EXPLICIT_BLOCKER.value,
    }
    if provider_factory is None:
        return diagnosis
    provider = provider_factory.for_event(ticker)
    if provider is None:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.INSTRUMENT_MAPPING_MISSING.value
        return diagnosis

    diagnosis["instrument_resolved"] = True
    diagnosis["instrument_uid"] = provider.instrument.instrument_uid
    diagnosis["figi"] = provider.instrument.figi
    diagnosis["market_provider"] = "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES"
    try:
        guard_sealed_live_epoch_post_event_price_read(
            epoch="LIVE_SHADOW_CORPUS",
            published_at=published,
            query_end_at=published,
            context="pre_event_feature_readiness",
        )
    except SealedLiveEpochOutcomeReadError:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.SEALED_GUARD_TOO_BROAD.value
        return diagnosis

    try:
        security_batch = await provider.security_candles(
            published_at=published, lookback_minutes=lookback_minutes
        )
        benchmark_batch = await provider.benchmark_candles(
            published_at=published, lookback_minutes=lookback_minutes
        )
    except Exception as exc:
        diagnosis["feature_builder_result"] = type(exc).__name__
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.ENVIRONMENT_UNAVAILABLE.value
        return diagnosis

    if security_batch.rejected_reasons:
        diagnosis["feature_builder_result"] = security_batch.rejected_reasons[0]
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.MARKET_DATA_UNAVAILABLE.value
        return diagnosis
    if benchmark_batch.rejected_reasons:
        diagnosis["feature_builder_result"] = benchmark_batch.rejected_reasons[0]
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.BENCHMARK_DATA_UNAVAILABLE.value
        return diagnosis

    raw_max_timestamp = _max_end_at((*security_batch.candles, *benchmark_batch.candles))
    if raw_max_timestamp and raw_max_timestamp > published:
        diagnosis["max_market_timestamp_read"] = raw_max_timestamp.isoformat()
        diagnosis["timestamp_violation"] = True
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.FEATURE_CONTRACT_FAILURE.value
        return diagnosis

    security = _complete_pre_event_candles(security_batch.candles, published)
    benchmark = _complete_pre_event_candles(benchmark_batch.candles, published)
    diagnosis["pre_event_candle_availability"] = _availability_payload(security)
    diagnosis["benchmark_candle_availability"] = _availability_payload(benchmark)
    diagnosis["market_session_status"] = _classify_pre_event_session(
        published, security, benchmark
    ).value
    max_timestamp = _max_end_at((*security, *benchmark))
    diagnosis["max_market_timestamp_read"] = max_timestamp.isoformat() if max_timestamp else None
    if max_timestamp and max_timestamp > published:
        diagnosis["timestamp_violation"] = True
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.FEATURE_CONTRACT_FAILURE.value
        return diagnosis
    if not security:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.MARKET_DATA_UNAVAILABLE.value
        return diagnosis
    if not benchmark:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.BENCHMARK_DATA_UNAVAILABLE.value
        return diagnosis
    if diagnosis["market_session_status"] != SessionState.DURING_MAIN_SESSION.value:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.OUTSIDE_SUPPORTED_SESSION.value
        return diagnosis

    builder = PointInTimeFeatureBuilder()
    diagnosis["feature_builder_invoked"] = True
    try:
        security_features = builder.build(
            candles=_feature_candles(security), as_of=published, prefix="security"
        )
        benchmark_features = builder.build(
            candles=_feature_candles(benchmark), as_of=published, prefix="benchmark"
        )
    except PointInTimeViolationError as exc:
        diagnosis["feature_builder_result"] = str(exc)
        diagnosis["timestamp_violation"] = True
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.FEATURE_CONTRACT_FAILURE.value
        return diagnosis
    missing = tuple(sorted((*security_features.missing, *benchmark_features.missing)))
    diagnosis["feature_builder_result"] = {
        "security": _feature_payload(security_features),
        "benchmark": _feature_payload(benchmark_features),
        "relative_returns": _relative_returns(security_features, benchmark_features),
        "missing": list(missing),
    }
    if missing:
        diagnosis["exact_blocker"] = FeatureReadinessBlocker.INSUFFICIENT_PRE_EVENT_LOOKBACK.value
        return diagnosis
    diagnosis["feature_ready_after"] = True
    diagnosis["exact_blocker"] = None
    return diagnosis


def probe_candidate_source(
    config: SourceProbeConfig,
    *,
    client: HttpClient,
    fetched_at: datetime | None = None,
) -> SourceProbeResult:
    if config.paid_source:
        return SourceProbeResult(
            config,
            SourceStatus.OUT_OF_SCOPE_PAID_SOURCE,
            TimestampLevel.UNKNOWN,
            False,
            (),
            "PAID_SOURCE_FORBIDDEN",
            None,
            0,
        )
    if not config.issuer_originated:
        return SourceProbeResult(
            config,
            SourceStatus.LIVE_NOT_ISSUER_ORIGINATED,
            TimestampLevel.UNKNOWN,
            False,
            (),
            "NOT_ISSUER_ORIGINATED",
            None,
            0,
        )
    if not _is_official_first_party(config.url, config.official_domain):
        return SourceProbeResult(
            config,
            SourceStatus.LIVE_NOT_ISSUER_ORIGINATED,
            TimestampLevel.UNKNOWN,
            False,
            (),
            "UNOFFICIAL_ENDPOINT",
            None,
            0,
        )
    result, attempts = _fetch_with_bounded_retry(client, config.url)
    if result.blocker is not None or result.status is None or result.status >= 400:
        return SourceProbeResult(
            config,
            SourceStatus.LIVE_TECHNICAL_BLOCKER,
            TimestampLevel.UNKNOWN,
            False,
            (),
            result.blocker or "HTTP_FAILURE",
            result,
            attempts,
        )
    try:
        items, alternates = _parse_source_items(config, result.body, config.url)
    except ValueError as exc:
        blocker = str(exc)
        return SourceProbeResult(
            config,
            _status_for_blocker(blocker),
            _timestamp_level_for_blocker(blocker),
            False,
            (),
            blocker,
            result,
            attempts,
        )
    if not items:
        return SourceProbeResult(
            config,
            SourceStatus.LIVE_TECHNICAL_BLOCKER,
            TimestampLevel.UNKNOWN,
            False,
            (),
            "NO_REAL_ITEM_OBSERVED",
            result,
            attempts,
            tuple(alternates),
        )
    if not all(item.source_item_id for item in items):
        return SourceProbeResult(
            config,
            SourceStatus.LIVE_NO_STABLE_ID,
            TimestampLevel.UNKNOWN,
            False,
            (),
            "STABLE_IDENTITY_REQUIRED",
            result,
            attempts,
            tuple(alternates),
        )
    level = min((item.timestamp_level for item in items), key=_timestamp_level_rank)
    status = (
        SourceStatus.LIVE_STRICT_EXACT_READY
        if level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B}
        else SourceStatus.LIVE_TIMESTAMP_UNVERIFIED
    )
    return SourceProbeResult(
        config,
        status,
        level,
        True,
        items,
        None,
        result,
        attempts,
        tuple(alternates),
    )


def default_candidate_sources() -> tuple[SourceProbeConfig, ...]:
    return (
        SourceProbeConfig(
            source_id="GAZP_GAZPROM_PRESS_HTML_ALT_JSONLD_V2",
            ticker="GAZP",
            legal_issuer='ПАО "Газпром"',
            official_domain="www.gazprom.com",
            url="https://www.gazprom.com/press/",
            mechanism="official_html_alternate_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || discovered feed item timestamp",
            identity_field="jsonld.url || canonical url || feed guid",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Check HTML alternates and JSON-LD instead of prior visible clock field."
            ),
            prior_rejection_source_id="GAZP_GAZPROM_PRESS_CLOCK_WITHOUT_TZ_REJECTED_V1",
        ),
        SourceProbeConfig(
            source_id="LKOH_LUKOIL_PRESS_HTML_ALT_JSONLD_V2",
            ticker="LKOH",
            legal_issuer="ПАО ЛУКОЙЛ",
            official_domain="www.lukoil.com",
            url="https://www.lukoil.com/PressCenter/Pressreleases",
            mechanism="official_html_alternate_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || discovered feed item timestamp",
            identity_field="jsonld.url || canonical url || feed guid",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis="Check alternate machine-readable links and structured metadata.",
            prior_rejection_source_id="LKOH_LUKOIL_PRESS_RELEASES_LIVE_CANDIDATE_V1",
        ),
        SourceProbeConfig(
            source_id="NVTK_NOVATEK_PRESS_HTML_ALT_JSONLD_V2",
            ticker="NVTK",
            legal_issuer="ПАО НОВАТЭК",
            official_domain="www.novatek.ru",
            url="https://www.novatek.ru/en/press/releases/",
            mechanism="official_html_alternate_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || discovered feed item timestamp",
            identity_field="jsonld.url || canonical url || feed guid",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Check English official release page for structured metadata/feed links."
            ),
        ),
        SourceProbeConfig(
            source_id="SBER_SBERBANK_PRESS_HTML_ALT_JSONLD_V2",
            ticker="SBER",
            legal_issuer="ПАО Сбербанк",
            official_domain="www.sberbank.com",
            url="https://www.sberbank.com/news-and-media/press-releases",
            mechanism="official_html_alternate_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || discovered feed item timestamp",
            identity_field="jsonld.url || canonical url || feed guid",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis="Check first-party press release page structured metadata and feeds.",
        ),
        SourceProbeConfig(
            source_id="SBERP_SBERBANK_PRESS_SAME_ISSUER_CONTROL_V2",
            ticker="SBERP",
            legal_issuer="ПАО Сбербанк",
            official_domain="www.sberbank.com",
            url="https://www.sberbank.com/news-and-media/press-releases",
            mechanism="same_legal_issuer_share_class_control",
            parser="same-issuer-collapse-v1",
            timestamp_field="same as SBER candidate",
            identity_field="same as SBER candidate",
            content_fields=("headline", "description"),
            new_hypothesis="Control row proving SBER/SBERP must collapse to one legal issuer.",
        ),
        SourceProbeConfig(
            source_id="VTBR_VTB_PRESS_HTML_ALT_JSONLD_V2",
            ticker="VTBR",
            legal_issuer="Банк ВТБ",
            official_domain="www.vtb.com",
            url="https://www.vtb.com/o-banke/press-centr/novosti-i-press-relizy/",
            mechanism="official_html_alternate_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || discovered feed item timestamp",
            identity_field="jsonld.url || canonical url || feed guid",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Check first-party press center for alternate feed/JSON-LD timestamp evidence."
            ),
        ),
    )


async def run_free_live_issuer_source_expansion_v2(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    shadow_corpus_path: Path = Path(
        "artifacts/free-live-issuer-accumulation-v1/live-shadow-corpus.jsonl"
    ),
    provider_factory: FeatureMarketProviderFactory | None = None,
    client: HttpClient | None = None,
    network_check: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if await asyncio.to_thread(_has_existing_output, output_root):
        raise FileExistsError(
            "immutable free live issuer source expansion v2 output already exists"
        )
    now = created_at or datetime.now(UTC)
    shadow_rows = _read_jsonl(shadow_corpus_path)
    feature_diagnostics = [
        await diagnose_shadow_event_feature_readiness(row, provider_factory=provider_factory)
        for row in shadow_rows
    ]
    http = client or BoundedHttpClient(timeout_seconds=8.0, redirect_limit=3)
    candidates = default_candidate_sources()
    source_results = (
        [probe_candidate_source(config, client=http, fetched_at=now) for config in candidates]
        if network_check
        else []
    )
    accepted = [_accepted_source_payload(result) for result in source_results if _is_ready(result)]
    rejected = [
        _rejected_source_payload(result)
        for result in source_results
        if not _is_ready(result) and result.status != SourceStatus.OUT_OF_SCOPE_PAID_SOURCE
    ]
    source_probe_rows = [_source_probe_payload(result, now) for result in source_results]
    timestamp_rows = [_timestamp_evidence_payload(result) for result in source_results]
    raw_snapshots = _new_source_shadow_snapshots(source_results, now)
    semantic_ready_events = len(shadow_rows) + len(raw_snapshots)
    unknown_count = _unknown_count(shadow_rows)
    new_legal_issuers = sorted(
        {
            source["legal_issuer"]
            for source in accepted
            if source["ticker"] not in HISTORICAL_ISSUER_TICKERS
        }
    )
    blockers = Counter(
        str(row["exact_blocker"]) for row in feature_diagnostics if row["exact_blocker"]
    )
    feature_ready_before = sum(bool(row["feature_ready_before"]) for row in feature_diagnostics)
    feature_ready_after = sum(bool(row["feature_ready_after"]) for row in feature_diagnostics)
    max_market_timestamp = _max_iso(
        str(row["max_market_timestamp_read"])
        for row in feature_diagnostics
        if row["max_market_timestamp_read"]
    )
    timestamp_violations = sum(bool(row["timestamp_violation"]) for row in feature_diagnostics)
    diversity_status = _live_diversity_status(len(new_legal_issuers), len(source_results))
    flags = live_accumulation_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "SOURCE_REGISTRY_VERSION": SOURCE_REGISTRY_VERSION,
        "FREE_SOURCES_ONLY": True,
        "EXISTING_SHADOW_EVENTS": len(shadow_rows),
        "FEATURE_ATTEMPTS": len(feature_diagnostics),
        "PRE_EVENT_FEATURE_READY_EVENTS_BEFORE": feature_ready_before,
        "PRE_EVENT_FEATURE_READY_EVENTS_AFTER": feature_ready_after,
        "PRE_EVENT_FEATURE_READY_EVENTS": feature_ready_after,
        "FEATURE_BLOCKERS_BY_CATEGORY": dict(sorted(blockers.items())),
        "MAXIMUM_MARKET_TIMESTAMP_READ": max_market_timestamp,
        "PUBLICATION_TIMESTAMPS": [row["published_at"] for row in feature_diagnostics],
        "TIMESTAMP_VIOLATIONS": timestamp_violations,
        "CANDIDATES_AUDITED": len(source_results),
        "NEW_HYPOTHESES_TESTED": sum(bool(item.new_hypothesis) for item in candidates),
        "NEW_READY_SOURCES": len(accepted),
        "NEW_TICKER_SYMBOLS": sorted(
            {
                source["ticker"]
                for source in accepted
                if source["ticker"] not in HISTORICAL_ISSUER_TICKERS
            }
        ),
        "NEW_DISTINCT_LEGAL_ISSUERS": new_legal_issuers,
        "NEW_DISTINCT_LEGAL_ISSUER_COUNT": len(new_legal_issuers),
        "TOTAL_READY_ISSUERS": len(
            {*_issuer_names_for_tickers(LIVE_READY_BASELINE_TICKERS), *new_legal_issuers}
        ),
        "TIMEZONE_PASS": sum(
            result.timestamp_level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B}
            for result in source_results
        ),
        "TIMEZONE_FAIL": sum(
            result.timestamp_level
            in {TimestampLevel.LEVEL_C, TimestampLevel.LEVEL_D, TimestampLevel.UNKNOWN}
            for result in source_results
        ),
        "REAL_ITEMS_OBSERVED": sum(result.real_item_observed for result in source_results),
        "NEW_RAW_SNAPSHOTS": len(raw_snapshots),
        "SEMANTIC_READY_EVENTS": semantic_ready_events,
        "UNKNOWN_EVENTS": unknown_count,
        "UNKNOWN_RATE": _rate(unknown_count, len(shadow_rows)),
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "FINAL_DIVERSITY_STATUS": diversity_status,
        "FEATURE_PIPELINE": "YES" if feature_ready_after > 0 else "NO",
        "SOURCE_DIVERSITY": "YES" if len(new_legal_issuers) >= 3 else "NO",
        "EXACT_NEXT_ACTION": _next_action(feature_ready_after, len(new_legal_issuers)),
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "git_sha", "ARTIFACT_SHA"}
        }
    )
    await asyncio.to_thread(output_root.mkdir, parents=True, exist_ok=False)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "candidate-universe.json", _candidate_universe_payload(candidates))
    _write_jsonl(output_root / "source-probes.jsonl", source_probe_rows)
    _write_jsonl(output_root / "timestamp-evidence.jsonl", timestamp_rows)
    _write_json(output_root / "accepted-sources.json", {"sources": accepted})
    _write_jsonl(output_root / "rejected-sources.jsonl", rejected)
    _write_jsonl(output_root / "feature-readiness-diagnosis.jsonl", feature_diagnostics)
    _write_json(output_root / "feature-readiness-summary.json", _feature_summary(manifest))
    _write_jsonl(output_root / "raw-publication-snapshots.jsonl", raw_snapshots)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _complete_pre_event_candles(
    candles: Sequence[TInvestMinuteCandle], published: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    return tuple(
        sorted(
            (
                candle
                for candle in candles
                if candle.is_complete and candle.end_at <= published.astimezone(UTC)
            ),
            key=lambda item: item.end_at,
        )
    )


def _feature_candles(candles: Sequence[TInvestMinuteCandle]) -> tuple[FeatureCandle, ...]:
    return tuple(
        FeatureCandle(
            end_at=candle.end_at,
            close=candle.close,
            volume=Decimal(candle.volume),
        )
        for candle in candles
    )


def _last_complete_observation_cutoff(published_at: datetime) -> datetime:
    published = published_at.astimezone(UTC)
    if published.second == 0 and published.microsecond == 0:
        return published
    return published.replace(second=0, microsecond=0)


def _classify_pre_event_session(
    published_at: datetime,
    security: Sequence[TInvestMinuteCandle],
    benchmark: Sequence[TInvestMinuteCandle],
) -> SessionState:
    published = published_at.astimezone(UTC)
    common_ends = {
        row.end_at
        for row in security
        if row.end_at.date() == published.date()
        and any(other.end_at == row.end_at for other in benchmark)
    }
    if not common_ends:
        return SessionState.NON_TRADING_DAY
    last_common = max(common_ends)
    if last_common <= published and published - last_common <= timedelta(minutes=1):
        return SessionState.DURING_MAIN_SESSION
    if last_common < published:
        return SessionState.AFTER_CLOSE
    return SessionState.UNKNOWN


def _availability_payload(candles: Sequence[TInvestMinuteCandle]) -> dict[str, Any]:
    last = max((item.end_at for item in candles), default=None)
    return {
        "available": bool(candles),
        "rows": len(candles),
        "last_observation_end_at": last.isoformat() if last else None,
    }


def _feature_payload(features: PointInTimeMarketFeatures) -> dict[str, Any]:
    return {
        "returns": {str(key): _decimal_or_none(value) for key, value in features.returns.items()},
        "log_returns": {
            str(key): _decimal_or_none(value) for key, value in features.log_returns.items()
        },
        "realized_volatility": {
            str(key): _decimal_or_none(value) for key, value in features.realized_volatility.items()
        },
        "volume_last_1m": _decimal_or_none(features.volume_last_1m),
        "volume_sums": {
            str(key): _decimal_or_none(value) for key, value in features.volume_sums.items()
        },
        "volume_ratio_5m_vs_60m": _decimal_or_none(features.volume_ratio_5m_vs_60m),
        "last_observation_end_at": (
            features.last_observation_end_at.isoformat()
            if features.last_observation_end_at is not None
            else None
        ),
        "missing": list(features.missing),
    }


def _relative_returns(
    security: PointInTimeMarketFeatures, benchmark: PointInTimeMarketFeatures
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for horizon, security_return in security.returns.items():
        benchmark_return = benchmark.returns.get(horizon)
        values[str(horizon)] = (
            None
            if security_return is None or benchmark_return is None
            else str(security_return - benchmark_return)
        )
    return values


def _decimal_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _fetch_with_bounded_retry(client: HttpClient, url: str) -> tuple[FetchResult, int]:
    attempts = 0
    result: FetchResult | None = None
    for attempt in range(2):
        attempts += 1
        result = client.get(url)
        if result.status not in {429, 500, 502, 503, 504} and result.blocker not in {
            "TIMEOUT",
            "HTTP_FAILURE",
            "RATE_LIMITED",
            "TECHNICAL_FAILURE",
        }:
            break
        if attempt == 0:
            time.sleep(0.25)
    if result is None:
        raise RuntimeError("HTTP_CLIENT_RETURNED_NO_RESULT")
    return result, attempts


def _parse_source_items(
    config: SourceProbeConfig, body: bytes, base_url: str
) -> tuple[tuple[ParsedSourceItem, ...], tuple[str, ...]]:
    text = body.decode("utf-8", errors="replace")
    if config.parser.startswith("rss") or config.mechanism.endswith("rss"):
        return _parse_rss(text, config), ()
    if config.parser.startswith("atom") or config.mechanism.endswith("atom"):
        return _parse_atom(text, config), ()
    if config.parser.startswith("json"):
        return _parse_json_endpoint(text, config), ()
    alternates = discover_html_alternates(text, base_url)
    jsonld_items = _parse_jsonld(text, config)
    if jsonld_items:
        return jsonld_items, tuple(alternates)
    raise ValueError("NO_ACCEPTED_MACHINE_READABLE_PUBLICATION_TIMESTAMP")


def discover_html_alternates(html: str, base_url: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            urljoin(base_url, match.group("href").strip()) for match in _HTML_LINK_RE.finditer(html)
        )
    )


def _parse_rss(text: str, config: SourceProbeConfig) -> tuple[ParsedSourceItem, ...]:
    root = ET.fromstring(text)
    items = [item for item in root.iter() if _local(item.tag) == "item"]
    parsed: list[ParsedSourceItem] = []
    for item in items[:5]:
        raw_timestamp = _child_text(item, "pubDate")
        if not raw_timestamp:
            raise ValueError("MISSING_PUBLICATION_TIMESTAMP")
        published, level = _parse_timestamp(raw_timestamp, config)
        identity = _child_text(item, "guid") or _child_text(item, "link")
        if not identity:
            raise ValueError("STABLE_IDENTITY_REQUIRED")
        link = _child_text(item, "link") or identity
        parsed.append(
            ParsedSourceItem(
                source_item_id=identity,
                canonical_url=link,
                published_at=published,
                published_raw=raw_timestamp,
                title=_child_text(item, "title"),
                content=_child_text(item, "description"),
                timestamp_level=level,
            )
        )
    return tuple(parsed)


def _parse_atom(text: str, config: SourceProbeConfig) -> tuple[ParsedSourceItem, ...]:
    root = ET.fromstring(text)
    entries = [item for item in root.iter() if _local(item.tag) == "entry"]
    parsed: list[ParsedSourceItem] = []
    for entry in entries[:5]:
        raw_timestamp = _child_text(entry, "published")
        if not raw_timestamp:
            raise ValueError("MISSING_PUBLICATION_TIMESTAMP")
        published, level = _parse_timestamp(raw_timestamp, config)
        identity = _child_text(entry, "id") or _atom_link(entry)
        if not identity:
            raise ValueError("STABLE_IDENTITY_REQUIRED")
        parsed.append(
            ParsedSourceItem(
                source_item_id=identity,
                canonical_url=_atom_link(entry) or identity,
                published_at=published,
                published_raw=raw_timestamp,
                title=_child_text(entry, "title"),
                content=_child_text(entry, "summary") or _child_text(entry, "content"),
                timestamp_level=level,
            )
        )
    return tuple(parsed)


def _parse_json_endpoint(text: str, config: SourceProbeConfig) -> tuple[ParsedSourceItem, ...]:
    raw = cast("object", json.loads(text))
    rows: list[object] | None
    if isinstance(raw, list):
        rows = cast("list[object]", raw)
    elif isinstance(raw, dict):
        raw_dict = cast("dict[str, object]", raw)
        raw_items = raw_dict.get("items")
        rows = cast("list[object]", raw_items) if isinstance(raw_items, list) else None
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("JSON_ITEMS_MISSING")
    parsed: list[ParsedSourceItem] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        item = cast("dict[str, Any]", row)
        raw_timestamp = _nested_string(item, config.timestamp_field)
        if not raw_timestamp:
            raise ValueError("MISSING_PUBLICATION_TIMESTAMP")
        published, level = _parse_timestamp(raw_timestamp, config)
        identity = _nested_string(item, config.identity_field)
        if not identity:
            raise ValueError("STABLE_IDENTITY_REQUIRED")
        link = _nested_string(item, "url") or _nested_string(item, "link") or identity
        title = _nested_string(item, "title") or _nested_string(item, "headline")
        content = " ".join(
            value for field in config.content_fields if (value := _nested_string(item, field))
        )
        parsed.append(
            ParsedSourceItem(identity, link, published, raw_timestamp, title, content, level)
        )
    return tuple(parsed)


def _parse_jsonld(text: str, config: SourceProbeConfig) -> tuple[ParsedSourceItem, ...]:
    parsed: list[ParsedSourceItem] = []
    for match in _JSONLD_RE.finditer(text):
        raw_payload = json.loads(match.group("body").strip())
        for item in _jsonld_items(raw_payload):
            raw_timestamp = _nested_string(item, "datePublished")
            if not raw_timestamp:
                if _nested_string(item, "dateModified"):
                    raise ValueError("DATE_MODIFIED_CANNOT_SUBSTITUTE_PUBLICATION_TIME")
                continue
            published, level = _parse_timestamp(raw_timestamp, config)
            identity = _nested_string(item, "url") or _nested_string(item, "@id")
            if not identity:
                raise ValueError("STABLE_IDENTITY_REQUIRED")
            title = _nested_string(item, "headline") or _nested_string(item, "name")
            content = _nested_string(item, "articleBody") or _nested_string(item, "description")
            parsed.append(
                ParsedSourceItem(
                    identity, identity, published, raw_timestamp, title, content, level
                )
            )
    return tuple(parsed)


def _jsonld_items(payload: object) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, list):
        rows = cast("list[object]", payload)
        return tuple(item for row in rows for item in _jsonld_items(row))
    if not isinstance(payload, dict):
        return ()
    typed = cast("dict[str, Any]", payload)
    graph = typed.get("@graph")
    if isinstance(graph, list):
        graph_rows = cast("list[object]", graph)
        return (typed, *(item for row in graph_rows for item in _jsonld_items(row)))
    return (typed,)


def _parse_timestamp(
    raw_timestamp: str, config: SourceProbeConfig
) -> tuple[datetime, TimestampLevel]:
    stripped = raw_timestamp.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        raise ValueError("LIVE_DATE_ONLY")
    contract = None
    if config.document_timezone:
        contract = {
            "evidence_type": f"DOCUMENTED_TIMEZONE_{config.document_timezone.upper()}",
            "evidence_value": config.document_timezone,
        }
    try:
        published = parse_publication_timestamp(stripped, contract)
    except ValueError as exc:
        if str(exc) == "INVALID_TIMEZONE":
            raise ValueError("LIVE_CLOCK_WITHOUT_TIMEZONE") from exc
        raise
    level = (
        TimestampLevel.LEVEL_B
        if config.document_timezone and not _has_explicit_offset(stripped)
        else TimestampLevel.LEVEL_A
    )
    return published, level


def _has_explicit_offset(value: str) -> bool:
    return bool(re.search(r"(?:Z|[+-]\d{2}:?\d{2}|GMT|UTC|UT)\s*$", value, re.IGNORECASE))


def _nested_string(payload: dict[str, Any], path: str) -> str:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return ""
        current = cast("dict[str, object]", current).get(part)
    if isinstance(current, str):
        return " ".join(current.split())
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return str(current)
    return ""


def _status_for_blocker(blocker: str) -> SourceStatus:
    if blocker == "LIVE_DATE_ONLY":
        return SourceStatus.LIVE_DATE_ONLY
    if blocker == "LIVE_CLOCK_WITHOUT_TIMEZONE":
        return SourceStatus.LIVE_CLOCK_WITHOUT_TIMEZONE
    if blocker == "STABLE_IDENTITY_REQUIRED":
        return SourceStatus.LIVE_NO_STABLE_ID
    return SourceStatus.LIVE_TIMESTAMP_UNVERIFIED


def _timestamp_level_for_blocker(blocker: str) -> TimestampLevel:
    if blocker == "LIVE_DATE_ONLY":
        return TimestampLevel.LEVEL_D
    if blocker == "LIVE_CLOCK_WITHOUT_TIMEZONE":
        return TimestampLevel.LEVEL_C
    return TimestampLevel.UNKNOWN


def _is_ready(result: SourceProbeResult) -> bool:
    return result.status == SourceStatus.LIVE_STRICT_EXACT_READY and result.real_item_observed


def _is_official_first_party(url: str, official_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc.lower() == official_domain.lower()


def _timestamp_level_rank(level: TimestampLevel) -> int:
    return {
        TimestampLevel.LEVEL_A: 0,
        TimestampLevel.LEVEL_B: 1,
        TimestampLevel.LEVEL_C: 2,
        TimestampLevel.LEVEL_D: 3,
        TimestampLevel.UNKNOWN: 4,
    }[level]


def _source_probe_payload(result: SourceProbeResult, fetched_at: datetime) -> dict[str, Any]:
    response = result.response
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "official_domain": result.config.official_domain,
        "url": result.config.url,
        "mechanism": result.config.mechanism,
        "new_hypothesis": result.config.new_hypothesis,
        "prior_rejection_source_id": result.config.prior_rejection_source_id,
        "status": result.status.value,
        "blocker": result.blocker,
        "timestamp_level": result.timestamp_level.value,
        "real_item_observed": result.real_item_observed,
        "items_observed": len(result.items_observed),
        "request_attempts": result.request_attempts,
        "http_status": None if response is None else response.status,
        "content_type": None if response is None else response.content_type,
        "bytes_received": None if response is None else len(response.body),
        "final_url": None if response is None else response.final_url,
        "alternate_links": list(result.alternate_links),
        "fetched_at": fetched_at.isoformat(),
    }


def _timestamp_evidence_payload(result: SourceProbeResult) -> dict[str, Any]:
    first = result.items_observed[0] if result.items_observed else None
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "timestamp_field": result.config.timestamp_field,
        "timestamp_level": result.timestamp_level.value,
        "published_raw": None if first is None else first.published_raw,
        "published_at_utc": None if first is None else first.published_at.isoformat(),
        "timezone_pass": result.timestamp_level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B},
        "blocker": result.blocker,
        "evidence_sha": sha256_payload(
            {
                "source_id": result.config.source_id,
                "published_raw": None if first is None else first.published_raw,
                "timestamp_field": result.config.timestamp_field,
                "document_timezone": result.config.document_timezone,
            }
        ),
    }


def _accepted_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    config = result.config
    payload = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "source_id": config.source_id,
        "issuer_id": _issuer_id(config.legal_issuer),
        "ticker": config.ticker,
        "legal_issuer": config.legal_issuer,
        "domain": config.official_domain,
        "discovery_url": config.url,
        "mechanism": config.mechanism,
        "timestamp_field": config.timestamp_field,
        "timezone_evidence": result.timestamp_level.value,
        "identity_mechanism": config.identity_field,
        "parser_version": config.parser,
        "polling_limit": {"bounded_retries": 1, "max_items_per_poll": 5},
        "enabled": False,
        "real_item_observed": result.real_item_observed,
    }
    return payload | {"contract_sha": sha256_payload(payload)}


def _rejected_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "status": result.status.value,
        "blocker": result.blocker,
        "new_hypothesis": result.config.new_hypothesis,
        "paid_fallback_considered": False,
    }


def _new_source_shadow_snapshots(
    results: Sequence[SourceProbeResult], observed_at: datetime
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if not _is_ready(result):
            continue
        for item in result.items_observed[:2]:
            event_id = str(
                uuid5(_EVENT_NAMESPACE, f"{result.config.source_id}|{item.source_item_id}")
            )
            raw_material = "\n".join(part for part in (item.title, item.content) if part)
            payload = {
                "snapshot_version": "live-issuer-source-expansion-v2-raw-snapshot",
                "event_id": event_id,
                "source_id": result.config.source_id,
                "source_item_id": item.source_item_id,
                "canonical_url": item.canonical_url,
                "ticker": result.config.ticker,
                "legal_issuer": result.config.legal_issuer,
                "published_at": item.published_at.isoformat(),
                "publication_timestamp_raw": item.published_raw,
                "first_observed_at": observed_at.isoformat(),
                "timezone_evidence": item.timestamp_level.value,
                "raw_material_sha": sha256_text(raw_material),
                "semantic_output": {
                    "status": "SEMANTIC_READY_NOT_RULES_TUNED",
                    "semantic_unknown": True,
                    "fact_count": 0,
                    "rules_v3_changed": False,
                },
                "pre_event_feature_availability": {
                    "available": False,
                    "reason": "new source smoke did not read market data in source discovery phase",
                    "upper_bound": item.published_at.isoformat(),
                },
                "TARGET_STATUS": "SEALED",
            }
            rows.append(payload | {"raw_snapshot_sha": sha256_payload(payload)})
    return rows


def _candidate_universe_payload(candidates: Sequence[SourceProbeConfig]) -> dict[str, Any]:
    by_ticker = {item.ticker: item.legal_issuer for item in candidates}
    return {
        "historical_issuer_tickers": list(HISTORICAL_ISSUER_TICKERS),
        "baseline_live_ready_tickers": list(LIVE_READY_BASELINE_TICKERS),
        "candidate_count": len(candidates),
        "candidate_tickers": sorted(by_ticker),
        "candidate_legal_issuers": sorted(set(by_ticker.values())),
        "ranking_basis": [
            "free official first-party",
            "timestamp quality",
            "machine-readable feed",
            "stable identity",
            "source availability",
            "publication frequency",
            "deterministic ticker mapping",
            "liquidity/listing relevance",
        ],
        "share_class_collapse": {"SBER": "ПАО Сбербанк", "SBERP": "ПАО Сбербанк"},
    }


def _feature_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_shadow_events": manifest["EXISTING_SHADOW_EVENTS"],
        "feature_attempts": manifest["FEATURE_ATTEMPTS"],
        "feature_ready": manifest["PRE_EVENT_FEATURE_READY_EVENTS_AFTER"],
        "feature_rejected": manifest["FEATURE_ATTEMPTS"]
        - manifest["PRE_EVENT_FEATURE_READY_EVENTS_AFTER"],
        "blocker_distribution": manifest["FEATURE_BLOCKERS_BY_CATEGORY"],
        "maximum_market_timestamp_read": manifest["MAXIMUM_MARKET_TIMESTAMP_READ"],
        "publication_timestamps": manifest["PUBLICATION_TIMESTAMPS"],
        "timestamp_violations": manifest["TIMESTAMP_VIOLATIONS"],
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    metrics = [
        ("BASE_MAIN_SHA", manifest["BASE_MAIN_SHA"]),
        ("HEAD_SHA", manifest["git_sha"]),
        ("artifact SHA", manifest["ARTIFACT_SHA"]),
        ("existing shadow events", manifest["EXISTING_SHADOW_EVENTS"]),
        ("feature attempts", manifest["FEATURE_ATTEMPTS"]),
        (
            "pre-event-feature-ready events BEFORE",
            manifest["PRE_EVENT_FEATURE_READY_EVENTS_BEFORE"],
        ),
        ("pre-event-feature-ready events AFTER", manifest["PRE_EVENT_FEATURE_READY_EVENTS_AFTER"]),
        ("feature blockers by category", manifest["FEATURE_BLOCKERS_BY_CATEGORY"]),
        ("maximum market timestamp read", manifest["MAXIMUM_MARKET_TIMESTAMP_READ"]),
        ("post-event price reads", manifest["LIVE_POST_EVENT_PRICE_READS"]),
        ("candidates audited", manifest["CANDIDATES_AUDITED"]),
        ("new hypotheses tested", manifest["NEW_HYPOTHESES_TESTED"]),
        ("new READY sources", manifest["NEW_READY_SOURCES"]),
        ("NEW ticker symbols", manifest["NEW_TICKER_SYMBOLS"]),
        ("NEW distinct legal issuers", manifest["NEW_DISTINCT_LEGAL_ISSUERS"]),
        ("total READY issuers", manifest["TOTAL_READY_ISSUERS"]),
        ("timezone PASS", manifest["TIMEZONE_PASS"]),
        ("timezone FAIL", manifest["TIMEZONE_FAIL"]),
        ("real items observed", manifest["REAL_ITEMS_OBSERVED"]),
        ("new raw snapshots", manifest["NEW_RAW_SNAPSHOTS"]),
        ("semantic-ready events", manifest["SEMANTIC_READY_EVENTS"]),
        (
            "UNKNOWN count/rate",
            {"count": manifest["UNKNOWN_EVENTS"], "rate": manifest["UNKNOWN_RATE"]},
        ),
        ("paid sources used", manifest["PAID_SOURCES_USED"]),
        ("outcomes read", manifest["LIVE_OUTCOMES_READ"]),
        ("targets computed", manifest["LIVE_TARGETS_COMPUTED"]),
        ("Rules v3 changed", manifest["RULES_V3_CHANGED"]),
        ("model trained", manifest["MODEL_TRAINING_PERFORMED"]),
        ("final diversity status", manifest["FINAL_DIVERSITY_STATUS"]),
        ("exact next action", manifest["EXACT_NEXT_ACTION"]),
    ]
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        "",
        f"FEATURE_PIPELINE={manifest['FEATURE_PIPELINE']}",
        f"SOURCE_DIVERSITY={manifest['SOURCE_DIVERSITY']}",
        "",
        "## Metrics",
        "",
        *[
            f"{index}. {name}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for index, (name, value) in enumerate(metrics, start=1)
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _live_diversity_status(new_legal_issuer_count: int, audited: int) -> str:
    if new_legal_issuer_count >= 3:
        return "THREE_NEW_FREE_ISSUERS_READY"
    if new_legal_issuer_count > 0:
        return "PARTIAL_NEW_FREE_ISSUER_COVERAGE"
    if audited > 0:
        return "FREE_SOURCE_UNIVERSE_EXHAUSTED_FOR_NOW"
    return "NO_NEW_FREE_ISSUERS"


def _next_action(feature_ready_after: int, new_legal_issuers: int) -> str:
    if feature_ready_after == 0:
        return "Acquire bounded T-Invest pre-event minute candles for ROSN/YDEX live events."
    if new_legal_issuers < 3:
        return "Continue targeted first-party source discovery beyond current official candidates."
    return "Enable additive live-issuer-sources-v2 under bounded smoke ingestion."


def _unknown_count(rows: Sequence[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        semantic = row.get("semantic_output")
        if isinstance(semantic, dict) and bool(
            cast("dict[str, object]", semantic).get("semantic_unknown")
        ):
            total += 1
    return total


def _issuer_names_for_tickers(tickers: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for ticker in tickers:
        seed = next((item for item in SEED_INSTRUMENTS if item.ticker == ticker), None)
        names.add(seed.issuer_name if seed is not None else ticker)
    return names


def _issuer_id(legal_issuer: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", legal_issuer.upper()).strip("_") or "UNKNOWN"


def _max_end_at(candles: Sequence[TInvestMinuteCandle]) -> datetime | None:
    return max((item.end_at for item in candles), default=None)


def _max_iso(values: Iterable[str]) -> str | None:
    parsed = [_parse_datetime(value) for value in values if value and value != "None"]
    return max(parsed).isoformat() if parsed else None


def _rate(numerator: int, denominator: int) -> str:
    return "0.000000" if denominator == 0 else f"{numerator / denominator:.6f}"


def _child_text(item: ET.Element, tag: str) -> str:
    value = next(
        (child.text for child in item if _local(child.tag) == tag and child.text is not None),
        "",
    )
    return " ".join(value.split()) if value else ""


def _atom_link(item: ET.Element) -> str:
    for child in item:
        if _local(child.tag) == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _has_existing_output(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
