from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid5

from src.events.domain.enums import EventType
from src.news.domain.enums import PublicationTimestampQuality

DATASET_VERSION = "event-market-predictive-dataset-v2"
SOURCE_REGISTRY_VERSION = "event-source-registry-v2"
MARKET_CONTEXT_VERSION = "event-market-context-v1"
PREDICTIVE_UNIT = "EVENT"
EVENT_RULES_VERSION = "event-rules-v3"
FACTS_VERSION = "financial-facts-v3"
QWEN_PROMPT_SHA = "1ec9cd98ca2de73736d06f7dcb405a10966551d28a6d30e7b281f840a246b192"
QWEN_SCHEMA_SHA = "e97ff8a1141c6b178c53f756f5c3664fe7e377d88403534ae8829d3106d3796b"
REACTION_EXACT = "EXACT_INTRADAY"
REACTION_DAILY = "DATE_SAFE_DAILY"
FLAT_RETURN_THRESHOLD = Decimal("0.002")
_EVENT_NAMESPACE = UUID("7a8e1054-6309-4bbf-9c30-ad316f9ae150")


class SourceRegistryStatus(StrEnum):
    SOURCE_READY = "SOURCE_READY"
    SOURCE_DISCOVERED = "SOURCE_DISCOVERED"
    SOURCE_IMPLEMENTATION_REQUIRED = "SOURCE_IMPLEMENTATION_REQUIRED"
    NO_OFFICIAL_NEWS_ARCHIVE = "NO_OFFICIAL_NEWS_ARCHIVE"
    BLOCKED_BY_ACCESS = "BLOCKED_BY_ACCESS"
    BLOCKED_BY_SOURCE_POLICY = "BLOCKED_BY_SOURCE_POLICY"
    UNSUPPORTED_STRUCTURE = "UNSUPPORTED_STRUCTURE"
    NO_HISTORICAL_ARCHIVE = "NO_HISTORICAL_ARCHIVE"

    DISCOVERED_NOT_IMPLEMENTED = "SOURCE_IMPLEMENTATION_REQUIRED"
    NO_OFFICIAL_SOURCE_FOUND = "NO_OFFICIAL_NEWS_ARCHIVE"
    BLOCKED_BY_TECHNICAL_ACCESS = "BLOCKED_BY_ACCESS"


@dataclass(frozen=True, slots=True)
class EventSourceRegistryEntry:
    ticker: str
    issuer_name: str
    instrument_uid: str
    figi: str | None
    official_source_url: str | None
    source_name: str | None
    source_type: str | None
    official_owner: str | None
    collection_method: str | None
    history_available: str
    historical_range: str | None
    live_supported: bool
    status: SourceRegistryStatus
    reason: str
    public_access: bool
    payment_required: bool
    authentication_required: bool
    robots_rate_limit_notes: str
    redistribution_status: str
    internal_research_use_status: str
    first_seen: str
    last_checked: str
    exact_timestamp_available: bool = False
    date_only_available: bool = False
    rss_available: bool = False
    api_available: bool = False
    incremental_collection_supported: bool = False
    timestamp_semantics: str = "UNVERIFIED"
    stable_item_identity: str = "UNVERIFIED"
    collector_family: str | None = None
    parser_version: str | None = None
    rights_status: str = "UNKNOWN_FAIL_CLOSED"

    def payload(self) -> dict[str, object]:
        if not self.ticker or not self.instrument_uid or not self.issuer_name:
            raise ValueError("source registry identity is incomplete")
        if self.official_source_url is not None and not self.official_source_url.startswith(
            "https://"
        ):
            raise ValueError("official source URL must use HTTPS")
        if self.payment_required or self.authentication_required:
            if self.status == SourceRegistryStatus.SOURCE_READY:
                raise ValueError("paid or authenticated source cannot be ready")
        return {
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "ticker": self.ticker,
            "issuer_name": self.issuer_name,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "official_source_url": self.official_source_url,
            "official_domain": (
                urlsplit(self.official_source_url).netloc.lower()
                if self.official_source_url is not None
                else None
            ),
            "source_url": self.official_source_url,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "official_owner": self.official_owner,
            "collection_method": self.collection_method,
            "history_available": self.history_available,
            "historical_range": self.historical_range,
            "live_supported": self.live_supported,
            "status": self.status.value,
            "collector_status": self.status.value,
            "reason": self.reason,
            "public_access": self.public_access,
            "payment_required": self.payment_required,
            "authentication_required": self.authentication_required,
            "robots_rate_limit_notes": self.robots_rate_limit_notes,
            "redistribution_status": self.redistribution_status,
            "internal_research_use_status": self.internal_research_use_status,
            "first_seen": self.first_seen,
            "last_checked": self.last_checked,
            "exact_timestamp_available": self.exact_timestamp_available,
            "date_only_available": self.date_only_available,
            "rss_available": self.rss_available,
            "api_available": self.api_available,
            "incremental_collection_supported": self.incremental_collection_supported,
            "timestamp_semantics": self.timestamp_semantics,
            "stable_item_identity": self.stable_item_identity,
            "collector_family": self.collector_family,
            "parser_version": self.parser_version,
            "rights_status": self.rights_status,
        }


@dataclass(frozen=True, slots=True)
class AcquiredEvent:
    event_id: UUID
    source_code: str
    source_item_id: str
    canonical_url: str
    ticker: str
    issuer_name: str
    instrument_uid: str
    figi: str | None
    title: str
    publication_date: date
    published_at: datetime | None
    timestamp_quality: PublicationTimestampQuality
    storage_policy: str
    source_rights_status: str
    title_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_code: str,
        source_item_id: str,
        source_url: str,
        ticker: str,
        issuer_name: str,
        instrument_uid: str,
        figi: str | None,
        title: str,
        publication_date: date,
        published_at: datetime | None,
        timestamp_quality: PublicationTimestampQuality,
        storage_policy: str = "METADATA_TITLE_HASH_ONLY",
        source_rights_status: str = "PRIVATE_INTERNAL_RESEARCH_ONLY",
    ) -> AcquiredEvent:
        canonical = canonical_url(source_url)
        normalized_title = " ".join(title.split())
        if not normalized_title or not source_item_id or not ticker:
            raise ValueError("event identity fields are required")
        if timestamp_quality == PublicationTimestampQuality.EXACT:
            if published_at is None or published_at.tzinfo is None:
                raise ValueError("EXACT event requires an aware timestamp")
        elif published_at is not None:
            raise ValueError("non-EXACT event must not carry an imputed timestamp")
        identity = f"{source_code}|{source_item_id}"
        return cls(
            event_id=uuid5(_EVENT_NAMESPACE, identity),
            source_code=source_code,
            source_item_id=source_item_id,
            canonical_url=canonical,
            ticker=ticker,
            issuer_name=issuer_name,
            instrument_uid=instrument_uid,
            figi=figi,
            title=normalized_title,
            publication_date=publication_date,
            published_at=published_at,
            timestamp_quality=timestamp_quality,
            storage_policy=storage_policy,
            source_rights_status=source_rights_status,
            title_hash=sha256_text(normalized_title),
        )

    def metadata_payload(self) -> dict[str, object]:
        return {
            "event_id": str(self.event_id),
            "source_code": self.source_code,
            "source_item_id": self.source_item_id,
            "canonical_url": self.canonical_url,
            "ticker": self.ticker,
            "issuer_name": self.issuer_name,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "publication_date": self.publication_date.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "publication_time_quality": self.timestamp_quality.value,
            "storage_policy": self.storage_policy,
            "source_rights_status": self.source_rights_status,
            "title_hash": self.title_hash,
        }


@dataclass(frozen=True, slots=True)
class EventMarketRow:
    event: AcquiredEvent
    reaction_family: str
    market_context_cutoff: datetime
    event_features: dict[str, object]
    market_features: dict[str, float]
    semantics: dict[str, object]

    def feature_payload(self) -> dict[str, object]:
        if self.market_context_cutoff.tzinfo is None:
            raise ValueError("market context cutoff must be timezone-aware")
        if self.event.timestamp_quality == PublicationTimestampQuality.EXACT:
            if (
                self.event.published_at is None
                or not self.market_context_cutoff < self.event.published_at
            ):
                raise ValueError("EXACT market context must precede publication timestamp")
        elif not self.market_context_cutoff.date() < self.event.publication_date:
            raise ValueError("DATE_ONLY market context must precede publication date")
        return {
            "metadata": {
                **self.event.metadata_payload(),
                "dataset_version": DATASET_VERSION,
                "predictive_unit": PREDICTIVE_UNIT,
                "market_only_daily_rows_as_event_examples": False,
                "reaction_family": self.reaction_family,
                "market_context_version": MARKET_CONTEXT_VERSION,
                "market_context_cutoff": self.market_context_cutoff.isoformat(),
                "event_rules_version": EVENT_RULES_VERSION,
                "financial_facts_version": FACTS_VERSION,
            },
            "event_features": self.event_features,
            "market_features": self.market_features,
            "event_semantics": self.semantics,
            "quality": {
                "event_available_at_cutoff": True,
                "market_context_available_at_cutoff": True,
                "post_event_values_in_features": False,
            },
        }


def event_feature_names() -> tuple[str, ...]:
    return (
        "primary_event_type",
        "event_count",
        "fact_count",
        *(f"event_type_{item.value.lower()}" for item in EventType),
    )


def deduplicate_events(
    events: list[AcquiredEvent],
    *,
    existing_events: list[AcquiredEvent] | None = None,
) -> tuple[list[AcquiredEvent], list[dict[str, object]]]:
    selected: list[AcquiredEvent] = []
    dropped: list[dict[str, object]] = []
    existing = existing_events or []
    exact_keys = {(item.source_code, item.source_item_id): item for item in existing}
    canonical_urls = {item.canonical_url: item for item in existing}
    story_keys = {(item.ticker, item.publication_date, item.title_hash) for item in existing}
    for event in sorted(
        events,
        key=lambda item: (
            item.publication_date,
            item.ticker,
            0 if item.timestamp_quality == PublicationTimestampQuality.EXACT else 1,
            item.source_code,
            item.source_item_id,
        ),
    ):
        exact_key = (event.source_code, event.source_item_id)
        story_key = (event.ticker, event.publication_date, event.title_hash)
        if exact_key in exact_keys:
            previous = exact_keys[exact_key]
            reason = (
                "DUPLICATE_SOURCE_RECORD"
                if previous.title_hash == event.title_hash
                else "UPDATED_PUBLICATION"
            )
            dropped.append(_drop(event, reason))
            continue
        if event.canonical_url in canonical_urls:
            previous = canonical_urls[event.canonical_url]
            reason = (
                "DUPLICATE_CANONICAL_URL"
                if previous.title_hash == event.title_hash
                else "UPDATED_PUBLICATION"
            )
            dropped.append(_drop(event, reason))
            continue
        if story_key in story_keys:
            dropped.append(_drop(event, "SAME_EVENT_REPEATED"))
            continue
        exact_keys[exact_key] = event
        canonical_urls[event.canonical_url] = event
        story_keys.add(story_key)
        selected.append(event)
    return selected, dropped


def classify_reaction(value: Decimal) -> str:
    if value > FLAT_RETURN_THRESHOLD:
        return "UP"
    if value < -FLAT_RETURN_THRESHOLD:
        return "DOWN"
    return "FLAT"


def readiness(feature_ready: int, unique_tickers: int) -> dict[str, str]:
    if feature_ready < 100:
        status = "EVENT_DATA_NOT_READY"
    elif feature_ready < 250:
        status = "EVENT_PILOT_READY"
    elif feature_ready < 500:
        status = "EVENT_BASELINE_EXPERIMENT_READY"
    else:
        status = "EVENT_BASELINE_TRAINING_READY"
    if unique_tickers < 5:
        diversity = "VERY_LOW_EVENT_DIVERSITY"
    elif unique_tickers < 10:
        diversity = "LOW_EVENT_DIVERSITY"
    elif unique_tickers < 25:
        diversity = "EVENT_DIVERSITY_PILOT_READY"
    elif unique_tickers < 50:
        diversity = "EVENT_DIVERSITY_EXPERIMENT_READY"
    else:
        diversity = "EVENT_DIVERSITY_BROAD"
    model_status = (
        "READY_FOR_BASELINE_EXPERIMENT"
        if feature_ready >= 500 and unique_tickers >= 10
        else "NOT_READY"
    )
    return {
        "event_data_readiness": status,
        "event_volume_status": status,
        "ticker_diversity_status": diversity,
        "event_diversity_status": diversity,
        "event_model_data_status": model_status,
        "trading_readiness": "NOT_TRADING_READY",
    }


def require_unambiguous_ticker(matches: tuple[str, ...]) -> str:
    normalized = tuple(sorted({item.strip().upper() for item in matches if item.strip()}))
    if len(normalized) != 1:
        raise ValueError("EVENT_TICKER_STATUS=AMBIGUOUS")
    return normalized[0]


def canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("canonical event URL must be public HTTPS")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(("https", parsed.netloc.lower(), path, query, ""))


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def _drop(event: AcquiredEvent, reason: str) -> dict[str, object]:
    return {
        "event_id": str(event.event_id),
        "source_code": event.source_code,
        "source_item_id": event.source_item_id,
        "ticker": event.ticker,
        "publication_date": event.publication_date.isoformat(),
        "reason": reason,
    }
