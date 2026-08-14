from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid5

DATASET_VERSION = "exact-event-market-dataset-v1"
SOURCE_REGISTRY_VERSION = "exact-event-source-registry-v1"
PARSER_VERSION = "official-app-state-exact-v1"
MARKET_ALIGNMENT_VERSION = "tinvest-exact-minute-alignment-v1"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
FUTURE_EVENT_HOLDOUT_STATUS = "ACCUMULATING"
_EVENT_NAMESPACE = UUID("6a1cd446-ef89-487a-944d-d9c73ccab165")
_CLUSTER_NAMESPACE = UUID("4ad29c5b-3bcd-4c1d-af29-832db67d7600")


class TimestampCapability(StrEnum):
    EXACT = "EXACT"
    DATE_ONLY = "DATE_ONLY"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class TimezoneSemantics(StrEnum):
    EXPLICIT = "EXPLICIT"
    SOURCE_DEFAULT_DOCUMENTED = "SOURCE_DEFAULT_DOCUMENTED"
    UNKNOWN = "UNKNOWN"


class SessionState(StrEnum):
    PRE_OPEN = "PRE_OPEN"
    DURING_MAIN_SESSION = "DURING_MAIN_SESSION"
    AFTER_CLOSE = "AFTER_CLOSE"
    NON_TRADING_DAY = "NON_TRADING_DAY"
    UNKNOWN = "OTHER/UNKNOWN"


@dataclass(frozen=True, slots=True)
class ExactSourceRegistryEntry:
    ticker: str
    issuer: str
    instrument_uid: str
    official_domain: str | None
    source_url: str | None
    source_family: str | None
    parser_version: str | None
    timestamp_capability: TimestampCapability
    timestamp_field_source: str | None
    timezone_semantics: TimezoneSemantics
    historical_archive_start: str | None
    historical_archive_end: str | None
    incremental_supported: bool
    public_access: bool
    payment_required: bool
    auth_required: bool
    source_policy_status: str
    collector_status: str
    reason: str

    def payload(self) -> dict[str, object]:
        if self.source_url is not None:
            parsed = urlsplit(self.source_url)
            if parsed.scheme != "https" or parsed.netloc.lower() != self.official_domain:
                raise ValueError("source registry requires matching official HTTPS domain")
        if self.timestamp_capability == TimestampCapability.EXACT:
            if (
                not self.timestamp_field_source
                or self.timezone_semantics == TimezoneSemantics.UNKNOWN
            ):
                raise ValueError("EXACT source requires timestamp field and timezone semantics")
        if self.payment_required or self.auth_required:
            if self.collector_status == "SOURCE_READY":
                raise ValueError("paid or authenticated news source cannot be ready")
        return {
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "ticker": self.ticker,
            "issuer": self.issuer,
            "instrument_uid": self.instrument_uid,
            "official_domain": self.official_domain,
            "source_url": self.source_url,
            "source_family": self.source_family,
            "parser_version": self.parser_version,
            "timestamp_capability": self.timestamp_capability.value,
            "timestamp_field_source": self.timestamp_field_source,
            "timezone_semantics": self.timezone_semantics.value,
            "historical_archive_start": self.historical_archive_start,
            "historical_archive_end": self.historical_archive_end,
            "incremental_supported": self.incremental_supported,
            "public_access": self.public_access,
            "payment_required": self.payment_required,
            "auth_required": self.auth_required,
            "source_policy_status": self.source_policy_status,
            "collector_status": self.collector_status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ExactEvent:
    event_id: UUID
    source_code: str
    source_item_id: str
    canonical_url: str
    ticker: str
    issuer: str
    instrument_uid: str
    title: str
    publication_timestamp_raw: str
    publication_timestamp_utc: datetime
    timestamp_source_field: str
    timestamp_quality: str = "EXACT"
    publication_timezone: str = "UTC"
    provenance: str = "REAL"
    storage_policy: str = "METADATA_TITLE_HASH_ONLY"

    @classmethod
    def create(
        cls,
        *,
        source_code: str,
        source_item_id: str,
        canonical_url: str,
        ticker: str,
        issuer: str,
        instrument_uid: str,
        title: str,
        publication_timestamp_raw: str,
        publication_timestamp_utc: datetime,
        timestamp_source_field: str,
        storage_policy: str = "METADATA_TITLE_HASH_ONLY",
        event_id: UUID | None = None,
    ) -> ExactEvent:
        if publication_timestamp_utc.tzinfo is None:
            raise ValueError("EXACT requires real timezone-aware source time")
        published = publication_timestamp_utc.astimezone(UTC)
        if not publication_timestamp_raw.strip() or not timestamp_source_field.strip():
            raise ValueError("EXACT requires raw timestamp provenance")
        canonical = canonicalize_url(canonical_url)
        normalized_title = " ".join(title.split())
        if not normalized_title:
            raise ValueError("event title must not be empty")
        identity = f"{source_code}|{source_item_id}"
        return cls(
            event_id=event_id or uuid5(_EVENT_NAMESPACE, identity),
            source_code=source_code,
            source_item_id=source_item_id,
            canonical_url=canonical,
            ticker=ticker,
            issuer=issuer,
            instrument_uid=instrument_uid,
            title=normalized_title,
            publication_timestamp_raw=publication_timestamp_raw,
            publication_timestamp_utc=published,
            timestamp_source_field=timestamp_source_field,
            storage_policy=storage_policy,
        )

    @property
    def publication_date(self) -> date:
        return self.publication_timestamp_utc.date()

    @property
    def title_hash(self) -> str:
        return sha256_text(self.title)

    def metadata_payload(self) -> dict[str, object]:
        return {
            "event_id": str(self.event_id),
            "source_code": self.source_code,
            "source_item_id": self.source_item_id,
            "canonical_url": self.canonical_url,
            "ticker": self.ticker,
            "issuer": self.issuer,
            "instrument_uid": self.instrument_uid,
            "publication_timestamp_raw": self.publication_timestamp_raw,
            "publication_date": self.publication_date.isoformat(),
            "publication_time": self.publication_timestamp_utc.time().isoformat(),
            "publication_timezone": self.publication_timezone,
            "publication_timestamp_utc": self.publication_timestamp_utc.isoformat(),
            "timestamp_source_field": self.timestamp_source_field,
            "timestamp_quality": self.timestamp_quality,
            "provenance": self.provenance,
            "storage_policy": self.storage_policy,
            "title_hash": self.title_hash,
        }


def deterministic_clusters(events: list[ExactEvent]) -> dict[UUID, UUID]:
    ordered = sorted(
        events, key=lambda item: (item.ticker, item.publication_timestamp_utc, str(item.event_id))
    )
    assignments: dict[UUID, UUID] = {}
    anchors: list[ExactEvent] = []
    for event in ordered:
        story = _story_key(event)
        anchor = next(
            (
                item
                for item in reversed(anchors)
                if item.ticker == event.ticker
                and event.publication_timestamp_utc - item.publication_timestamp_utc
                <= timedelta(minutes=15)
                and _story_key(item) == story
            ),
            None,
        )
        cluster_seed = str(anchor.event_id if anchor is not None else event.event_id)
        assignments[event.event_id] = uuid5(_CLUSTER_NAMESPACE, cluster_seed)
        if anchor is None:
            anchors.append(event)
    return assignments


def exact_readiness(feature_ready: int, unique_tickers: int) -> tuple[str, str]:
    if feature_ready < 50:
        status = "EXACT_DATA_NOT_READY"
    elif feature_ready < 100:
        status = "EXACT_PILOT_READY"
    elif feature_ready < 250:
        status = "EXACT_BASELINE_EXPERIMENT_READY"
    else:
        status = "EXACT_BASELINE_TRAINING_READY"
    diversity = "EXACT_LOW_TICKER_DIVERSITY" if unique_tickers < 10 else "EXACT_DIVERSITY_OK"
    return status, diversity


def canonicalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("event URL must be public HTTPS")
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
    return urlunsplit(("https", parsed.netloc.lower(), path, parsed.query, ""))


def sha256_payload(payload: object) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _story_key(event: ExactEvent) -> str:
    title = re.sub(r"\W+", " ", event.title.casefold()).strip()
    return sha256_text(f"{event.canonical_url}|{title}")
