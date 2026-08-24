from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.exact_event_official_source_discovery.domain import (
    INPUT_DATASET_SHA as V5_INPUT_DATASET_SHA,
)
from src.exact_event_official_source_discovery.domain import (
    MAX_ITEMS_PER_SOURCE,
    MAX_PAGES_PER_SOURCE,
    MAX_REQUESTS_PER_DOMAIN,
    MAX_TICKERS,
    MAX_URLS_PER_TICKER,
    discovery_safety_flags,
)

ARTIFACT_VERSION = "exact-event-live-official-source-snapshot-v1"
INPUT_DATASET_SHA = V5_INPUT_DATASET_SHA
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_REDIRECTS = 3
MIN_DOMAIN_DELAY_SECONDS = 0.5
USER_AGENT = "raest4ik-trade-official-source-discovery/1.0"

SUPPORTED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/json",
        "application/ld+json",
    }
)

STANDARD_PATH_PROBES = (
    "/sitemap.xml",
    "/news",
    "/press",
    "/press-releases",
    "/media",
    "/investors",
    "/investor-relations",
    "/rss",
    "/feed",
)

NETWORK_LIMITS: dict[str, Any] = {
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "max_response_bytes": MAX_RESPONSE_BYTES,
    "max_redirects": MAX_REDIRECTS,
    "min_domain_delay_seconds": MIN_DOMAIN_DELAY_SECONDS,
    "max_tickers": MAX_TICKERS,
    "max_urls_per_ticker": MAX_URLS_PER_TICKER,
    "max_requests_per_domain": MAX_REQUESTS_PER_DOMAIN,
    "max_pages_per_source": MAX_PAGES_PER_SOURCE,
    "max_items_per_source": MAX_ITEMS_PER_SOURCE,
    "supported_content_types": sorted(SUPPORTED_CONTENT_TYPES),
}

CACHE_SCHEMA: dict[str, Any] = {
    "version": "exact-event-live-source-snapshot-cache-v1",
    "candidate_fields": [
        "source_url",
        "source_domain",
        "source_type",
        "source_family",
        "official_source_confirmed",
        "timestamp_capability",
        "timestamp_field",
        "timezone_provenance",
        "archive_capability",
        "discovery_method",
        "policy_status",
        "technical_status",
        "public_access",
        "auth_required",
        "captcha_required",
        "payment_required",
        "instrument_uid",
        "items",
    ],
}


class LiveBlocker(StrEnum):
    NO_OFFICIAL_DOMAIN = "NO_OFFICIAL_DOMAIN"
    OFFICIAL_DOMAIN_AMBIGUOUS = "OFFICIAL_DOMAIN_AMBIGUOUS"
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    CAPTCHA_BLOCKED = "CAPTCHA_BLOCKED"
    PAYMENT_REQUIRED = "PAYMENT_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    TLS_FAILED = "TLS_FAILED"
    DNS_FAILED = "DNS_FAILED"
    TIMEOUT = "TIMEOUT"
    HTTP_4XX = "HTTP_4XX"
    HTTP_5XX = "HTTP_5XX"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
    NO_DISCOVERY_LINKS = "NO_DISCOVERY_LINKS"
    NO_TIMESTAMP_SOURCE = "NO_TIMESTAMP_SOURCE"
    DATE_ONLY_SOURCE = "DATE_ONLY_SOURCE"
    EXACT_SOURCE_NO_ARCHIVE = "EXACT_SOURCE_NO_ARCHIVE"
    EXACT_SOURCE_READY = "EXACT_SOURCE_READY"
    TECHNICAL_FETCH_FAILED = "TECHNICAL_FETCH_FAILED"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"
    NETWORK_UNAVAILABLE = "NETWORK_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class NetworkProvenance:
    request_url: str
    final_url: str | None
    http_status: int | None
    content_type: str | None
    fetched_at: str
    content_sha256: str | None
    bytes_received: int
    robots_status: str
    policy_status: str
    blocker: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "REQUEST_URL": self.request_url,
            "FINAL_URL": self.final_url,
            "HTTP_STATUS": self.http_status,
            "CONTENT_TYPE": self.content_type,
            "FETCHED_AT": self.fetched_at,
            "CONTENT_SHA256": self.content_sha256,
            "BYTES_RECEIVED": self.bytes_received,
            "ROBOTS_STATUS": self.robots_status,
            "POLICY_STATUS": self.policy_status,
            "BLOCKER": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class SourceReport:
    ticker: str
    issuer: str
    source_url: str | None
    source_domain: str | None
    official_source_confirmed: bool
    official_identity_evidence: str | None
    source_type: str | None
    source_family: str | None
    discovery_method: str
    timestamp_capability: str
    timestamp_field: str | None
    timezone_provenance: str | None
    archive_capability: bool
    items_discovered: int
    exact_items_discovered: int
    date_only_items_discovered: int
    policy_status: str
    technical_status: str
    blocker: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "TICKER": self.ticker,
            "ISSUER": self.issuer,
            "SOURCE_URL": self.source_url,
            "SOURCE_DOMAIN": self.source_domain,
            "OFFICIAL_SOURCE_CONFIRMED": self.official_source_confirmed,
            "OFFICIAL_IDENTITY_EVIDENCE": self.official_identity_evidence,
            "SOURCE_TYPE": self.source_type,
            "SOURCE_FAMILY": self.source_family,
            "DISCOVERY_METHOD": self.discovery_method,
            "TIMESTAMP_CAPABILITY": self.timestamp_capability,
            "TIMESTAMP_FIELD": self.timestamp_field,
            "TIMEZONE_PROVENANCE": self.timezone_provenance,
            "ARCHIVE_CAPABILITY": self.archive_capability,
            "ITEMS_DISCOVERED": self.items_discovered,
            "EXACT_ITEMS_DISCOVERED": self.exact_items_discovered,
            "DATE_ONLY_ITEMS_DISCOVERED": self.date_only_items_discovered,
            "POLICY_STATUS": self.policy_status,
            "TECHNICAL_STATUS": self.technical_status,
            "BLOCKER": self.blocker,
        }


def live_safety_flags() -> dict[str, bool | int]:
    flags = discovery_safety_flags()
    flags["TINVEST_READONLY_USED"] = False
    return flags


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def counter_payload(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))
