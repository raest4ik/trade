from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.exact_event_official_source_discovery.domain import (
    INPUT_DATASET_SHA as V5_INPUT_DATASET_SHA,
)
from src.exact_event_official_source_discovery.domain import (
    discovery_safety_flags,
)

ARTIFACT_VERSION = "exact-event-official-domain-registry-enrichment-v1"
CREATED_BY = ARTIFACT_VERSION
REGISTRY_VERSION = "exact-event-official-domain-registry-v1"
INPUT_DATASET_SHA = V5_INPUT_DATASET_SHA

MAX_TICKERS = 50
MAX_SEARCH_QUERIES_PER_TICKER = 2
MAX_CANDIDATE_DOMAINS_PER_TICKER = 5
MAX_VALIDATION_URLS_PER_DOMAIN = 5
MAX_REQUESTS_PER_DOMAIN = 10
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_REDIRECTS = 3
MIN_DOMAIN_DELAY_SECONDS = 0.5

DOMAIN_DISCOVERY_LIMITS: dict[str, Any] = {
    "max_tickers": MAX_TICKERS,
    "max_search_queries_per_ticker": MAX_SEARCH_QUERIES_PER_TICKER,
    "max_candidate_domains_per_ticker": MAX_CANDIDATE_DOMAINS_PER_TICKER,
    "max_validation_urls_per_domain": MAX_VALIDATION_URLS_PER_DOMAIN,
    "max_requests_per_domain": MAX_REQUESTS_PER_DOMAIN,
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "max_response_bytes": MAX_RESPONSE_BYTES,
    "max_redirects": MAX_REDIRECTS,
    "min_domain_delay_seconds": MIN_DOMAIN_DELAY_SECONDS,
}

EVIDENCE_SCHEMA: dict[str, Any] = {
    "version": "exact-event-official-domain-evidence-v1",
    "required_fields": [
        "TICKER",
        "ISSUER",
        "CANDIDATE_DOMAIN",
        "CONFIRMED_HOST",
        "REGISTERED_DOMAIN",
        "DISCOVERY_ORIGIN",
        "EVIDENCE_TYPE",
        "EVIDENCE_URL",
        "EVIDENCE_CONTENT_SHA256",
        "LEGAL_NAME_EXPECTED",
        "LEGAL_NAME_OBSERVED",
        "IDENTIFIER_MATCH",
        "OFFICIAL_DOMAIN_CONFIRMED",
        "AMBIGUITY_REASON",
        "HTTP_STATUS",
        "FINAL_URL",
        "BLOCKER",
    ],
}


class DomainBlocker(StrEnum):
    DOMAIN_CONFIRMED = "DOMAIN_CONFIRMED"
    NO_CANDIDATE_DOMAIN = "NO_CANDIDATE_DOMAIN"
    OFFICIAL_DOMAIN_AMBIGUOUS = "OFFICIAL_DOMAIN_AMBIGUOUS"
    LEGAL_ENTITY_MISMATCH = "LEGAL_ENTITY_MISMATCH"
    PARENT_SUBSIDIARY_AMBIGUITY = "PARENT_SUBSIDIARY_AMBIGUITY"
    NO_IDENTITY_EVIDENCE = "NO_IDENTITY_EVIDENCE"
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    CAPTCHA_BLOCKED = "CAPTCHA_BLOCKED"
    PAYMENT_REQUIRED = "PAYMENT_REQUIRED"
    TLS_FAILED = "TLS_FAILED"
    DNS_FAILED = "DNS_FAILED"
    TIMEOUT = "TIMEOUT"
    HTTP_4XX = "HTTP_4XX"
    HTTP_5XX = "HTTP_5XX"
    RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    TECHNICAL_FETCH_FAILED = "TECHNICAL_FETCH_FAILED"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"
    NETWORK_UNAVAILABLE = "NETWORK_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class DomainEvidenceRecord:
    ticker: str
    issuer: str
    instrument_uid: str | None
    candidate_domain: str | None
    confirmed_host: str | None
    registered_domain: str | None
    discovery_origin: str
    evidence_type: str | None
    evidence_url: str | None
    evidence_content_sha256: str | None
    legal_name_expected: str
    legal_name_observed: str | None
    identifier_match: str | None
    official_domain_confirmed: bool
    ambiguity_reason: str | None
    http_status: int | None
    final_url: str | None
    blocker: str

    def payload(self) -> dict[str, Any]:
        return {
            "TICKER": self.ticker,
            "ISSUER": self.issuer,
            "INSTRUMENT_UID": self.instrument_uid,
            "CANDIDATE_DOMAIN": self.candidate_domain,
            "CONFIRMED_HOST": self.confirmed_host,
            "REGISTERED_DOMAIN": self.registered_domain,
            "DISCOVERY_ORIGIN": self.discovery_origin,
            "EVIDENCE_TYPE": self.evidence_type,
            "EVIDENCE_URL": self.evidence_url,
            "EVIDENCE_CONTENT_SHA256": self.evidence_content_sha256,
            "LEGAL_NAME_EXPECTED": self.legal_name_expected,
            "LEGAL_NAME_OBSERVED": self.legal_name_observed,
            "IDENTIFIER_MATCH": self.identifier_match,
            "OFFICIAL_DOMAIN_CONFIRMED": self.official_domain_confirmed,
            "AMBIGUITY_REASON": self.ambiguity_reason,
            "HTTP_STATUS": self.http_status,
            "FINAL_URL": self.final_url,
            "BLOCKER": self.blocker,
        }


def domain_safety_flags() -> dict[str, bool | int]:
    flags = discovery_safety_flags()
    flags["TINVEST_READONLY_USED"] = False
    return flags


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
