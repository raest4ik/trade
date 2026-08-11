from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.news.domain.enums import PublicationTimestampQuality

DATA_BUDGET = "ZERO"
MAX_PILOT_ITEMS = 200
MAX_CONCURRENCY = 2


class FreeSourceStatus(StrEnum):
    COMPLIANT_EXACT = "COMPLIANT_EXACT"
    COMPLIANT_DATE_ONLY = "COMPLIANT_DATE_ONLY"
    DISCOVERY_ONLY = "DISCOVERY_ONLY"
    UNSTABLE = "UNSTABLE"
    REJECTED_PAID = "REJECTED_PAID"
    REJECTED_POLICY = "REJECTED_POLICY"
    REJECTED_NO_TIMESTAMP = "REJECTED_NO_TIMESTAMP"


class VolumeEvidence(StrEnum):
    VERIFIED = "VERIFIED"
    ESTIMATED = "ESTIMATED"


class ModelReadiness(StrEnum):
    NOT_READY = "NOT_READY"
    PILOT_ONLY = "PILOT_ONLY"
    BASELINE_EXPERIMENT_READY = "BASELINE_EXPERIMENT_READY"
    BASELINE_TRAINING_READY = "BASELINE_TRAINING_READY"


@dataclass(frozen=True, slots=True)
class FreeSourceAudit:
    source_code: str
    issuer: str
    tickers: tuple[str, ...]
    source_name: str
    source_url: str
    source_kind: HistoricalNewsSourceKind
    status: FreeSourceStatus
    issuer_owned: bool
    official: bool
    free: bool
    automation_allowed: bool
    robots_status: str
    terms_status: str
    archive_available: bool
    pagination_available: bool
    sitemap_available: bool
    machine_readable: bool
    publication_datetime_available: bool
    timestamp_precision: PublicationTimestampQuality
    timezone_verified: bool
    earliest_verified_date: str | None
    latest_verified_date: str | None
    estimated_items: int | None
    full_text_available: bool
    storage_policy: ContentStoragePolicy
    stable_source_item_id: str
    recommendation: str
    blocking_reason: str | None

    def validate(self) -> None:
        required = (
            self.source_code,
            self.issuer,
            self.source_name,
            self.robots_status,
            self.terms_status,
            self.stable_source_item_id,
            self.recommendation,
        )
        if any(not value.strip() for value in required):
            raise ValueError("source audit fields must not be blank")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source audit URL must use HTTPS")
        if not self.tickers:
            raise ValueError("source audit requires at least one ticker")
        if self.status == FreeSourceStatus.REJECTED_PAID and self.free:
            raise ValueError("paid source cannot be marked free")
        if not self.free and self.status != FreeSourceStatus.REJECTED_PAID:
            raise ValueError("non-free source must be REJECTED_PAID")
        if self.status in {
            FreeSourceStatus.COMPLIANT_EXACT,
            FreeSourceStatus.COMPLIANT_DATE_ONLY,
        }:
            if not (self.official and self.free and self.automation_allowed):
                raise ValueError("compliant source must be official, free, and automation-safe")
            if self.storage_policy == ContentStoragePolicy.UNKNOWN:
                raise ValueError("compliant source requires a verified storage policy")
            if self.blocking_reason is not None:
                raise ValueError("compliant source cannot have a blocker")
        elif not (self.blocking_reason or "").strip():
            raise ValueError("non-compliant source requires a blocking reason")
        if self.status == FreeSourceStatus.COMPLIANT_EXACT:
            if self.timestamp_precision != PublicationTimestampQuality.EXACT:
                raise ValueError("COMPLIANT_EXACT requires source publication date and clock time")
            if not self.timezone_verified:
                raise ValueError("COMPLIANT_EXACT requires verified timezone semantics")
            if not self.publication_datetime_available:
                raise ValueError("COMPLIANT_EXACT requires source publication datetime")
        if self.status == FreeSourceStatus.COMPLIANT_DATE_ONLY:
            if self.timestamp_precision != PublicationTimestampQuality.DATE_ONLY:
                raise ValueError("COMPLIANT_DATE_ONLY requires DATE_ONLY precision")

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["tickers"] = list(self.tickers)
        payload["source_kind"] = self.source_kind.value
        payload["status"] = self.status.value
        payload["timestamp_precision"] = self.timestamp_precision.value
        payload["storage_policy"] = self.storage_policy.value
        return payload


@dataclass(frozen=True, slots=True)
class AcquisitionBounds:
    date_from: datetime
    date_to: datetime
    issuer: str
    limit: int
    dry_run: bool
    max_pages: int = 20
    concurrency: int = 1
    min_request_interval_seconds: float = 0.5

    def validate(self) -> None:
        if self.date_to < self.date_from:
            raise ValueError("date_to must not precede date_from")
        if not self.issuer.strip():
            raise ValueError("issuer is required")
        if not 1 <= self.limit <= MAX_PILOT_ITEMS:
            raise ValueError(f"pilot limit must be between 1 and {MAX_PILOT_ITEMS}")
        if not 1 <= self.max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        if not 1 <= self.concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
        if self.min_request_interval_seconds < 0.1:
            raise ValueError("request interval must be at least 0.1 seconds")


@dataclass(frozen=True, slots=True)
class SourceVolume:
    source_code: str
    evidence: VolumeEvidence
    available_items: int
    exact_items: int
    date_only_items: int
    date_from: str | None
    date_to: str | None
    issuers: tuple[str, ...]
    tickers: tuple[str, ...]
    eligible_for_acquisition: bool
    note: str

    def validate(self) -> None:
        if min(self.available_items, self.exact_items, self.date_only_items) < 0:
            raise ValueError("source volume counts cannot be negative")
        if self.exact_items + self.date_only_items > self.available_items:
            raise ValueError("timestamp buckets cannot exceed available items")
        if not self.source_code.strip() or not self.issuers or not self.tickers:
            raise ValueError("source volume identity is incomplete")
        if not self.note.strip():
            raise ValueError("source volume note is required")

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["evidence"] = self.evidence.value
        payload["issuers"] = list(self.issuers)
        payload["tickers"] = list(self.tickers)
        return payload


@dataclass(frozen=True, slots=True)
class DiscoveryIdentity:
    source_code: str
    source_item_id: str
    source_url: str
    published_at: datetime | None
    first_seen_at: datetime
    content_hash: str | None

    @property
    def stable_key(self) -> str:
        raw = f"{self.source_code.strip().upper()}\0{self.source_item_id.strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def validate(self) -> None:
        if not self.source_code.strip() or not self.source_item_id.strip():
            raise ValueError("stable source identity is required")
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("discovered source URL must use HTTPS")
        if self.published_at is not None and self.published_at == self.first_seen_at:
            raise ValueError("first_seen_at must not be substituted for publication time")


def readiness_for_feature_rows(
    feature_ready: int,
    *,
    ticker_count: int,
    source_count: int,
    month_count: int,
) -> dict[str, Any]:
    if feature_ready < 100:
        status = ModelReadiness.NOT_READY
    elif feature_ready < 500:
        status = ModelReadiness.PILOT_ONLY
    elif feature_ready < 1000:
        status = ModelReadiness.BASELINE_EXPERIMENT_READY
    else:
        status = ModelReadiness.BASELINE_TRAINING_READY
    blockers: list[str] = []
    if ticker_count < 3:
        blockers.append("INSUFFICIENT_TICKER_DIVERSITY")
    if source_count < 2:
        blockers.append("INSUFFICIENT_SOURCE_DIVERSITY")
    if month_count < 6:
        blockers.append("INSUFFICIENT_TIME_DIVERSITY")
    if blockers and status != ModelReadiness.NOT_READY:
        status = ModelReadiness.PILOT_ONLY
    return {
        "status": status.value,
        "feature_ready": feature_ready,
        "diversity_blockers": blockers,
        "market_regime_diversity": "UNKNOWN",
        "predictive_ml_trained": False,
    }


def source_volume_summary(volumes: tuple[SourceVolume, ...]) -> dict[str, Any]:
    for item in volumes:
        item.validate()
    verified = [item for item in volumes if item.evidence == VolumeEvidence.VERIFIED]
    estimated = [item for item in volumes if item.evidence == VolumeEvidence.ESTIMATED]
    eligible = [item for item in volumes if item.eligible_for_acquisition]
    return {
        "data_budget": DATA_BUDGET,
        "verified": {
            "available_items": sum(item.available_items for item in verified),
            "exact_items": sum(item.exact_items for item in verified),
            "date_only_items": sum(item.date_only_items for item in verified),
        },
        "estimated_additional": {
            "available_items": sum(item.available_items for item in estimated),
            "exact_items": sum(item.exact_items for item in estimated),
            "date_only_items": sum(item.date_only_items for item in estimated),
        },
        "eligible": {
            "available_items": sum(item.available_items for item in eligible),
            "exact_items": sum(item.exact_items for item in eligible),
        },
        "thresholds": [100, 500, 1000, 5000],
        "sources": [item.payload() for item in volumes],
    }


def select_pilot_candidates(
    rows: list[dict[str, Any]],
    *,
    existing_stable_keys: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= MAX_PILOT_ITEMS:
        raise ValueError(f"pilot limit must be between 1 and {MAX_PILOT_ITEMS}")
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_code = str(row.get("source_code", "")).strip().upper()
        source_item_id = str(row.get("source_item_id", "")).strip()
        published_at = str(row.get("published_at", "")).strip()
        if not source_code or not source_item_id or not published_at:
            raise ValueError("pilot candidate misses source identity or publication time")
        stable_key = hashlib.sha256(f"{source_code}\0{source_item_id}".encode()).hexdigest()
        if stable_key not in existing_stable_keys:
            unique.setdefault(stable_key, row)
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["source_code"]),
            str(row["published_at"]),
            str(row["source_item_id"]),
        ),
    )[:limit]
