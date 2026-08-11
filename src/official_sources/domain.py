from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.news.domain.enums import PublicationTimestampQuality

SOURCE_AUDIT_VERSION = "official-source-expansion-v1"
CORPUS_VERSION = "reaction-ready-corpus-v3"
ANNOTATION_BATCH_VERSION = "annotation-batch-003"


class OfficialSourceStatus(StrEnum):
    REACTION_READY = "REACTION_READY"
    NLP_ONLY_DATE_ONLY = "NLP_ONLY_DATE_ONLY"
    NLP_ONLY_UNKNOWN_TIME = "NLP_ONLY_UNKNOWN_TIME"
    ACCESS_BLOCKED = "ACCESS_BLOCKED"
    LICENSED_SOURCE_REQUIRED = "LICENSED_SOURCE_REQUIRED"
    UNSTABLE_SOURCE = "UNSTABLE_SOURCE"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class OfficialSourceConfig:
    source_code: str
    tickers: tuple[str, ...]
    issuer: str
    owner: str
    landing_url: str
    feed_url: str | None
    source_kind: HistoricalNewsSourceKind
    status: OfficialSourceStatus
    timestamp_quality: PublicationTimestampQuality
    timestamp_semantics: str
    stable_identity: str
    storage_policy: ContentStoragePolicy
    legal_public_access: str
    bounded_method: str
    historical_depth: str
    blocker: str | None = None

    def validate(self) -> None:
        if not self.source_code.strip() or not self.tickers:
            raise ValueError("official source requires code and tickers")
        landing = urlparse(self.landing_url)
        if landing.scheme != "https" or not landing.netloc:
            raise ValueError("official source landing URL must use HTTPS")
        required = (
            self.issuer,
            self.owner,
            self.timestamp_semantics,
            self.stable_identity,
            self.legal_public_access,
            self.bounded_method,
            self.historical_depth,
        )
        if any(not value.strip() for value in required):
            raise ValueError("official source evidence fields must not be empty")
        if self.status == OfficialSourceStatus.REACTION_READY:
            if self.feed_url is None:
                raise ValueError("reaction-ready source requires a feed URL")
            parsed_feed = urlparse(self.feed_url)
            if parsed_feed.scheme != "https" or not parsed_feed.netloc:
                raise ValueError("reaction-ready feed must use HTTPS")
            if self.timestamp_quality != PublicationTimestampQuality.EXACT:
                raise ValueError("reaction-ready source requires EXACT timestamps")
            if self.storage_policy in {
                ContentStoragePolicy.UNKNOWN,
                ContentStoragePolicy.METADATA_ONLY,
            }:
                raise ValueError("reaction-ready source requires a usable storage policy")
            if self.blocker is not None:
                raise ValueError("reaction-ready source cannot have a blocker")
        elif not (self.blocker or "").strip():
            raise ValueError("non-reaction-ready source requires a blocker")

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["tickers"] = list(self.tickers)
        payload["source_kind"] = self.source_kind.value
        payload["status"] = self.status.value
        payload["timestamp_quality"] = self.timestamp_quality.value
        payload["storage_policy"] = self.storage_policy.value
        return payload


@dataclass(frozen=True, slots=True)
class ControlledImport:
    source_code: str
    date_from: datetime
    date_to: datetime
    limit: int
    source_order: str = "feed order"

    def validate(self) -> None:
        if not self.source_code.strip():
            raise ValueError("source code is required")
        if self.date_to < self.date_from:
            raise ValueError("controlled import range is invalid")
        if not 1 <= self.limit <= 100:
            raise ValueError("controlled live import limit must be between 1 and 100")


def audit_payload(configs: tuple[OfficialSourceConfig, ...]) -> dict[str, Any]:
    for config in configs:
        config.validate()
    return {
        "schema_version": SOURCE_AUDIT_VERSION,
        "sources": [config.payload() for config in configs],
        "reaction_ready_source_codes": [
            config.source_code
            for config in configs
            if config.status == OfficialSourceStatus.REACTION_READY
        ],
        "selection_policy": "source + date range + issuer + source ordering + limit",
        "uses_rule_or_ai_output_for_selection": False,
        "uses_future_market_data_for_selection": False,
    }
