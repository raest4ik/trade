from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, cast
from urllib.parse import urlparse

ARTIFACT_VERSION = "free-live-issuer-accumulation-v1"
SHADOW_CORPUS_VERSION = "live-issuer-shadow-corpus-v1"
SOURCE_REGISTRY_VERSION = "live-issuer-sources-v1"
PARSER_VERSION = "rss-item-pubdate-explicit-offset-v1"
RAW_SNAPSHOT_VERSION = "live-issuer-raw-publication-snapshot-v1"
DEFAULT_SOURCE_REGISTRY_PATH = "config/live_issuer_sources_v1.json"
FUTURE_HOLDOUT_START = datetime(2026, 8, 11, tzinfo=UTC)
EXPECTED_RULES_V3_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
MAX_SOURCES_PER_SMOKE = 5
MAX_ITEMS_PER_SOURCE = 5


class SourceQualificationStatus(StrEnum):
    LIVE_STRICT_EXACT_READY = "LIVE_STRICT_EXACT_READY"
    LIVE_TIMESTAMP_UNVERIFIED = "LIVE_TIMESTAMP_UNVERIFIED"
    LIVE_DATE_ONLY = "LIVE_DATE_ONLY"
    LIVE_CLOCK_WITHOUT_TIMEZONE = "LIVE_CLOCK_WITHOUT_TIMEZONE"
    LIVE_TECHNICAL_BLOCKER = "LIVE_TECHNICAL_BLOCKER"
    LIVE_NO_STABLE_ID = "LIVE_NO_STABLE_ID"
    LIVE_NOT_ISSUER_ORIGINATED = "LIVE_NOT_ISSUER_ORIGINATED"
    OUT_OF_SCOPE_PAID_SOURCE = "OUT_OF_SCOPE_PAID_SOURCE"
    LIVE_READY_FOR_IMPLEMENTATION = "LIVE_READY_FOR_IMPLEMENTATION"


class LiveEventStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    TIMESTAMP_VERIFIED = "TIMESTAMP_VERIFIED"
    TICKER_RESOLVED = "TICKER_RESOLVED"
    RAW_SNAPSHOT_FROZEN = "RAW_SNAPSHOT_FROZEN"
    SEMANTIC_READY = "SEMANTIC_READY"
    PRE_EVENT_FEATURE_READY = "PRE_EVENT_FEATURE_READY"
    SHADOW_READY = "SHADOW_READY"
    REJECTED = "REJECTED"


class SealedLiveEpochOutcomeReadError(RuntimeError):
    """Raised when sealed live shadow data is used for outcomes or targets."""


class PointInTimeFeatureBoundError(ValueError):
    """Raised when a pre-event feature attempts to use future observations."""


class LiveModelPredictionError(RuntimeError):
    """Raised when sealed live shadow data is used for model predictions."""


@dataclass(frozen=True, slots=True)
class LiveIssuerSource:
    source_id: str
    ticker: str
    issuer: str
    canonical_domain: str
    discovery_url: str
    discovery_type: str
    parser: str
    timestamp_contract: dict[str, Any]
    timestamp_path: str
    identity_path: str
    content_path: tuple[str, ...]
    enabled: bool
    polling_policy: dict[str, Any]
    source_version: int
    source_status: SourceQualificationStatus
    source_origin: str
    stable_identity: str
    expected_publication_frequency: str
    ticker_binding: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LiveIssuerSource:
        status = SourceQualificationStatus(str(payload["source_status"]))
        content_path = payload.get("content_path")
        if not isinstance(content_path, list):
            raise ValueError("CONTENT_PATH_MISSING")
        content_path_rows = cast("list[object]", content_path)
        canonical_domain = str(payload.get("canonical_domain") or payload["official_domain"])
        parser = str(payload.get("parser") or payload["parser_type"])
        source = cls(
            source_id=str(payload["source_id"]),
            ticker=str(payload["ticker"]),
            issuer=str(payload["issuer"]),
            canonical_domain=canonical_domain,
            discovery_url=str(payload["discovery_url"]),
            discovery_type=str(payload["discovery_type"]),
            parser=parser,
            timestamp_contract=cast("dict[str, Any]", payload["timestamp_contract"]),
            timestamp_path=str(payload["timestamp_path"]),
            identity_path=str(payload["identity_path"]),
            content_path=tuple(str(item) for item in content_path_rows),
            enabled=bool(payload["enabled"]),
            polling_policy=cast("dict[str, Any]", payload["polling_policy"]),
            source_version=int(payload["source_version"]),
            source_status=status,
            source_origin=str(payload["source_origin"]),
            stable_identity=str(payload["stable_identity"]),
            expected_publication_frequency=str(payload["expected_publication_frequency"]),
            ticker_binding=cast("dict[str, Any]", payload["ticker_binding"]),
        )
        source.validate()
        return source

    def validate(self) -> None:
        parsed = urlparse(self.discovery_url)
        if parsed.scheme != "https":
            raise ValueError("SOURCE_URL_MUST_BE_HTTPS")
        if parsed.netloc != self.canonical_domain:
            raise ValueError("SOURCE_URL_DOMAIN_MISMATCH")
        if self.enabled and self.source_status != SourceQualificationStatus.LIVE_STRICT_EXACT_READY:
            raise ValueError("ENABLED_SOURCE_MUST_BE_LIVE_STRICT_EXACT_READY")
        if self.enabled and "dateModified" in self.timestamp_path:
            raise ValueError("DATE_MODIFIED_CANNOT_BE_PUBLICATION_TIMESTAMP")
        if self.enabled and self.source_origin != "ISSUER_ORIGINATED":
            raise ValueError("ENABLED_SOURCE_MUST_BE_ISSUER_ORIGINATED")
        if self.enabled and self.ticker == "MULTI":
            raise ValueError("ENABLED_SOURCE_REQUIRES_DETERMINISTIC_TICKER")
        evidence_type = str(self.timestamp_contract.get("evidence_type") or "")
        evidence_value = str(self.timestamp_contract.get("evidence_value") or "")
        if self.enabled and not (
            "EXPLICIT_OFFSET" in evidence_type
            or "DOCUMENTED_TIMEZONE" in evidence_type
            or re.search(r"(?:[+-]\d{2}:?\d{2}|\bZ\b|UTC)", evidence_value)
        ):
            raise ValueError("ENABLED_SOURCE_REQUIRES_EXPLICIT_TIMEZONE_CONTRACT")
        if (
            self.source_status == SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
            and self.enabled
        ):
            raise ValueError("PAID_SOURCE_CANNOT_BE_ENABLED")

    def payload(self) -> dict[str, Any]:
        return {
            "canonical_domain": self.canonical_domain,
            "contract_sha": self.contract_sha(),
            "content_path": list(self.content_path),
            "discovery_type": self.discovery_type,
            "discovery_url": self.discovery_url,
            "enabled": self.enabled,
            "expected_publication_frequency": self.expected_publication_frequency,
            "identity_path": self.identity_path,
            "issuer": self.issuer,
            "official_domain": self.canonical_domain,
            "parser": self.parser,
            "parser_type": self.parser,
            "polling_policy": self.polling_policy,
            "source_id": self.source_id,
            "source_origin": self.source_origin,
            "source_status": self.source_status.value,
            "source_version": self.source_version,
            "stable_identity": self.stable_identity,
            "ticker": self.ticker,
            "ticker_binding": self.ticker_binding,
            "timestamp_contract": self.timestamp_contract,
            "timestamp_path": self.timestamp_path,
        }

    def contract_sha(self) -> str:
        contract = {
            "content_path": list(self.content_path),
            "discovery_type": self.discovery_type,
            "discovery_url": self.discovery_url,
            "identity_path": self.identity_path,
            "parser": self.parser,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "timestamp_contract": self.timestamp_contract,
            "timestamp_path": self.timestamp_path,
        }
        return sha256_payload(contract)


@dataclass(frozen=True, slots=True)
class ParsedPublication:
    source_item_id: str
    canonical_url: str
    title: str
    description: str
    content: str
    publication_timestamp_raw: str
    publication_timestamp_utc: datetime
    raw_payload: dict[str, Any]
    raw_item: str

    def material(self) -> str | None:
        parts = [self.title.strip(), self.description.strip(), self.content.strip()]
        unique = [part for index, part in enumerate(parts) if part and part not in parts[:index]]
        return "\n".join(unique) if unique else None


def parse_publication_timestamp(
    value: str, timestamp_contract: dict[str, Any] | None = None
) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("MISSING_EXACT_TIMESTAMP")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        raise ValueError("MISSING_EXACT_TIMESTAMP")
    if _has_iso_offset(raw):
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        if not re.search(r"(?:[+-]\d{4}|GMT|UTC|UT)\s*$", raw, re.IGNORECASE):
            if _contract_documents_utc(timestamp_contract):
                parsed = datetime.fromisoformat(raw).replace(tzinfo=UTC)
            else:
                raise ValueError("INVALID_TIMEZONE")
        else:
            parsed = parsedate_to_datetime(raw)
    if parsed.tzinfo is None:
        raise ValueError("INVALID_TIMEZONE")
    return parsed.astimezone(UTC)


def assert_pre_event_feature_upper_bound(
    *, feature_timestamp: datetime, published_at: datetime
) -> None:
    if feature_timestamp > published_at:
        raise PointInTimeFeatureBoundError("PRE_EVENT_FEATURE_TIMESTAMP_AFTER_PUBLICATION")


def assert_market_query_upper_bound(*, end_at: datetime, published_at: datetime) -> None:
    if end_at > published_at:
        raise PointInTimeFeatureBoundError("PRE_EVENT_MARKET_QUERY_END_AFTER_PUBLICATION")


def guard_sealed_live_epoch_outcome_read(
    *,
    epoch: str,
    target_status: str,
    context: str,
) -> None:
    if epoch == "LIVE_SHADOW_CORPUS" and target_status == "SEALED":
        raise SealedLiveEpochOutcomeReadError(f"SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT:{context}")


def guard_sealed_live_epoch_post_event_price_read(
    *,
    epoch: str,
    published_at: datetime,
    query_end_at: datetime,
    context: str,
) -> None:
    if epoch == "LIVE_SHADOW_CORPUS" and query_end_at > published_at:
        raise SealedLiveEpochOutcomeReadError(f"SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT:{context}")


def guard_sealed_live_epoch_model_prediction(
    *,
    epoch: str,
    target_status: str,
    context: str,
) -> None:
    if epoch == "LIVE_SHADOW_CORPUS" and target_status == "SEALED":
        raise LiveModelPredictionError(f"SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT:{context}")


def live_accumulation_safety_flags() -> dict[str, Any]:
    return {
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_API_CALLS": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "MODEL_PREDICTIONS_PERFORMED": False,
        "LIVE_MODEL_PREDICTIONS": 0,
        "RULES_V3_CHANGED": False,
        "OLD_BASELINE_TEST_USED_FOR_TUNING": False,
        "LIVE_OUTCOMES_READ": 0,
        "LIVE_TARGETS_COMPUTED": 0,
        "LIVE_POST_EVENT_PRICE_READS": 0,
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "REAL_TRADING_ALLOWED": False,
        "BROKER_MUTATION_ALLOWED": False,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _has_iso_offset(value: str) -> bool:
    return bool(re.search(r"(?:[+-]\d{2}:\d{2}|Z)$", value))


def _contract_documents_utc(timestamp_contract: dict[str, Any] | None) -> bool:
    if timestamp_contract is None:
        return False
    evidence_type = str(timestamp_contract.get("evidence_type") or "")
    evidence_value = str(timestamp_contract.get("evidence_value") or "")
    return "DOCUMENTED_TIMEZONE_UTC" in evidence_type or "UTC" in evidence_value.upper()
