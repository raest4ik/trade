from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, cast

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START

ARTIFACT_VERSION = "exact-event-live-official-collection-v1"
SOURCE_REGISTRY_VERSION = "exact-event-live-official-source-registry-v1"
PARSER_VERSION = "live-publication-exact-parser-v2"
RAW_PUBLICATION_SNAPSHOT_VERSION = "raw-publication-snapshot-v1"
SEMANTIC_MATERIAL_PROVENANCE_VERSION = "semantic-material-provenance-v1"
DEFAULT_SOURCE_REGISTRY_PATH = "config/exact-event-live-official-sources.json"

MAX_SOURCES = 20
MAX_ITEMS_PER_SOURCE = 50
REQUEST_TIMEOUT_SECONDS = 10.0
RETRY_COUNT = 1
REDIRECT_LIMIT = 3
MAX_RESPONSE_BYTES = 2_000_000
SUPPORTED_MECHANISMS = frozenset({"RSS", "ATOM", "JSON", "HTML"})
PUBLICATION_TIMESTAMP_FIELDS = frozenset(
    {
        "RSS item pubDate",
        "Atom entry published",
        "JSON published_at",
        "JSON publication_date",
        "HTML datePublished",
        "HTML article:published_time",
        "HTML pubdate",
    }
)
MODIFICATION_TIMESTAMP_FIELDS = frozenset(
    {
        "dateModified",
        "article:modified_time",
        "updated",
        "dateUpdated",
        "Atom entry updated",
        "JSON updated_at",
    }
)
ISO_TZ_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})\b")
ISO_NAIVE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?\b")

NETWORK_LIMITS: dict[str, Any] = {
    "max_sources": MAX_SOURCES,
    "max_items_per_source": MAX_ITEMS_PER_SOURCE,
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "retry_count": RETRY_COUNT,
    "redirect_limit": REDIRECT_LIMIT,
    "max_response_bytes": MAX_RESPONSE_BYTES,
    "data_cost_rub": 0,
}


class SourceStatus(StrEnum):
    SUCCESS = "SUCCESS"
    NO_NEW_ITEMS = "NO_NEW_ITEMS"
    HTTP_FAILURE = "HTTP_FAILURE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    INVALID_RSS = "INVALID_RSS"
    MISSING_EXACT_TIMESTAMP = "MISSING_EXACT_TIMESTAMP"
    INVALID_TIMEZONE = "INVALID_TIMEZONE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    PUBLICATION_MATERIAL_MISSING = "PUBLICATION_MATERIAL_MISSING"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    SOURCE_DISABLED = "SOURCE_DISABLED"


@dataclass(frozen=True, slots=True)
class LiveExactSource:
    source_id: str
    ticker: str
    issuer: str
    instrument_uid: str
    source_family: str
    source_url: str
    official_domain: str
    mechanism_type: str
    timestamp_field: str
    timestamp_policy: str
    archive_capability: bool
    live_capability: bool
    provenance_evidence_url: str
    provenance_evidence_sha: str
    enabled: bool
    event_origin: str = "ISSUER_ORIGINATED"
    ticker_attribution_method: str = "ISSUER_OWNED_SINGLE_ISSUER_SOURCE"
    item_match_any: tuple[str, ...] = ()
    source_registry_version: str = SOURCE_REGISTRY_VERSION
    parser_version: str = PARSER_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LiveExactSource:
        if payload.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
            raise ValueError("SOURCE_REGISTRY_VERSION_MISMATCH")
        source = cls(
            source_id=str(payload["source_id"]),
            ticker=str(payload["ticker"]),
            issuer=str(payload["issuer"]),
            instrument_uid=str(payload["instrument_uid"]),
            source_family=str(payload["source_family"]),
            source_url=str(payload["source_url"]),
            official_domain=str(payload["official_domain"]),
            mechanism_type=str(payload["mechanism_type"]),
            timestamp_field=str(payload["timestamp_field"]),
            timestamp_policy=str(payload["timestamp_policy"]),
            archive_capability=bool(payload["archive_capability"]),
            live_capability=bool(payload["live_capability"]),
            provenance_evidence_url=str(payload["provenance_evidence_url"]),
            provenance_evidence_sha=str(payload["provenance_evidence_sha"]),
            enabled=bool(payload["enabled"]),
            event_origin=str(
                payload.get("event_origin")
                or (
                    "EXCHANGE_ORIGINATED"
                    if str(payload.get("source_family"))
                    == "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
                    else "ISSUER_ORIGINATED"
                )
            ),
            ticker_attribution_method=str(
                payload.get("ticker_attribution_method") or "ISSUER_OWNED_SINGLE_ISSUER_SOURCE"
            ),
            item_match_any=tuple(
                str(item)
                for item in cast("list[object]", payload.get("item_match_any") or [])
                if str(item).strip()
            ),
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not self.source_url.startswith(f"https://{self.official_domain}/"):
            raise ValueError("SOURCE_URL_MUST_MATCH_OFFICIAL_DOMAIN")
        if self.mechanism_type not in SUPPORTED_MECHANISMS:
            raise ValueError("UNSUPPORTED_LIVE_EXACT_MECHANISM")
        if self.timestamp_field in MODIFICATION_TIMESTAMP_FIELDS:
            raise ValueError("MODIFICATION_TIMESTAMP_DOES_NOT_QUALIFY")
        if self.timestamp_field not in PUBLICATION_TIMESTAMP_FIELDS:
            raise ValueError("UNSUPPORTED_TIMESTAMP_FIELD")
        if "+0300" not in self.timestamp_policy and "explicit" not in self.timestamp_policy:
            raise ValueError("TIMESTAMP_POLICY_MUST_REQUIRE_EXPLICIT_TIMEZONE")
        if not self.live_capability:
            raise ValueError("LIVE_EXACT_SOURCE_REQUIRES_LIVE_CAPABILITY")
        if self.event_origin not in {
            "ISSUER_ORIGINATED",
            "EXCHANGE_ORIGINATED",
            "REGULATOR_ORIGINATED",
            "OTHER_OFFICIAL",
        }:
            raise ValueError("UNSUPPORTED_EVENT_ORIGIN")
        if self.event_origin != "ISSUER_ORIGINATED" and not self.item_match_any:
            raise ValueError("SHARED_NON_ISSUER_SOURCE_REQUIRES_ITEM_ATTRIBUTION")
        if (
            self.event_origin == "ISSUER_ORIGINATED"
            and self.ticker_attribution_method != "ISSUER_OWNED_SINGLE_ISSUER_SOURCE"
            and not self.item_match_any
        ):
            raise ValueError("SHARED_ISSUER_SOURCE_REQUIRES_ITEM_ATTRIBUTION")
        if any(" " in token.strip() for token in self.item_match_any):
            raise ValueError("ITEM_MATCH_TOKENS_MUST_BE_ATOMIC")

    def payload(self) -> dict[str, Any]:
        return {
            "source_registry_version": self.source_registry_version,
            "source_id": self.source_id,
            "ticker": self.ticker,
            "issuer": self.issuer,
            "instrument_uid": self.instrument_uid,
            "source_family": self.source_family,
            "source_url": self.source_url,
            "official_domain": self.official_domain,
            "mechanism_type": self.mechanism_type,
            "timestamp_field": self.timestamp_field,
            "timestamp_policy": self.timestamp_policy,
            "archive_capability": self.archive_capability,
            "live_capability": self.live_capability,
            "provenance_evidence_url": self.provenance_evidence_url,
            "provenance_evidence_sha": self.provenance_evidence_sha,
            "enabled": self.enabled,
            "event_origin": self.event_origin,
            "ticker_attribution_method": self.ticker_attribution_method,
            "parser_version": self.parser_version,
            "item_match_any": list(self.item_match_any),
        }


def parse_rss_pubdate_exact(value: str) -> datetime:
    return parse_publication_timestamp_exact(value, field_name="RSS item pubDate")


def parse_publication_timestamp_exact(value: str, *, field_name: str) -> datetime:
    if field_name in MODIFICATION_TIMESTAMP_FIELDS:
        raise ValueError("MODIFICATION_TIMESTAMP_DOES_NOT_QUALIFY")
    raw = value.strip()
    if not raw or _looks_date_only(raw):
        raise ValueError("TIMESTAMP_NOT_EXACT")
    if not _has_explicit_timezone(raw):
        raise ValueError("INVALID_TIMEZONE")
    if ISO_TZ_PATTERN.fullmatch(raw):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError as exc:
            raise ValueError("TIMESTAMP_NOT_EXACT") from exc
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("TIMESTAMP_NOT_EXACT") from exc
    if parsed.tzinfo is None:
        raise ValueError("INVALID_TIMEZONE")
    return parsed.astimezone(UTC)


def is_future_metadata_only(publication_date: date) -> bool:
    return publication_date >= FUTURE_EVENT_HOLDOUT_START


def collection_safety_flags() -> dict[str, bool | int]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_ACQUISITION_ONLY": True,
        "DATA_COST_RUB": 0,
        "RAW_PUBLICATION_PRESERVATION_ENABLED": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "SPARSE_FAMILY_CREATED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "CONFIRMED_SIGNAL": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
        "REAL_TRADING_APPROVED": False,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "REAL_STOP_ORDER_ALLOWED": False,
        "REAL_MONEY_MOVEMENT_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MARGIN_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publication_material(snapshot: dict[str, Any]) -> str | None:
    fragments: list[str] = []
    for key in ("title", "description", "content"):
        value = snapshot.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(_normalize_material_text(value))
    deduped = list(dict.fromkeys(fragment for fragment in fragments if fragment))
    return "\n".join(deduped) if deduped else None


def publication_material_sha(snapshot: dict[str, Any]) -> str | None:
    material = publication_material(snapshot)
    return sha256_text(material) if material is not None else None


def _looks_date_only(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _has_explicit_timezone(value: str) -> bool:
    if ISO_TZ_PATTERN.fullmatch(value.strip()):
        return True
    if ISO_NAIVE_PATTERN.fullmatch(value.strip()):
        return False
    tail = value.rsplit(" ", 1)[-1].strip()
    return tail in {"GMT", "UTC"} or bool(
        tail.endswith("Z")
        or (len(tail) == 5 and tail[0] in {"+", "-"} and tail[1:].isdigit())
        or (len(tail) == 6 and tail[0] in {"+", "-"} and tail[3] == ":")
    )


def _normalize_material_text(value: str) -> str:
    return " ".join(value.split())
