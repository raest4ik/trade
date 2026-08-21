from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, cast

ARTIFACT_VERSION = "exact-event-source-diversity-v3"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v3-source-diversity"
INPUT_WARMUP_DATASET_SHA = "669aa6e8b11763131f3a940d669e446537a110066da22e7710649cdb2eaba6ff"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
SOURCE_REGISTRY_VERSION = "exact-event-source-registry-v3"
PARSER_VERSION = "official-moex-rss-diversity-v3"


class SourceStatus(StrEnum):
    EXACT = "EXACT"
    MIXED = "MIXED"
    DATE_ONLY = "DATE_ONLY"
    UNKNOWN = "UNKNOWN"
    TECHNICAL_BLOCKED = "TECHNICAL_BLOCKED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    FAILED_CLOSED = "FAILED_CLOSED"


@dataclass(frozen=True, slots=True)
class SourceDiscoveryRecord:
    issuer: str
    ticker: str
    official_domain: str | None
    source_url: str | None
    source_family: str | None
    transport_type: str | None
    timestamp_capability: str
    timezone_semantics: str
    archive_depth: str | None
    pagination_method: str | None
    machine_readable_status: str
    policy_status: str
    technical_status: str
    acquisition_status: str
    provenance: str
    official_ownership_proof: str | None
    source_found: bool
    exact_timestamp: bool
    archive: bool
    source_ready: bool
    technical_blocker: str | None
    policy_blocker: str | None
    notes: str

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "source_registry_version": SOURCE_REGISTRY_VERSION}


def parse_rss_pubdate_utc(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("TIMESTAMP_TIMEZONE_UNRESOLVED")
    from datetime import UTC

    return parsed.astimezone(UTC)


def source_diversity_safety_flags() -> dict[str, bool | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_ACQUISITION_ONLY": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
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
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "PRICE_ADJUSTMENT_STATUS": "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES",
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def concentration(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    shares = sorted((count / total for count in counter.values()), reverse=True) if total else []
    hhi = sum(share * share for share in shares)
    return {
        "counts": dict(sorted(counter.items())),
        "top1_share": shares[0] if shares else 0.0,
        "top3_share": sum(shares[:3]),
        "hhi": hhi,
        "effective_count": 1 / hhi if hhi else 0.0,
    }


def require_warmup_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("OUTPUT_DATASET_SHA") != INPUT_WARMUP_DATASET_SHA:
        raise ValueError("INPUT_WARMUP_DATASET_SHA_MISMATCH")
    if manifest.get("EXISTING_FEATURE_ROWS_PRESERVED") != "PASS":
        raise ValueError("INPUT_FEATURE_PRESERVATION_NOT_PASS")
    if manifest.get("LEAKAGE_CHECK") != "PASS":
        raise ValueError("INPUT_LEAKAGE_CHECK_NOT_PASS")
    safety_raw = manifest.get("safety", {})
    safety = cast("dict[str, Any]", safety_raw) if isinstance(safety_raw, dict) else {}
    if safety.get("TEST_OUTCOME_USED") or safety.get("FUTURE_EVENT_HOLDOUT_USED"):
        raise ValueError("INPUT_SAFETY_FLAGS_NOT_PASS")
