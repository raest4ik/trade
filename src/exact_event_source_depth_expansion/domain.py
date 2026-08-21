from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from src.exact_event_source_diversity_v3.domain import concentration

ARTIFACT_VERSION = "exact-event-source-depth-expansion-v4"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v4-source-depth"
INPUT_DATASET_SHA = "62908b80f854c09c928bfd608009ea003ee887bcc93420b74ac556e0914853c4"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
MAX_SOURCES_PER_RUN = 25
MAX_PAGES_PER_SOURCE = 5
MAX_ITEMS_PER_SOURCE = 200

SOURCE_DEPTH_PRIORITY_RULES: dict[str, Any] = {
    "version": "source-depth-priority-rules-v4",
    "selection_inputs": ["current_exact_event_counts_by_ticker", "source_registry_metadata"],
    "forbidden_inputs": ["returns", "targets", "predictions", "model_metrics", "TEST_metrics"],
    "tiers": {
        "TIER_1": "exact events <= 5",
        "TIER_2": "exact events 6..20",
        "TIER_3": "exact events 21..50",
        "DEPRIORITIZED": "exact events > 50",
    },
    "tie_break": [
        "tier_order",
        "official_archive_candidate_descending",
        "official_exact_candidate_descending",
        "official_source_found_descending",
        "ticker_ascending",
    ],
    "active_archive_expansion_limit": MAX_SOURCES_PER_RUN,
    "max_pages_per_source": MAX_PAGES_PER_SOURCE,
    "max_items_per_source": MAX_ITEMS_PER_SOURCE,
}


class ArchiveBlocker(StrEnum):
    NO_OFFICIAL_SOURCE_FOUND = "NO_OFFICIAL_SOURCE_FOUND"
    SOURCE_DATE_ONLY = "SOURCE_DATE_ONLY"
    NO_ARCHIVE = "NO_ARCHIVE"
    ARCHIVE_EMPTY = "ARCHIVE_EMPTY"
    ARCHIVE_DEPTH_LIMIT_REACHED = "ARCHIVE_DEPTH_LIMIT_REACHED"
    ROBOTS_OR_POLICY_BLOCKED = "ROBOTS_OR_POLICY_BLOCKED"
    RATE_LIMITED = "RATE_LIMITED"
    TECHNICAL_FETCH_FAILED = "TECHNICAL_FETCH_FAILED"
    TIMESTAMP_NOT_EXACT = "TIMESTAMP_NOT_EXACT"
    TICKER_UNMATCHED = "TICKER_UNMATCHED"
    TICKER_AMBIGUOUS = "TICKER_AMBIGUOUS"
    DUPLICATE_ONLY = "DUPLICATE_ONLY"
    NO_HISTORICAL_ITEMS = "NO_HISTORICAL_ITEMS"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class ArchiveAuditRecord:
    ticker: str
    issuer: str
    official_source_url: str | None
    source_found: bool
    source_type: str
    source_family: str | None
    priority_tier: str
    exact_capable: bool
    archive_capable: bool
    earliest_discoverable_date: str | None
    latest_discoverable_date: str | None
    pages_probed: int
    items_discovered: int
    exact_items_discovered: int
    date_only_items_discovered: int
    new_canonical_events: int
    duplicates: int
    ambiguous: int
    blocker: str | None
    provenance: str

    def payload(self) -> dict[str, Any]:
        return {
            "TICKER": self.ticker,
            "ISSUER": self.issuer,
            "OFFICIAL_SOURCE_URL": self.official_source_url,
            "SOURCE_FOUND": self.source_found,
            "SOURCE_TYPE": self.source_type,
            "SOURCE_FAMILY": self.source_family,
            "PRIORITY_TIER": self.priority_tier,
            "EXACT_CAPABLE": self.exact_capable,
            "ARCHIVE_CAPABLE": self.archive_capable,
            "EARLIEST_DISCOVERABLE_DATE": self.earliest_discoverable_date,
            "LATEST_DISCOVERABLE_DATE": self.latest_discoverable_date,
            "PAGES_PROBED": self.pages_probed,
            "ITEMS_DISCOVERED": self.items_discovered,
            "EXACT_ITEMS_DISCOVERED": self.exact_items_discovered,
            "DATE_ONLY_ITEMS_DISCOVERED": self.date_only_items_discovered,
            "NEW_CANONICAL_EVENTS": self.new_canonical_events,
            "DUPLICATES": self.duplicates,
            "AMBIGUOUS": self.ambiguous,
            "BLOCKER": self.blocker,
            "PROVENANCE": self.provenance,
        }


def source_depth_safety_flags() -> dict[str, bool | int]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_ACQUISITION_ONLY": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
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
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "DATA_COST_RUB": 0,
    }


def priority_tier(count: int) -> str:
    if count <= 5:
        return "TIER_1"
    if count <= 20:
        return "TIER_2"
    if count <= 50:
        return "TIER_3"
    return "DEPRIORITIZED"


def parse_rfc822_timestamp(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("TIMESTAMP_NOT_EXACT")
    return parsed.astimezone(UTC)


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [row["metadata"] for row in events]
    by_event = {str(row["metadata"]["event_id"]): row["metadata"] for row in events}
    ticker_counts = Counter(str(row["ticker"]) for row in metadata)
    issuer_counts = Counter(str(row["issuer"]) for row in metadata)
    source_counts = Counter(str(row["source_code"]) for row in metadata)
    feature_by_ticker = Counter(str(by_event[str(row["event_id"])]["ticker"]) for row in features)
    feature_by_issuer = Counter(str(by_event[str(row["event_id"])]["issuer"]) for row in features)
    return {
        "EXACT_TOTAL": len(events),
        "EXACT_UNIQUE_TICKERS": len(ticker_counts),
        "EXACT_UNIQUE_ISSUERS": len(issuer_counts),
        "REACTION_READY": sum(bool(row["target_availability"]["reaction_ready"]) for row in events),
        "FEATURE_READY": len(features),
        "FEATURE_READY_UNIQUE_TICKERS": len(feature_by_ticker),
        "events_by_ticker": dict(sorted(ticker_counts.items())),
        "feature_ready_by_ticker": dict(sorted(feature_by_ticker.items())),
        "ticker_concentration": concentration(ticker_counts),
        "issuer_concentration": concentration(issuer_counts),
        "source_concentration": concentration(source_counts),
        "feature_ready_ticker_concentration": concentration(feature_by_ticker),
        "feature_ready_issuer_concentration": concentration(feature_by_issuer),
    }
