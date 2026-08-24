from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from src.exact_event_source_depth_expansion.domain import metrics

ARTIFACT_VERSION = "exact-event-official-source-discovery-v5"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v5-official-source-discovery"
INPUT_DATASET_SHA = "62908b80f854c09c928bfd608009ea003ee887bcc93420b74ac556e0914853c4"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
MAX_TICKERS = 50
MAX_URLS_PER_TICKER = 10
MAX_REQUESTS_PER_DOMAIN = 25
MAX_PAGES_PER_SOURCE = 5
MAX_ITEMS_PER_SOURCE = 200
DEPRIORITIZED_TICKERS = frozenset({"MGNT", "T", "X5"})

DISCOVERY_PRIORITY_RULES: dict[str, Any] = {
    "version": "official-source-discovery-priority-rules-v5",
    "selection_inputs": [
        "current_exact_event_counts_by_ticker",
        "current_feature_ready_counts_by_ticker",
        "canonical_tqbr_rub_universe_metadata",
        "existing_source_registry_metadata",
    ],
    "forbidden_inputs": ["returns", "targets", "predictions", "model_metrics", "TEST_metrics"],
    "tiers": {
        "A_ZERO_FEATURE_READY": "ticker already in exact corpus with 0 feature-ready rows",
        "B_EXACT_1_5": "ticker has 1..5 exact events",
        "C_EXACT_6_20": "ticker has 6..20 exact events",
        "D_CANONICAL_TQBR_NOT_IN_EXACT": "canonical TQBR RUB ticker absent from exact corpus",
        "DEPRIORITIZED": "dominant or already well-covered cohorts",
    },
    "deprioritized_tickers": sorted(DEPRIORITIZED_TICKERS),
    "tie_break": [
        "tier_order",
        "existing_source_unknown_descending",
        "ticker_ascending",
    ],
    "max_tickers": MAX_TICKERS,
    "max_urls_per_ticker": MAX_URLS_PER_TICKER,
    "max_requests_per_domain": MAX_REQUESTS_PER_DOMAIN,
    "max_pages_per_source": MAX_PAGES_PER_SOURCE,
    "max_items_per_source": MAX_ITEMS_PER_SOURCE,
}


class DiscoveryState(StrEnum):
    EXACT_SOURCE_READY = "EXACT_SOURCE_READY"
    EXACT_SOURCE_FOUND_NO_ARCHIVE = "EXACT_SOURCE_FOUND_NO_ARCHIVE"
    DATE_ONLY_SOURCE = "DATE_ONLY_SOURCE"
    NO_TIMESTAMP_SOURCE = "NO_TIMESTAMP_SOURCE"
    NO_OFFICIAL_SOURCE_FOUND = "NO_OFFICIAL_SOURCE_FOUND"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    CAPTCHA_BLOCKED = "CAPTCHA_BLOCKED"
    TECHNICAL_FETCH_FAILED = "TECHNICAL_FETCH_FAILED"
    AMBIGUOUS_SOURCE_IDENTITY = "AMBIGUOUS_SOURCE_IDENTITY"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class SourceDiscoveryAuditRecord:
    ticker: str
    issuer: str
    source_url: str | None
    source_domain: str | None
    source_type: str | None
    source_family: str | None
    official_source_confirmed: bool
    timestamp_capability: str
    timestamp_field: str | None
    timezone_provenance: str | None
    archive_capability: bool
    discovery_method: str
    policy_status: str
    technical_status: str
    priority_tier: str
    candidate_state: str
    new_official_source: bool
    exact_capable: bool
    new_canonical_events: int
    duplicates: int
    ambiguous: int
    blocker: str | None
    provenance: str

    def payload(self) -> dict[str, Any]:
        return {
            "TICKER": self.ticker,
            "ISSUER": self.issuer,
            "SOURCE_URL": self.source_url,
            "SOURCE_DOMAIN": self.source_domain,
            "SOURCE_TYPE": self.source_type,
            "SOURCE_FAMILY": self.source_family,
            "OFFICIAL_SOURCE_CONFIRMED": self.official_source_confirmed,
            "TIMESTAMP_CAPABILITY": self.timestamp_capability,
            "TIMESTAMP_FIELD": self.timestamp_field,
            "TIMEZONE_PROVENANCE": self.timezone_provenance,
            "ARCHIVE_CAPABILITY": self.archive_capability,
            "DISCOVERY_METHOD": self.discovery_method,
            "POLICY_STATUS": self.policy_status,
            "TECHNICAL_STATUS": self.technical_status,
            "PRIORITY_TIER": self.priority_tier,
            "CANDIDATE_STATE": self.candidate_state,
            "NEW_OFFICIAL_SOURCE": self.new_official_source,
            "EXACT_CAPABLE": self.exact_capable,
            "NEW_CANONICAL_EVENTS": self.new_canonical_events,
            "DUPLICATES": self.duplicates,
            "AMBIGUOUS": self.ambiguous,
            "BLOCKER": self.blocker,
            "PROVENANCE": self.provenance,
        }


def discovery_safety_flags() -> dict[str, bool | int]:
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
        "SPARSE_FAMILY_CREATED": False,
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "DATA_COST_RUB": 0,
    }


def priority_tier(
    *,
    ticker: str,
    exact_count: int,
    feature_ready_count: int,
    in_exact_corpus: bool,
) -> str:
    if ticker in DEPRIORITIZED_TICKERS or exact_count > 20:
        return "DEPRIORITIZED"
    if in_exact_corpus and feature_ready_count == 0:
        return "A_ZERO_FEATURE_READY"
    if 1 <= exact_count <= 5:
        return "B_EXACT_1_5"
    if 6 <= exact_count <= 20:
        return "C_EXACT_6_20"
    if not in_exact_corpus:
        return "D_CANONICAL_TQBR_NOT_IN_EXACT"
    return "DEPRIORITIZED"


def parse_exact_timestamp(value: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("TIMESTAMP_NOT_EXACT")
    if _looks_date_only(raw):
        raise ValueError("TIMESTAMP_NOT_EXACT")
    parsed: datetime
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed = parsedate_to_datetime(raw)
    if parsed.tzinfo is None:
        raise ValueError("TIMESTAMP_NOT_EXACT")
    return parsed.astimezone(UTC)


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics(events, features)


def counter_payload(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _looks_date_only(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"
