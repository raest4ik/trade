from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-event-live-source-breadth-expansion-v1"
MOEX_RISK_PARAMETERS_RSS_URL = "https://www.moex.com/export/news.aspx?cat=122"
MOEX_RISK_PARAMETERS_SOURCE_FAMILY = "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
DEFAULT_INPUT_EVENTS_PATH = "artifacts/chep-historical-exact-maturation-v1/events.jsonl"
DEFAULT_UNIVERSE_PATH = "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
DEFAULT_ELIGIBILITY_MANIFEST_PATH = (
    "artifacts/exact-event-security-tradability-eligibility-v1/manifest.json"
)
DEFAULT_LIVE_REGISTRY_PATH = "config/exact-event-live-official-sources.json"

MAX_TARGET_TICKERS = 30
TARGET_BATCH_SIZE = 5
MAX_FEED_CANDIDATES = 3
MAX_ITEMS_PER_FEED_DISCOVERY = 80
MAX_NEW_LIVE_SOURCES = 5
REQUEST_TIMEOUT_SECONDS = 10.0
RETRY_COUNT = 1
REDIRECT_LIMIT = 3
MAX_RESPONSE_BYTES = 2_000_000

DISCOVERY_LIMITS: dict[str, Any] = {
    "max_target_tickers": MAX_TARGET_TICKERS,
    "target_batch_size": TARGET_BATCH_SIZE,
    "max_feed_candidates": MAX_FEED_CANDIDATES,
    "max_items_per_feed_discovery": MAX_ITEMS_PER_FEED_DISCOVERY,
    "max_new_live_sources": MAX_NEW_LIVE_SOURCES,
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "retry_count": RETRY_COUNT,
    "redirect_limit": REDIRECT_LIMIT,
    "max_response_bytes": MAX_RESPONSE_BYTES,
    "data_cost_rub": 0,
}


class CandidateStatus(StrEnum):
    EXACT_LIVE_READY = "EXACT_LIVE_READY"
    EXACT_ARCHIVE_READY = "EXACT_ARCHIVE_READY"
    DATE_ONLY = "DATE_ONLY"
    NO_PUBLICATION_TIME = "NO_PUBLICATION_TIME"
    NO_TIMEZONE = "NO_TIMEZONE"
    NO_FEED_OR_ENDPOINT = "NO_FEED_OR_ENDPOINT"
    OFFICIAL_SOURCE_NOT_FOUND = "OFFICIAL_SOURCE_NOT_FOUND"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    ROBOTS_OR_POLICY_BLOCKED = "ROBOTS_OR_POLICY_BLOCKED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    CAPTCHA_BLOCKED = "CAPTCHA_BLOCKED"
    RATE_LIMITED = "RATE_LIMITED"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    CURRENTLY_NON_TRADABLE = "CURRENTLY_NON_TRADABLE"


def source_breadth_safety_flags() -> dict[str, bool | int]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_ACQUISITION_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "MARKET_MATURATION_INVOKED": False,
        "REACTION_MATURATION_INVOKED": False,
        "FEATURE_MATURATION_INVOKED": False,
        "CONFIRMED_SIGNAL": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
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
