from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START

ARTIFACT_VERSION = "timezone-verified-issuer-exact-source-discovery-v2"
DEFAULT_READINESS_AUDIT_ROOT = "artifacts/exact-dataset-readiness-audit-v1"
DEFAULT_ISSUER_DIVERSITY_ROOT = "artifacts/issuer-exact-historical-diversity-expansion-v1"
MAX_DOMAINS_TO_AUDIT = 15
MAX_NEW_STRICT_EXACT_SOURCE_CANDIDATES = 5
MIN_HISTORICAL_ITEMS_VERIFIED = 3


class SourceStatus(StrEnum):
    STRICT_EXACT_HISTORICAL_READY = "STRICT_EXACT_HISTORICAL_READY"
    STRICT_EXACT_LIVE_ONLY = "STRICT_EXACT_LIVE_ONLY"
    CLOCK_TIME_WITHOUT_TIMEZONE = "CLOCK_TIME_WITHOUT_TIMEZONE"
    DATE_ONLY = "DATE_ONLY"
    NO_PUBLIC_ARCHIVE = "NO_PUBLIC_ARCHIVE"
    NO_PUBLICATION_MATERIAL = "NO_PUBLICATION_MATERIAL"
    TECHNICAL_BLOCKER = "TECHNICAL_BLOCKER"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    ALREADY_COVERED = "ALREADY_COVERED"


class FinalDecision(StrEnum):
    NEW_HISTORICAL_STRICT_EXACT_SOURCES_FOUND = "NEW_HISTORICAL_STRICT_EXACT_SOURCES_FOUND"
    HISTORICAL_STRICT_EXACT_SOURCE_YIELD_LOW = "HISTORICAL_STRICT_EXACT_SOURCE_YIELD_LOW"
    HISTORICAL_STRICT_EXACT_SOURCES_EFFECTIVELY_EXHAUSTED = (
        "HISTORICAL_STRICT_EXACT_SOURCES_EFFECTIVELY_EXHAUSTED"
    )
    SOURCE_EVIDENCE_REVIEW_REQUIRED = "SOURCE_EVIDENCE_REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class CandidateSource:
    ticker: str
    issuer: str
    official_domain: str
    source_url: str
    source_mechanism: str
    event_origin: str = "ISSUER_ORIGINATED"
    source_family: str | None = None
    known_prior_status: str | None = None

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_family"] = self.source_family or _source_family(self.ticker, self.source_url)
        return result


def _source_family(ticker: str, source_url: str) -> str:
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:10].upper()
    return f"{ticker}_TZ_VERIFIED_DISCOVERY_{digest}"


def safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_COST_RUB": 0,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "FEATURE_DEFINITION_CHANGED": False,
        "REACTION_METHODOLOGY_CHANGED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "TINVEST_REQUESTS": 0,
        "MARKET_PRICE_LOOKUPS": 0,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "REAL_STOP_ORDER_ALLOWED": False,
        "REAL_MONEY_MOVEMENT_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MARGIN_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
        "SOURCE_RANKING_USED_OUTCOME_FIELDS": False,
        "SOURCE_RANKING_USED_MODEL_METRICS": False,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_sha(manifest: dict[str, Any]) -> str:
    excluded = {"ARTIFACT_SHA", "NETWORK_PROVENANCE_SHA", "created_at", "git_sha"}
    return sha256_payload({key: value for key, value in manifest.items() if key not in excluded})


def is_historical(publication_date: date) -> bool:
    return publication_date < FUTURE_EVENT_HOLDOUT_START
