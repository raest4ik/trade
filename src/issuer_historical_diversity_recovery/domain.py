from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "historical-issuer-diversity-recovery-v1"
SOURCE_OPTIONS_VERSION = "issuer-historical-exact-source-options-v1"

DEFAULT_BACKFILL_ROOT = "artifacts/historical-exact-semantic-backfill-v1"
DEFAULT_ML_V2_READINESS_ROOT = "artifacts/ml-v2-readiness-audit-v1"
DEFAULT_TZ_DISCOVERY_ROOT = "artifacts/timezone-verified-issuer-exact-source-discovery-v2"
DEFAULT_ISSUER_DIVERSITY_ROOT = "artifacts/issuer-exact-historical-diversity-expansion-v1"
DEFAULT_CONSOLIDATED_MATURATION_ROOT = (
    "artifacts/consolidated-active-exact-historical-maturation-v1"
)
DEFAULT_CHEP_MATURATION_ROOT = "artifacts/chep-historical-exact-maturation-v1-cache-only"

FUTURE_HOLDOUT_START = "2026-08-11"
CANONICAL_COHORT = "ISSUER_ORIGINATED_STRICT_EXACT_HISTORICAL_FEATURE_READY"


class SourceOptionStatus(StrEnum):
    ALREADY_EXHAUSTED = "ALREADY_EXHAUSTED"
    DATE_ONLY = "DATE_ONLY"
    CLOCK_WITHOUT_TIMEZONE = "CLOCK_WITHOUT_TIMEZONE"
    TECHNICAL_BLOCKER = "TECHNICAL_BLOCKER"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    PAID_LICENSE_REQUIRED = "PAID_LICENSE_REQUIRED"
    AUTHENTICATED_OFFICIAL_API = "AUTHENTICATED_OFFICIAL_API"
    NEW_MECHANISM_NOT_YET_TESTED = "NEW_MECHANISM_NOT_YET_TESTED"
    STRICT_EXACT_HISTORICAL_CAPABLE = "STRICT_EXACT_HISTORICAL_CAPABLE"


class RecoveryDecision(StrEnum):
    HISTORICAL_DIVERSITY_RECOVERY_READY = "HISTORICAL_DIVERSITY_RECOVERY_READY"
    PAID_OR_AUTHENTICATED_SOURCE_REQUIRED = "PAID_OR_AUTHENTICATED_SOURCE_REQUIRED"
    EXISTING_LOCAL_DATA_RECOVERY_AVAILABLE = "EXISTING_LOCAL_DATA_RECOVERY_AVAILABLE"
    NO_METHOD_SAFE_HISTORICAL_PATH_FOUND = "NO_METHOD_SAFE_HISTORICAL_PATH_FOUND"
    PARTIAL_DIVERSITY_GAIN_ONLY = "PARTIAL_DIVERSITY_GAIN_ONLY"


@dataclass(frozen=True, slots=True)
class SourceOption:
    provider: str
    ticker_scope: str
    issuer_scope: str
    mechanism: str
    status: SourceOptionStatus
    evidence_source: str
    evidence_url: str | None
    timestamp_contract: str
    timezone_contract: str
    historical_archive: str
    publication_identity: str
    access_status: str
    storage_or_license: str
    internal_ml_research_status: str
    candidate_count: int = 1
    no_reaudit_reason: str | None = None
    network_requests_performed: int = 0
    future_outcomes_read: int = 0
    future_targets_read: int = 0
    future_price_lookups: int = 0

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


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
        "NEW_MODEL_CREATED": False,
        "HYPERPARAMETER_SEARCH_PERFORMED": False,
        "TEST_MODEL_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "OLD_BASELINE_TEST_OBSERVED": True,
        "OLD_BASELINE_TEST_USED_FOR_SELECTION": False,
        "SOURCE_SELECTION_USED_MARKET_OUTCOMES": False,
        "SOURCE_SELECTION_USED_MODEL_PERFORMANCE": False,
        "SOURCE_SELECTION_USED_FUTURE_OUTCOMES": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
        "FUTURE_PRICE_LOOKUPS": 0,
        "REAL_TRADING_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_sha(manifest: dict[str, Any]) -> str:
    excluded = {"ARTIFACT_SHA", "created_at", "git_sha"}
    return sha256_payload({key: value for key, value in manifest.items() if key not in excluded})
