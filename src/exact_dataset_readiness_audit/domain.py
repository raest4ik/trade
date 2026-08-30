from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-dataset-readiness-audit-v1"
DEFAULT_INPUT_ARTIFACT_ROOT = "artifacts/historical-exact-semantic-backfill-v1"
EXPECTED_INPUT_ARTIFACT_SHA = "5f85b63f94e8b03a2726ef76d90911c9ba1123f2c814e2d030c7ec170c2627a7"
EXPECTED_RULES_V3_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")


class EventOrigin(StrEnum):
    ISSUER = "ISSUER_ORIGINATED"
    EXCHANGE = "EXCHANGE_ORIGINATED"
    REGULATOR = "REGULATOR_ORIGINATED"
    OTHER_OFFICIAL = "OTHER_OFFICIAL"
    UNKNOWN = "UNKNOWN_ORIGIN"


class ReadinessDecision(StrEnum):
    DATASET_READY_FOR_CONTROLLED_BASELINE = "DATASET_READY_FOR_CONTROLLED_BASELINE"
    ISSUER_COHORT_READY_EXCHANGE_COHORT_SEPARATE = "ISSUER_COHORT_READY_EXCHANGE_COHORT_SEPARATE"
    MORE_ISSUER_EVENT_DATA_REQUIRED = "MORE_ISSUER_EVENT_DATA_REQUIRED"
    SEMANTIC_REPRESENTATION_TOO_WEAK = "SEMANTIC_REPRESENTATION_TOO_WEAK"
    SOURCE_CONCENTRATION_TOO_HIGH = "SOURCE_CONCENTRATION_TOO_HIGH"
    DATASET_COMPOSITION_REVIEW_REQUIRED = "DATASET_COMPOSITION_REVIEW_REQUIRED"


def safety_flags() -> dict[str, bool | int]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "CONFIRMED_SIGNAL": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
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


def artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)
