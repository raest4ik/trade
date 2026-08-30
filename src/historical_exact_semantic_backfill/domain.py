from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "historical-exact-semantic-backfill-v1"
DEFAULT_DIAGNOSIS_ARTIFACT_ROOT = "artifacts/exact-feature-readiness-recovery-v1"
DEFAULT_MARKET_ARTIFACT_ROOT = "artifacts/exact-feature-readiness-recovery-v1"
DEFAULT_SNAPSHOT_ROOTS = (
    "artifacts/exact-event-live-source-breadth-expansion-v1/live-collection",
    "artifacts/exact-event-live-source-breadth-expansion-v2/live-collection",
)
EXPECTED_DIAGNOSIS_ARTIFACT_SHA = "08a1c38e15c03da95ac3a60477d784b749ee3694495ac69361da6b14b5514c17"
EXPECTED_RULES_V3_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)


class SemanticBackfillBlocker(StrEnum):
    SNAPSHOT_IDENTITY_UNRESOLVED = "PUBLICATION_SNAPSHOT_IDENTITY_UNRESOLVED"
    PUBLICATION_MATERIAL_MISSING = "PUBLICATION_MATERIAL_MISSING"
    SEMANTIC_EXTRACTION_FAILED = "SEMANTIC_EXTRACTION_FAILED"
    MARKET_FEATURES_MISSING = "MARKET_FEATURES_MISSING"
    MARKET_FEATURES_INCOMPLETE = "MARKET_FEATURES_INCOMPLETE"
    FEATURE_LEAKAGE_GUARD_REJECTED = "FEATURE_LEAKAGE_GUARD_REJECTED"
    FEATURE_STATE_NOT_PROPAGATED = "FEATURE_STATE_NOT_PROPAGATED"


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
        "FEATURE_DEFINITION_CHANGED": False,
        "REACTION_METHODOLOGY_CHANGED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "USES_MARKET_DATA_FOR_SEMANTICS": False,
        "USES_REACTION_DATA_FOR_SEMANTICS": False,
        "USES_TARGET_DATA_FOR_SEMANTICS": False,
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
