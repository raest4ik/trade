from __future__ import annotations

import hashlib
import json
from typing import Any

ARTIFACT_VERSION = "exact-event-live-source-breadth-expansion-v2"
EXPECTED_V1_ARTIFACT_SHA = "40df18a108113c5b897ed7bb1e089f42e8e10cf5cab76788d205d59d23b8e6e2"
EXPECTED_V1_CANDIDATE_SET_SHA = "796f099e2abc8b3159a5eefb7a8c45fe7d452bb339c8e0caf41f4d953ee06b04"
DEFAULT_V1_ARTIFACT_ROOT = "artifacts/exact-event-live-source-breadth-expansion-v1"
DEFAULT_BASE_EVENTS_PATH = "artifacts/chep-historical-exact-maturation-v1/events.jsonl"
DEFAULT_LIVE_REGISTRY_PATH = "config/exact-event-live-official-sources.json"
MAX_NEW_LIVE_SOURCES = 5
V1_ONBOARDED_TICKERS = ("AFKS", "ASTR", "ELMT", "OZON", "RUAL")
EXCLUDED_TICKERS = ("CHEP", *V1_ONBOARDED_TICKERS)


def safety_flags() -> dict[str, bool | int]:
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
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "MARKET_MATURATION_INVOKED": False,
        "REACTION_MATURATION_INVOKED": False,
        "FEATURE_MATURATION_INVOKED": False,
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


def artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)
