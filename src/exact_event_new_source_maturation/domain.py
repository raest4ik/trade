from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from typing import Any

ARTIFACT_VERSION = "exact-event-new-source-maturation-v1"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v3-new-source-matured-v1"
INPUT_DATASET_SHA = "91998b73c8dd3243626e7c61c074969ec0aab72b62ff1b05955c981126b87cd9"
PREVIOUS_DATASET_SHA = "669aa6e8b11763131f3a940d669e446537a110066da22e7710649cdb2eaba6ff"
PR35_ARTIFACT_SHA = "299bd5eb45027cd379f211865c6555ed347017b95b0562ad3b8d16235ffdcfda"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")


def maturation_safety_flags() -> dict[str, bool | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_MATURATION_ONLY": True,
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


def require_input_manifests(previous: dict[str, Any], current: dict[str, Any]) -> None:
    if previous.get("OUTPUT_DATASET_SHA") != PREVIOUS_DATASET_SHA:
        raise ValueError("PREVIOUS_DATASET_SHA_MISMATCH")
    if current.get("OUTPUT_DATASET_SHA") != INPUT_DATASET_SHA:
        raise ValueError("INPUT_DATASET_SHA_MISMATCH")
    if current.get("ARTIFACT_SHA") != PR35_ARTIFACT_SHA:
        raise ValueError("PR35_ARTIFACT_SHA_MISMATCH")
    if current.get("EXACT_V2_PRESERVED") != "YES":
        raise ValueError("PR35_EXACT_V2_NOT_PRESERVED")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(current.get(key)):
            raise ValueError(f"INPUT_{key}_NOT_SAFE")
