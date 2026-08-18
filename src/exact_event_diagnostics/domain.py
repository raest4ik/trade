from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = "exact-event-data-diagnostics-v1"
DATASET_VERSION = "exact-event-market-dataset-v2"
EXPECTED_DATASET_SHA = "20ab67ff4d94c59d6cf714f8b2f7c048031bda120bbd92ceb6e6185a838e14c3"
EXPECTED_BASE_MAIN_SHA = "c3a9795a570af9ed88bbe1e26dc87899d77ac040"
EXPECTED_BASELINE_VERSION = "exact-event-predictive-baseline-v1"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
EXACT_HORIZONS = ("1m", "5m", "15m", "30m", "60m")
PRIMARY_EXACT_HORIZON = "15m"
FLAT_RETURN_THRESHOLD = 0.002
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    dataset_root: Path
    baseline_root: Path
    output_root: Path
    git_sha: str


def diagnostic_safety_labels() -> dict[str, bool | str]:
    return {
        "RESEARCH_ONLY": True,
        "DIAGNOSTIC_ONLY": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
        "REAL_TRADING_APPROVED": False,
        "CONFIRMED_SIGNAL": False,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "REAL_STOP_ORDER_ALLOWED": False,
        "REAL_MONEY_MOVEMENT_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MARGIN_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
        "PRICE_ADJUSTMENT_STATUS": PRICE_ADJUSTMENT_STATUS,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_expected_exact_dataset(manifest: dict[str, Any], holdout: dict[str, Any]) -> None:
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("EXACT_DATASET_VERSION_MISMATCH")
    if manifest.get("exact_dataset_sha") != EXPECTED_DATASET_SHA:
        raise ValueError("EXACT_DATASET_SHA_MISMATCH")
    if manifest.get("EVENT_MARKET_LEAKAGE_CHECK") != "PASS":
        raise ValueError("EVENT_MARKET_LEAKAGE_CHECK_NOT_PASS")
    if manifest.get("holdout_guard") != "PASS" or holdout.get("holdout_guard") != "PASS":
        raise ValueError("FUTURE_HOLDOUT_GUARD_NOT_PASS")
    if bool(manifest.get("FUTURE_EVENT_HOLDOUT_OBSERVED")):
        raise ValueError("FUTURE_EVENT_HOLDOUT_OBSERVED_IN_DATASET_MANIFEST")
    if bool(holdout.get("FUTURE_EVENT_HOLDOUT_OBSERVED")):
        raise ValueError("FUTURE_EVENT_HOLDOUT_OBSERVED_IN_HOLDOUT_STATUS")
    if int(holdout.get("outcome_fields_exported_for_future", -1)) != 0:
        raise ValueError("FUTURE_HOLDOUT_OUTCOMES_EXPORTED")
    if holdout.get("FUTURE_EVENT_HOLDOUT_START") != FUTURE_EVENT_HOLDOUT_START.isoformat():
        raise ValueError("FUTURE_HOLDOUT_START_CHANGED")
    for flag in ("rules_changed", "qwen_changed", "qwen_run"):
        if bool(manifest.get(flag)):
            raise ValueError(f"{flag.upper()}_NOT_FROZEN")


def require_baseline_split_manifest(manifest: dict[str, Any], split: dict[str, Any]) -> None:
    if manifest.get("model_version") != EXPECTED_BASELINE_VERSION:
        raise ValueError("BASELINE_VERSION_MISMATCH")
    if manifest.get("dataset_sha") != EXPECTED_DATASET_SHA:
        raise ValueError("BASELINE_DATASET_SHA_MISMATCH")
    if bool(manifest.get("FUTURE_EVENT_HOLDOUT_USED")):
        raise ValueError("BASELINE_USED_FUTURE_HOLDOUT")
    if bool(manifest.get("FUTURE_EVENT_HOLDOUT_OBSERVED")):
        raise ValueError("BASELINE_OBSERVED_FUTURE_HOLDOUT")
    if manifest.get("TEST_STATUS") != "OBSERVED_AFTER_EXACT_BASELINE_V1":
        raise ValueError("BASELINE_TEST_STATUS_UNEXPECTED")
    if split.get("target_outcomes_inspected_before_lock") is not False:
        raise ValueError("SPLIT_TARGET_OUTCOMES_INSPECTED_BEFORE_LOCK")
    if split.get("cluster_integrity") != "PASS" or split.get("leakage_check") != "PASS":
        raise ValueError("BASELINE_SPLIT_LEAKAGE_CHECK_NOT_PASS")
