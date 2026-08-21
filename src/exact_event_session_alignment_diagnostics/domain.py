from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-event-session-alignment-diagnostics-v1"
INPUT_DATASET_SHA = "62908b80f854c09c928bfd608009ea003ee887bcc93420b74ac556e0914853c4"
OUTPUT_DATASET_SHA = INPUT_DATASET_SHA
PR38_ARTIFACT_SHA = "d020851303c9536b654338918fadadb05deafac5e509af16cde73140e44818c3"
PR38_RECOVERY_COHORT_SHA = "b9e9ef0b3e9c65b33c30b492161399dac9dcf812e075ebec68991a72c56630f4"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
NEIGHBORHOOD_BEFORE = timedelta(minutes=30)
NEIGHBORHOOD_AFTER = timedelta(minutes=30)


class SessionAlignmentRootCause(StrEnum):
    SESSION_UNKNOWN_NO_COMMON_CANDLE = "SESSION_UNKNOWN_NO_COMMON_CANDLE"
    SESSION_UNKNOWN_COMMON_CANDLE_TOO_FAR = "SESSION_UNKNOWN_COMMON_CANDLE_TOO_FAR"
    BASELINE_SECURITY_MISSING = "BASELINE_SECURITY_MISSING"
    BASELINE_BENCHMARK_MISSING = "BASELINE_BENCHMARK_MISSING"
    EFFECTIVE_SECURITY_MISSING = "EFFECTIVE_SECURITY_MISSING"
    EFFECTIVE_BENCHMARK_MISSING = "EFFECTIVE_BENCHMARK_MISSING"
    SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH = "SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH"
    SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH = "SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH"
    SECURITY_CANDLE_INCOMPLETE = "SECURITY_CANDLE_INCOMPLETE"
    BENCHMARK_CANDLE_INCOMPLETE = "BENCHMARK_CANDLE_INCOMPLETE"
    SECURITY_MINUTE_GAP = "SECURITY_MINUTE_GAP"
    BENCHMARK_MINUTE_GAP = "BENCHMARK_MINUTE_GAP"
    COMMON_MINUTE_GAP = "COMMON_MINUTE_GAP"
    CACHE_WINDOW_TOO_NARROW = "CACHE_WINDOW_TOO_NARROW"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"


class RecoveryRecommendationType(StrEnum):
    RECOVERY_BY_CACHE_EXTENSION = "RECOVERY_BY_CACHE_EXTENSION"
    RECOVERY_BY_MISSING_CANDLE_ACQUISITION = "RECOVERY_BY_MISSING_CANDLE_ACQUISITION"
    METHODOLOGY_CHANGE_REQUIRED = "METHODOLOGY_CHANGE_REQUIRED"
    DATA_PROVIDER_GAP = "DATA_PROVIDER_GAP"
    NO_SAFE_RECOVERY = "NO_SAFE_RECOVERY"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class SessionDiagnosticIdentity:
    event_id: str
    ticker: str
    publication_timestamp: str
    figi: str
    instrument_uid: str
    class_code: str

    def payload(self) -> dict[str, str]:
        return asdict(self)


def session_diagnostic_safety_flags() -> dict[str, bool | str]:
    return {
        "RESEARCH_ONLY": True,
        "DIAGNOSTICS_ONLY": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "ALIGNMENT_METHODOLOGY_CHANGED": False,
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
    }


def require_pr38_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != PR38_ARTIFACT_SHA:
        raise ValueError("PR38_ARTIFACT_SHA_MISMATCH")
    if (
        manifest.get("INPUT_DATASET_SHA")
        != "19c822112f8b79e09b4067fa253c10118d448981592f66c5b40bfb01495ffc46"
    ):
        raise ValueError("PR38_INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("OUTPUT_DATASET_SHA") != INPUT_DATASET_SHA:
        raise ValueError("PR38_OUTPUT_DATASET_SHA_MISMATCH")
    if manifest.get("RECOVERY_COHORT_SHA") != PR38_RECOVERY_COHORT_SHA:
        raise ValueError("PR38_RECOVERY_COHORT_SHA_MISMATCH")
    if manifest.get("EXISTING_FEATURE_ROWS_PRESERVED") != "PASS":
        raise ValueError("PR38_FEATURE_PRESERVATION_NOT_PASS")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(manifest.get(key)):
            raise ValueError(f"PR38_{key}_NOT_SAFE")


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
