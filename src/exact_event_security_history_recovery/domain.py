from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-event-security-history-recovery-v1"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v3-security-history-recovered-v1"
PR36_OUTPUT_DATASET_SHA = "19c822112f8b79e09b4067fa253c10118d448981592f66c5b40bfb01495ffc46"
PR36_ARTIFACT_SHA = "8af80c5ad4d23df1debaff5f352166796e4be86517c7b000bcc8d8dbf990f786"
PR37_ARTIFACT_SHA = "f80cf31f1c8a6143a6a7b6af3d5128e76123d594382d612fbbc3ff689b0e53e9"
PR37_DIAGNOSTIC_COHORT_SHA = "b9e9ef0b3e9c65b33c30b492161399dac9dcf812e075ebec68991a72c56630f4"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")
MAX_HISTORY_DAYS = 7


class RecoveryBlocker(StrEnum):
    SECURITY_CACHE_ACQUISITION_FAILED = "SECURITY_CACHE_ACQUISITION_FAILED"
    SECURITY_HISTORY_INSUFFICIENT = "SECURITY_HISTORY_INSUFFICIENT"
    MARKET_HISTORY_WARMUP = "MARKET_HISTORY_WARMUP"
    BENCHMARK_HISTORY_MISSING = "BENCHMARK_HISTORY_MISSING"
    SESSION_ALIGNMENT_FAILED = "SESSION_ALIGNMENT_FAILED"
    PRE_OPEN = "PRE_OPEN"
    AFTER_CLOSE = "AFTER_CLOSE"
    NON_TRADING_DAY = "NON_TRADING_DAY"
    REACTION_WINDOW_INCOMPLETE = "REACTION_WINDOW_INCOMPLETE"
    SECURITY_REACTION_MISSING = "SECURITY_REACTION_MISSING"
    BENCHMARK_REACTION_MISSING = "BENCHMARK_REACTION_MISSING"
    IDENTITY_CHANGED = "IDENTITY_CHANGED"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class RecoveryIdentity:
    event_id: str
    ticker: str
    issuer: str
    publication_timestamp: datetime
    figi: str
    instrument_uid: str
    class_code: str

    def payload(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "issuer": self.issuer,
            "publication_timestamp": self.publication_timestamp.isoformat(),
            "figi": self.figi,
            "instrument_uid": self.instrument_uid,
            "class_code": self.class_code,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    source: str = "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES"
    interval: str = "1m"
    max_history_days: int = MAX_HISTORY_DAYS
    request_span: str = "UTC_CALENDAR_DAY"
    identity_binding: str = "ticker+figi+instrument_uid+class_code+interval"
    benchmark_methodology: str = "UNCHANGED_IMOEX_EXISTING_CACHE"
    feature_methodology: str = "FROZEN_EXACT_PRE_EVENT_MARKET_CONTEXT"
    reaction_horizons: tuple[str, ...] = HORIZONS
    future_event_holdout_start: str = FUTURE_EVENT_HOLDOUT_START.isoformat()

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def recovery_safety_flags() -> dict[str, bool | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_RECOVERY_ONLY": True,
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
        "TINVEST_SOURCE_ONLY": True,
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "SYNTHETIC_MARKET_DATA_USED": False,
        "PRICE_ADJUSTMENT_STATUS": "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES",
    }


def acquisition_bounds(published_at: datetime) -> tuple[datetime, datetime]:
    published = published_at.astimezone(UTC)
    start_day = published.date() - timedelta(days=MAX_HISTORY_DAYS)
    end_day = published.date() + timedelta(days=1)
    return datetime.combine(start_day, time.min, UTC), datetime.combine(end_day, time.min, UTC)


def acquisition_day_bounds(published_at: datetime) -> tuple[tuple[datetime, datetime], ...]:
    start, end = acquisition_bounds(published_at)
    days: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        next_day = cursor + timedelta(days=1)
        days.append((cursor, next_day))
        cursor = next_day
    return tuple(days)


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_pr36_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("OUTPUT_DATASET_SHA") != PR36_OUTPUT_DATASET_SHA:
        raise ValueError("PR36_OUTPUT_DATASET_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != PR36_ARTIFACT_SHA:
        raise ValueError("PR36_ARTIFACT_SHA_MISMATCH")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(manifest.get(key)):
            raise ValueError(f"PR36_{key}_NOT_SAFE")


def require_pr37_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != PR37_ARTIFACT_SHA:
        raise ValueError("PR37_ARTIFACT_SHA_MISMATCH")
    if manifest.get("INPUT_DATASET_SHA") != PR36_OUTPUT_DATASET_SHA:
        raise ValueError("PR37_INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("OUTPUT_DATASET_SHA") != PR36_OUTPUT_DATASET_SHA:
        raise ValueError("PR37_OUTPUT_DATASET_SHA_MISMATCH")
    if manifest.get("DIAGNOSTIC_COHORT_SHA") != PR37_DIAGNOSTIC_COHORT_SHA:
        raise ValueError("PR37_DIAGNOSTIC_COHORT_SHA_MISMATCH")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(manifest.get(key)):
            raise ValueError(f"PR37_{key}_NOT_SAFE")
