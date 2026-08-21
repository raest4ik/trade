from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-event-sparse-session-methodology-study-v1"
INPUT_DATASET_SHA = "62908b80f854c09c928bfd608009ea003ee887bcc93420b74ac556e0914853c4"
OUTPUT_DATASET_SHA = INPUT_DATASET_SHA
PR39_ARTIFACT_SHA = "703ea7b1446b49b1f76d8619b6898b0fe9841edd54f705ef17e15340b97618a4"
PR39_SESSION_DIAGNOSTIC_COHORT_SHA = (
    "173201ad2197cedc45c152bc794fcc6cf25bf50dadc75840ae5235a503451225"
)
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
STUDY_WINDOW = timedelta(minutes=30)
PRE_EVENT_DENSITY_WINDOWS = (15, 30, 60)
DELAY_THRESHOLDS_SECONDS = (60, 120, 180, 300, 600)
DEVELOPMENT_SPLITS = frozenset({"TRAIN", "VALIDATION"})


class MethodologyStudyRecommendation(StrEnum):
    KEEP_STRICT_ONLY = "KEEP_STRICT_ONLY"
    SEPARATE_SPARSE_FAMILY_STUDY_JUSTIFIED = "SEPARATE_SPARSE_FAMILY_STUDY_JUSTIFIED"
    INSUFFICIENT_DEVELOPMENT_EVIDENCE = "INSUFFICIENT_DEVELOPMENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class ExactEventMetadata:
    event_id: str
    ticker: str
    issuer: str
    source_family: str
    publication_timestamp: datetime
    instrument_uid: str
    future_holdout: bool
    timestamp_quality: str

    def identity_payload(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "issuer": self.issuer,
            "source_family": self.source_family,
            "publication_timestamp": self.publication_timestamp.isoformat(),
            "instrument_uid": self.instrument_uid,
            "timestamp_quality": self.timestamp_quality,
        }


@dataclass(frozen=True, slots=True)
class TimestampCandle:
    instrument_uid: str
    begin_at: datetime
    end_at: datetime
    is_complete: bool
    ticker: str | None = None
    figi: str | None = None
    class_code: str | None = None

    def timestamp_payload(self) -> dict[str, str | bool | None]:
        return asdict(self) | {
            "begin_at": self.begin_at.isoformat(),
            "end_at": self.end_at.isoformat(),
        }


DECISION_RULES: dict[str, Any] = {
    "registered_before_study_summary": True,
    "version": "sparse-session-study-decision-rules-v1",
    "recommendation_order": [
        MethodologyStudyRecommendation.INSUFFICIENT_DEVELOPMENT_EVIDENCE.value,
        MethodologyStudyRecommendation.SEPARATE_SPARSE_FAMILY_STUDY_JUSTIFIED.value,
        MethodologyStudyRecommendation.KEEP_STRICT_ONLY.value,
    ],
    "minimum_recommendation_sample_size": 30,
    "minimum_sparse_gt60_events": 5,
    "minimum_sparse_unique_tickers": 3,
    "minimum_300s_incremental_share_vs_60s": 0.05,
    "maximum_sparse_top1_share": 0.60,
    "maximum_sparse_top3_share": 0.90,
    "cache_uncertain_rows_excluded_from_recommendation": True,
    "outcomes_models_returns_used": False,
    "methodological_rationale": (
        "A separate sparse family needs repeated multi-security metadata-only evidence, "
        "material coverage gain beyond the strict 60s rule, and no single-ticker dominance."
    ),
}


def sparse_study_safety_flags() -> dict[str, bool | str | int]:
    return {
        "RESEARCH_ONLY": True,
        "METHODOLOGY_STUDY_ONLY": True,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "OBSERVED_TEST_ROWS_USED": 0,
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
        "PRODUCTION_DATASET_CHANGED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "MARKET_DATA_METHOD_CHANGED": False,
        "TINVEST_READONLY_ONLY": True,
        "BROKER_WRITE_SURFACE_USED": False,
    }


def require_pr39_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != PR39_ARTIFACT_SHA:
        raise ValueError("PR39_ARTIFACT_SHA_MISMATCH")
    if manifest.get("INPUT_DATASET_SHA") != INPUT_DATASET_SHA:
        raise ValueError("PR39_INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("OUTPUT_DATASET_SHA") != OUTPUT_DATASET_SHA:
        raise ValueError("PR39_OUTPUT_DATASET_SHA_MISMATCH")
    if manifest.get("SESSION_DIAGNOSTIC_COHORT_SHA") != PR39_SESSION_DIAGNOSTIC_COHORT_SHA:
        raise ValueError("PR39_SESSION_DIAGNOSTIC_COHORT_SHA_MISMATCH")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(manifest.get(key)):
            raise ValueError(f"PR39_{key}_NOT_SAFE")


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("NAIVE_TIMESTAMP_NOT_ALLOWED")
    return parsed.astimezone(UTC)


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
