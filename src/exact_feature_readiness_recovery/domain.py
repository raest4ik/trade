from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from typing import Any

ARTIFACT_VERSION = "exact-feature-readiness-recovery-v1"
DEFAULT_INPUT_ARTIFACT_ROOT = "artifacts/consolidated-active-exact-historical-maturation-v1"
EXPECTED_INPUT_MATURATION_ARTIFACT_SHA = (
    "24946801c94e6194284bdf8bdfe1fa40ca1c6f614352552545dab7343ea0a80b"
)
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")


class FeatureRecoveryBlocker(StrEnum):
    FEATURE_PIPELINE_NOT_INVOKED = "FEATURE_PIPELINE_NOT_INVOKED"
    FEATURE_STATE_NOT_PROPAGATED = "FEATURE_STATE_NOT_PROPAGATED"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    PRE_EVENT_WARMUP_MISSING = "PRE_EVENT_WARMUP_MISSING"
    PRE_EVENT_WARMUP_INSUFFICIENT = "PRE_EVENT_WARMUP_INSUFFICIENT"
    SECURITY_HISTORY_MISSING = "SECURITY_HISTORY_MISSING"
    BENCHMARK_HISTORY_MISSING = "BENCHMARK_HISTORY_MISSING"
    SESSION_ALIGNMENT_FAILED = "SESSION_ALIGNMENT_FAILED"
    NON_TRADING_SESSION = "NON_TRADING_SESSION"
    PRE_OPEN = "PRE_OPEN"
    AFTER_CLOSE = "AFTER_CLOSE"
    INSTRUMENT_IDENTITY_UNRESOLVED = "INSTRUMENT_IDENTITY_UNRESOLVED"
    INSTRUMENT_IDENTITY_AMBIGUOUS = "INSTRUMENT_IDENTITY_AMBIGUOUS"
    MARKET_FEATURE_INPUT_INCOMPLETE = "MARKET_FEATURE_INPUT_INCOMPLETE"
    FEATURE_CALCULATION_FAILED = "FEATURE_CALCULATION_FAILED"
    FEATURE_LEAKAGE_GUARD_REJECTED = "FEATURE_LEAKAGE_GUARD_REJECTED"
    SEMANTIC_EVENT_FEATURES_MISSING = "SEMANTIC_EVENT_FEATURES_MISSING"
    CANONICAL_INTEGRATION_FAILED = "CANONICAL_INTEGRATION_FAILED"


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    feature_ready_decider: str
    canonical_feature_builder: str
    semantic_event_feature_builder: str
    required_pre_event_market_inputs: tuple[str, ...]
    minimum_warmup_window: str
    security_price_requirements: str
    benchmark_requirements: str
    session_requirements: str
    feature_schema_version: str
    leakage_guard: str
    canonical_write_location: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def pipeline_trace() -> PipelineTrace:
    return PipelineTrace(
        feature_ready_decider=(
            "src.consolidated_active_exact_historical_maturation.application:"
            "run_consolidated_active_exact_historical_maturation"
        ),
        canonical_feature_builder="src.exact_event_corpus.market:align_exact_event",
        semantic_event_feature_builder="src.events.domain.v3:EventAnalyzerV3",
        required_pre_event_market_inputs=(
            "security complete 1m T-Invest candles ending at or before publication",
            "IMOEX complete 1m T-Invest candles ending at or before publication",
            "5m/15m/30m/60m pre-event return windows",
        ),
        minimum_warmup_window="60m pre-event market context",
        security_price_requirements="complete T-Invest minute close prices; no forward-fill",
        benchmark_requirements="complete IMOEX minute close prices; same frozen benchmark path",
        session_requirements="DURING_MAIN_SESSION from exact_event_corpus.market.classify_session",
        feature_schema_version=(
            "event_features(primary_event_type,event_count,fact_count) from stored canonical "
            "semantics or EventAnalyzerV3 reconstruction; market_features from frozen exact "
            "pre-event candles"
        ),
        leakage_guard="max(feature_input_timestamp) <= publication_timestamp_utc",
        canonical_write_location=(
            "event.pre_event_market_features, event.target_availability.feature_ready, "
            "features.jsonl"
        ),
    )


def safety_flags() -> dict[str, bool | int]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "CONFIRMED_SIGNAL": False,
        "BACKTEST_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "REACTION_METHODOLOGY_CHANGED": False,
        "FEATURE_DEFINITION_CHANGED": False,
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
