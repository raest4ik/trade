from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Any

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START

ARTIFACT_VERSION = "chep-historical-exact-maturation-v1"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v4-chep-historical-matured-v1"
EXPECTED_COLLECTOR_ARTIFACT_SHA = "4f2683ac57deda95f74eb10e0d889d07b00a35ea1d68abfd347868836ca9d91b"
EXPECTED_COLLECTOR_SOURCE_REGISTRY_SHA = (
    "1c7ab9d7c269585884541fc2295e60fba697aa5c6ec883a8e528780736fc55b1"
)
EXPECTED_COLLECTOR_NETWORK_PROVENANCE_SHA = (
    "265e5954cd182872227a62c83880b8142c21f6b0f21f440768f0639e845f2442"
)
EXPECTED_COLLECTOR_RAW_SNAPSHOT_SHA = (
    "c4501abbd7445199fe9790a1fd44393f6f8761ceae98abd2edc1bb34f128a52d"
)
EXPECTED_COLLECTOR_EVENT_METADATA_SHA = (
    "c4b7b838d60d0105a3f86d8485cf27129e0df623c6e1d3421815fd23c0858f25"
)
EXPECTED_COLLECTOR_DEDUPE_STATE_SHA = (
    "2a45ac4e48ee8725d86e4613aae391c824e243e385ba7309a8e4d9b32dc5a39a"
)
EXPECTED_COLLECTOR_ITEMS = 50
EXPECTED_HISTORICAL_COHORT = 44
EXPECTED_FUTURE_COHORT = 6
HORIZONS = ("1m", "5m", "15m", "30m", "60m")
MAX_HISTORY_DAYS = 7


class ChepMaturationBlocker(StrEnum):
    SECURITY_HISTORY_MISSING = "SECURITY_HISTORY_MISSING"
    BENCHMARK_HISTORY_MISSING = "BENCHMARK_HISTORY_MISSING"
    SESSION_ALIGNMENT_FAILED = "SESSION_ALIGNMENT_FAILED"
    PRE_EVENT_WARMUP_MISSING = "PRE_EVENT_WARMUP_MISSING"
    EVENT_FEATURES_MISSING = "EVENT_FEATURES_MISSING"
    NON_TRADING_SESSION = "NON_TRADING_SESSION"
    AFTER_CLOSE = "AFTER_CLOSE"
    PRE_OPEN = "PRE_OPEN"
    REACTION_MISSING = "REACTION_MISSING"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    FUTURE_METADATA_ONLY = "FUTURE_METADATA_ONLY"
    OTHER_EXISTING_CANONICAL_BLOCKER = "OTHER_EXISTING_CANONICAL_BLOCKER"


class FutureHoldoutReadAttemptError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChepIdentity:
    ticker: str
    issuer: str
    instrument_uid: str
    figi: str | None
    class_code: str | None
    exchange: str | None
    currency: str | None
    identity_provenance: str
    history_available: bool | None
    first_1day_candle_date: str | None
    last_1day_candle_date: str | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MarketAcquisitionConfig:
    source: str = "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES"
    interval: str = "1m"
    max_history_days: int = MAX_HISTORY_DAYS
    request_span: str = "UTC_CALENDAR_DAY"
    identity_binding: str = "ticker+instrument_uid+optional_figi+optional_class_code"
    benchmark_methodology: str = "UNCHANGED_IMOEX_EXISTING_CACHE"
    feature_methodology: str = "FROZEN_EXACT_PRE_EVENT_MARKET_CONTEXT"
    reaction_horizons: tuple[str, ...] = HORIZONS
    future_event_holdout_start: str = FUTURE_EVENT_HOLDOUT_START.isoformat()
    data_cost_rub: int = 0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def guard_future_market_access(publication_timestamp_utc: datetime) -> None:
    if publication_timestamp_utc.astimezone(UTC).date() >= FUTURE_EVENT_HOLDOUT_START:
        raise FutureHoldoutReadAttemptError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def acquisition_day_bounds(published_at: datetime) -> tuple[tuple[datetime, datetime], ...]:
    published = published_at.astimezone(UTC)
    start_day = published.date() - timedelta(days=MAX_HISTORY_DAYS)
    end_day = published.date() + timedelta(days=1)
    cursor = datetime.combine(start_day, time.min, UTC)
    end = datetime.combine(end_day, time.min, UTC)
    days: list[tuple[datetime, datetime]] = []
    while cursor < end:
        next_day = cursor + timedelta(days=1)
        days.append((cursor, next_day))
        cursor = next_day
    return tuple(days)


def maturation_safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_MATURATION_ONLY": True,
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
        "SPARSE_FAMILY_CREATED": False,
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


def require_collector_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "ARTIFACT_SHA": EXPECTED_COLLECTOR_ARTIFACT_SHA,
        "SOURCE_REGISTRY_SHA": EXPECTED_COLLECTOR_SOURCE_REGISTRY_SHA,
        "NETWORK_PROVENANCE_SHA": EXPECTED_COLLECTOR_NETWORK_PROVENANCE_SHA,
        "RAW_SNAPSHOT_SHA": EXPECTED_COLLECTOR_RAW_SNAPSHOT_SHA,
        "COLLECTED_EVENT_METADATA_SHA": EXPECTED_COLLECTOR_EVENT_METADATA_SHA,
        "DEDUPE_STATE_SHA": EXPECTED_COLLECTOR_DEDUPE_STATE_SHA,
        "ITEMS_FETCHED": EXPECTED_COLLECTOR_ITEMS,
        "NEW_HISTORICAL_EXACT_EVENTS": EXPECTED_HISTORICAL_COHORT,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": EXPECTED_FUTURE_COHORT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"COLLECTOR_{key}_MISMATCH")
    for key in (
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
        "TEST_OUTCOME_USED",
        "MODEL_TRAINING_PERFORMED",
        "TEST_EVALUATION_PERFORMED",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"COLLECTOR_{key}_NOT_SAFE")


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
