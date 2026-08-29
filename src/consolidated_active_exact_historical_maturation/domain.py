from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

ARTIFACT_VERSION = "consolidated-active-exact-historical-maturation-v1"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v5-active-historical-matured-v1"
DEFAULT_V1_ARTIFACT_ROOT = "artifacts/exact-event-live-source-breadth-expansion-v1"
DEFAULT_V2_ARTIFACT_ROOT = "artifacts/exact-event-live-source-breadth-expansion-v2"
DEFAULT_BASE_DATASET_ROOT = "artifacts/chep-historical-exact-maturation-v1"
DEFAULT_LIVE_REGISTRY_PATH = "config/exact-event-live-official-sources.json"
DEFAULT_UNIVERSE_ROOT = "artifacts/tinvest-market-universe-raw-v1"
EXPECTED_V1_ARTIFACT_SHA = "40df18a108113c5b897ed7bb1e089f42e8e10cf5cab76788d205d59d23b8e6e2"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")
BENCHMARK_TICKER = "IMOEX"


@dataclass(frozen=True, slots=True)
class ActiveIdentity:
    ticker: str
    issuer: str
    instrument_uid: str
    figi: str | None
    class_code: str | None
    exchange: str | None
    currency: str | None
    source_id: str
    source_family: str
    identity_provenance: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MarketAcquisitionConfig:
    source: str = "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES"
    interval: str = "1m"
    max_history_days: int = 7
    request_span: str = "UTC_CALENDAR_DAY"
    identity_binding: str = "ticker+instrument_uid+figi+class_code"
    benchmark: str = BENCHMARK_TICKER
    benchmark_methodology: str = "UNCHANGED_IMOEX_EXACT_INTRADAY"
    feature_methodology: str = "FROZEN_EXACT_PRE_EVENT_MARKET_CONTEXT"
    reaction_horizons: tuple[str, ...] = HORIZONS
    future_event_holdout_start: str = FUTURE_EVENT_HOLDOUT_START.isoformat()
    data_cost_rub: int = 0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_MATURATION_ONLY": True,
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
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "SYNTHETIC_MARKET_DATA_USED": False,
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
        "PRICE_ADJUSTMENT_STATUS": "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES",
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
