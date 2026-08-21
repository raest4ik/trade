from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = "exact-event-market-history-warmup-recovery-v1"
INPUT_DATASET_VERSION = "exact-event-market-dataset-v2"
OUTPUT_DATASET_VERSION = "exact-event-market-dataset-v2-warmup-recovered-v1"
EXPECTED_INPUT_DATASET_SHA = "20ab67ff4d94c59d6cf714f8b2f7c048031bda120bbd92ceb6e6185a838e14c3"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
REQUIRED_LOOKBACK_MINUTES = 60
PRE_EVENT_HORIZONS_MINUTES = (5, 15, 30, 60)
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    base_main_sha: str
    input_dataset_sha: str = EXPECTED_INPUT_DATASET_SHA
    artifact_version: str = ARTIFACT_VERSION
    output_dataset_version: str = OUTPUT_DATASET_VERSION
    recovery_source: str = "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES_CACHE"
    acquisition_mode: str = "CACHE_ONLY_EXISTING_TINVEST_PROVENANCE"
    max_history_days: int = 7
    deterministic_safety_buffer_days: int = 7
    required_lookback_minutes: int = REQUIRED_LOOKBACK_MINUTES
    feature_methodology: str = "FROZEN_EXACT_PRE_EVENT_MARKET_CONTEXT"
    use_targets: bool = False
    use_test_outcomes: bool = False
    use_future_holdout_outcomes: bool = False
    use_moex_substitution: bool = False
    use_forward_fill: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WarmupEventRootCause:
    event_id: str
    ticker: str
    issuer: str
    source: str
    publication_timestamp_utc: str
    required_lookback_minutes: int
    earliest_required_timestamp_utc: str
    available_security_history_start_utc: str | None
    available_benchmark_history_start_utc: str | None
    missing_history_amount_minutes: int | None
    blocking_features_before: tuple[str, ...]

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
        "PRICE_ADJUSTMENT_STATUS": PRICE_ADJUSTMENT_STATUS,
    }


def acquisition_dates(published_at: datetime, *, max_history_days: int) -> tuple[date, ...]:
    if max_history_days < 0:
        raise ValueError("max_history_days must be non-negative")
    published = published_at.date()
    return tuple(published - timedelta(days=offset) for offset in range(max_history_days + 1))


def earliest_required_timestamp(published_at: datetime) -> datetime:
    return published_at - timedelta(minutes=REQUIRED_LOOKBACK_MINUTES)


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("dataset_version") != INPUT_DATASET_VERSION:
        raise ValueError("INPUT_DATASET_VERSION_MISMATCH")
    if manifest.get("exact_dataset_sha") != EXPECTED_INPUT_DATASET_SHA:
        raise ValueError("INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("EVENT_MARKET_LEAKAGE_CHECK") != "PASS":
        raise ValueError("INPUT_LEAKAGE_CHECK_NOT_PASS")
    if bool(manifest.get("FUTURE_EVENT_HOLDOUT_OBSERVED")):
        raise ValueError("INPUT_FUTURE_HOLDOUT_OBSERVED")
    for name in ("rules_changed", "qwen_changed", "qwen_run"):
        if bool(manifest.get(name)):
            raise ValueError(f"{name.upper()}_NOT_FROZEN")
