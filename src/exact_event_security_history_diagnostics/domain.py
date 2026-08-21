from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

from src.tinvest_market.client import TInvestDailyCandle, TInvestInstrument, TInvestMinuteCandle

ARTIFACT_VERSION = "exact-event-security-history-diagnostics-v1"
PR36_INPUT_DATASET_SHA = "91998b73c8dd3243626e7c61c074969ec0aab72b62ff1b05955c981126b87cd9"
PR36_OUTPUT_DATASET_SHA = "19c822112f8b79e09b4067fa253c10118d448981592f66c5b40bfb01495ffc46"
PR36_ARTIFACT_SHA = "8af80c5ad4d23df1debaff5f352166796e4be86517c7b000bcc8d8dbf990f786"
PR36_COHORT_SHA = "30e6d58e4c1287a65ad3f6d3c625b4fb23482755081ca6dfc2535cae96617fd9"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
PROBE_OFFSETS_DAYS = (120, 60, 30, 10)


class RootCauseStatus(StrEnum):
    CURRENT_IDENTITY_HAS_HISTORY = "CURRENT_IDENTITY_HAS_HISTORY"
    HISTORICAL_IDENTITY_FOUND = "HISTORICAL_IDENTITY_FOUND"
    CURRENT_IDENTITY_LIFECYCLE_MISMATCH = "CURRENT_IDENTITY_LIFECYCLE_MISMATCH"
    TICKER_RENAMED = "TICKER_RENAMED"
    RELISTED_SECURITY = "RELISTED_SECURITY"
    SECURITY_REISSUED = "SECURITY_REISSUED"
    INSTRUMENT_NOT_TRADING_AT_EVENT_TIME = "INSTRUMENT_NOT_TRADING_AT_EVENT_TIME"
    TINVEST_HISTORY_UNAVAILABLE = "TINVEST_HISTORY_UNAVAILABLE"
    TINVEST_INSTRUMENT_NOT_FOUND = "TINVEST_INSTRUMENT_NOT_FOUND"
    CLASS_CODE_MISMATCH = "CLASS_CODE_MISMATCH"
    NON_SUPPORTED_SECURITY_TYPE = "NON_SUPPORTED_SECURITY_TYPE"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    OTHER_FAIL_CLOSED = "OTHER_FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class CandleAvailabilityProbe:
    label: str
    instrument_uid: str
    interval: str
    date_from: str
    date_to: str
    candle_count: int
    complete_candle_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    rejected_reasons: tuple[str, ...]
    status: str

    def payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "instrument_uid": self.instrument_uid,
            "interval": self.interval,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "candle_count": self.candle_count,
            "complete_candle_count": self.complete_candle_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "rejected_reasons": list(self.rejected_reasons),
            "status": self.status,
        }


def security_history_safety_flags() -> dict[str, bool | str]:
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
        "SYNTHETIC_MARKET_DATA_USED": False,
        "PRICE_ADJUSTMENT_STATUS": "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES",
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_pr36_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("INPUT_DATASET_SHA") != PR36_INPUT_DATASET_SHA:
        raise ValueError("PR36_INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("OUTPUT_DATASET_SHA") != PR36_OUTPUT_DATASET_SHA:
        raise ValueError("PR36_OUTPUT_DATASET_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != PR36_ARTIFACT_SHA:
        raise ValueError("PR36_ARTIFACT_SHA_MISMATCH")
    if manifest.get("INPUT_NEW_EVENT_COHORT_SHA") != PR36_COHORT_SHA:
        raise ValueError("PR36_COHORT_SHA_MISMATCH")
    if manifest.get("EXACT_V3_PRESERVED") != "YES":
        raise ValueError("PR36_EXACT_V3_NOT_PRESERVED")
    if manifest.get("EXISTING_EVENT_ROWS_PRESERVED") != "PASS":
        raise ValueError("PR36_EVENT_PRESERVATION_NOT_PASS")
    if manifest.get("EXISTING_FEATURE_ROWS_PRESERVED") != "PASS":
        raise ValueError("PR36_FEATURE_PRESERVATION_NOT_PASS")
    for key in ("FUTURE_EVENT_HOLDOUT_USED", "FUTURE_EVENT_HOLDOUT_OBSERVED", "TEST_OUTCOME_USED"):
        if bool(manifest.get(key)):
            raise ValueError(f"PR36_{key}_NOT_SAFE")


def instrument_payload(instrument: TInvestInstrument | dict[str, Any]) -> dict[str, object]:
    if isinstance(instrument, TInvestInstrument):
        payload = instrument.payload()
    else:
        payload = {
            "ticker": str(instrument.get("ticker", "")).upper(),
            "class_code": str(instrument.get("class_code", "")),
            "instrument_uid": str(instrument.get("instrument_uid", "")),
            "figi": _optional_string(instrument.get("figi")),
            "instrument_type": str(instrument.get("instrument_type", "")),
            "first_1day_candle_date": _date_string(instrument.get("first_1day_candle_date")),
            "name": str(instrument.get("name", "")),
            "exchange": _optional_string(instrument.get("exchange")),
            "currency": _optional_string(instrument.get("currency")),
            "real_exchange": _optional_string(instrument.get("real_exchange")),
            "trading_status": _optional_string(instrument.get("trading_status")),
            "api_trade_available_flag": _optional_bool(instrument.get("api_trade_available_flag")),
            "buy_available_flag": _optional_bool(instrument.get("buy_available_flag")),
            "sell_available_flag": _optional_bool(instrument.get("sell_available_flag")),
            "last_1day_candle_date": _date_string(instrument.get("last_1day_candle_date")),
        }
    return {key: value for key, value in payload.items() if key != "resolved_at"}


def valid_at_event_time(
    instrument: dict[str, object] | None, publication_date: date
) -> bool | None:
    if instrument is None:
        return None
    first = instrument.get("first_1day_candle_date")
    last = instrument.get("last_1day_candle_date")
    first_date = date.fromisoformat(first) if isinstance(first, str) and first else None
    last_date = date.fromisoformat(last) if isinstance(last, str) and last else None
    if first_date is not None and first_date > publication_date:
        return False
    if last_date is not None and last_date < publication_date:
        return False
    return True


def daily_probe(
    *,
    label: str,
    instrument_uid: str,
    date_from: date,
    date_to: date,
    candles: tuple[TInvestDailyCandle, ...],
    rejected_reasons: tuple[str, ...] = (),
) -> CandleAvailabilityProbe:
    complete = [item for item in candles if item.is_complete]
    return CandleAvailabilityProbe(
        label=label,
        instrument_uid=instrument_uid,
        interval="1d",
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        candle_count=len(candles),
        complete_candle_count=len(complete),
        first_timestamp=min((item.trade_date.isoformat() for item in candles), default=None),
        last_timestamp=max((item.trade_date.isoformat() for item in candles), default=None),
        rejected_reasons=rejected_reasons,
        status="HISTORY_PRESENT" if candles else "EMPTY",
    )


def minute_probe(
    *,
    label: str,
    instrument_uid: str,
    date_from: datetime,
    date_to: datetime,
    candles: tuple[TInvestMinuteCandle, ...],
    rejected_reasons: tuple[str, ...] = (),
) -> CandleAvailabilityProbe:
    complete = [item for item in candles if item.is_complete]
    return CandleAvailabilityProbe(
        label=label,
        instrument_uid=instrument_uid,
        interval="1m",
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        candle_count=len(candles),
        complete_candle_count=len(complete),
        first_timestamp=min((item.begin_at.isoformat() for item in candles), default=None),
        last_timestamp=max((item.end_at.isoformat() for item in candles), default=None),
        rejected_reasons=rejected_reasons,
        status="HISTORY_PRESENT" if candles else "EMPTY",
    )


def probe_windows(publication: datetime) -> tuple[tuple[str, date, date], ...]:
    event_date = publication.date()
    result: list[tuple[str, date, date]] = []
    for offset in PROBE_OFFSETS_DAYS:
        start = event_date - timedelta(days=offset)
        result.append((f"d_minus_{offset}", start, min(event_date, start + timedelta(days=7))))
    return tuple(result)


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["ROOT_CAUSE"]) for row in rows)
    return {status.value: counter[status.value] for status in RootCauseStatus}


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _date_string(value: object) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value[:10]
    return None
