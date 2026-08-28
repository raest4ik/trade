from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START
from src.tinvest_market.client import TInvestDailyCandle, TInvestInstrument, TInvestMinuteCandle

ARTIFACT_VERSION = "chep-security-history-diagnostics-v1"
EXPECTED_INPUT_MATURATION_ARTIFACT_SHA = (
    "236ab1579cafda265eceeefc148b359d3ab2e5c54538d1d434bc789fc5775305"
)
EXPECTED_HISTORICAL_COHORT_SHA = "b7cff6dd3a94df7560da4565a0c06cca42dae5c1dd2ca0bc0cf736b54c10092e"
EXPECTED_FUTURE_METADATA_COHORT_SHA = (
    "ae1568923e517968020bf108ebf59b0371720f4567c21ee2fef0ea59e7be35c3"
)
EXPECTED_INSTRUMENT_IDENTITY_SHA = (
    "5f4945a81aca2d55916ccf379e03b8496d765da7229984d0014410af3c5d6c56"
)
EXPECTED_CHEP_HISTORICAL_EVENTS = 44
EXPECTED_CHEP_REACTION_READY = 0
EXPECTED_CHEP_FEATURE_READY = 0
EXPECTED_SECURITY_HISTORY_MISSING = 44
EXPECTED_CHEP_TICKER = "CHEP"
EXPECTED_CHEP_FIGI = "BBG000Q49F45"
EXPECTED_CHEP_CLASS_CODE = "TQBR"
EXPECTED_CHEP_UID = "b1f4f4fc-dac5-4e29-ae56-95fe441416ee"


class FutureHoldoutProbeError(RuntimeError):
    pass


class CandidateClassification(StrEnum):
    CURRENT_CONFIRMED = "CURRENT_CONFIRMED"
    HISTORICAL_CONFIRMED = "HISTORICAL_CONFIRMED"
    LEGACY_POSSIBLE = "LEGACY_POSSIBLE"
    UNRELATED = "UNRELATED"
    AMBIGUOUS = "AMBIGUOUS"


class PrimaryRootCause(StrEnum):
    CURRENT_IDENTITY_HAS_MINUTE_HISTORY = "CURRENT_IDENTITY_HAS_MINUTE_HISTORY"
    LEGACY_IDENTITY_HAS_MINUTE_HISTORY = "LEGACY_IDENTITY_HAS_MINUTE_HISTORY"
    TINVEST_MINUTE_HISTORY_UNAVAILABLE = "TINVEST_MINUTE_HISTORY_UNAVAILABLE"
    WRONG_INSTRUMENT_IDENTITY = "WRONG_INSTRUMENT_IDENTITY"
    REQUEST_IMPLEMENTATION_BUG = "REQUEST_IMPLEMENTATION_BUG"
    HISTORICAL_SECURITY_NOT_SUPPORTED = "HISTORICAL_SECURITY_NOT_SUPPORTED"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RecoveryFeasibility(StrEnum):
    RECOVERY_WITH_CURRENT_TINVEST_IDENTITY = "RECOVERY_WITH_CURRENT_TINVEST_IDENTITY"
    RECOVERY_WITH_VERIFIED_LEGACY_TINVEST_IDENTITY = (
        "RECOVERY_WITH_VERIFIED_LEGACY_TINVEST_IDENTITY"
    )
    RECOVERY_WITH_OFFICIAL_MOEX_MARKET_HISTORY = "RECOVERY_WITH_OFFICIAL_MOEX_MARKET_HISTORY"
    RECOVERY_REQUIRES_NEW_DATA_SOURCE = "RECOVERY_REQUIRES_NEW_DATA_SOURCE"
    NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES = "NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES"
    MORE_DIAGNOSTICS_REQUIRED = "MORE_DIAGNOSTICS_REQUIRED"


@dataclass(frozen=True, slots=True)
class ProbeWindow:
    label: str
    event_id: str
    publication_timestamp_utc: datetime
    minute_from: datetime
    minute_to: datetime
    daily_from: date
    daily_to: date

    def payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "event_id": self.event_id,
            "publication_timestamp_utc": self.publication_timestamp_utc.isoformat(),
            "minute_from": self.minute_from.isoformat(),
            "minute_to": self.minute_to.isoformat(),
            "daily_from": self.daily_from.isoformat(),
            "daily_to": self.daily_to.isoformat(),
        }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diagnostics_safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DIAGNOSTICS_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "FUTURE_CHEP_PRICE_LOOKUPS": 0,
        "FUTURE_CHEP_REACTIONS_COMPUTED": 0,
        "FUTURE_CHEP_TARGETS_COMPUTED": 0,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
        "REAL_TRADING_APPROVED": False,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "SYNTHETIC_MARKET_DATA_USED": False,
        "LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE": False,
    }


def require_maturation_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "ARTIFACT_SHA": EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
        "HISTORICAL_COHORT_SHA": EXPECTED_HISTORICAL_COHORT_SHA,
        "FUTURE_METADATA_COHORT_SHA": EXPECTED_FUTURE_METADATA_COHORT_SHA,
        "INSTRUMENT_IDENTITY_SHA": EXPECTED_INSTRUMENT_IDENTITY_SHA,
        "CHEP_HISTORICAL_EVENTS_TOTAL": EXPECTED_CHEP_HISTORICAL_EVENTS,
        "CHEP_REACTION_READY": EXPECTED_CHEP_REACTION_READY,
        "CHEP_FEATURE_READY": EXPECTED_CHEP_FEATURE_READY,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_MATURATION_{key}_MISMATCH")
    blockers = manifest.get("BLOCKER_COUNTS")
    if not isinstance(blockers, dict):
        raise ValueError("INPUT_MATURATION_BLOCKER_COUNTS_INVALID")
    typed_blockers = cast("dict[str, object]", blockers)
    if typed_blockers.get("SECURITY_HISTORY_MISSING") != EXPECTED_SECURITY_HISTORY_MISSING:
        raise ValueError("INPUT_MATURATION_SECURITY_HISTORY_MISSING_MISMATCH")
    for key in (
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_MATURATION_{key}_NOT_SAFE")


def instrument_payload(instrument: TInvestInstrument | dict[str, Any]) -> dict[str, object]:
    if isinstance(instrument, TInvestInstrument):
        payload = instrument.payload()
    else:
        payload = {
            "ticker": str(instrument.get("ticker", "")).upper(),
            "class_code": _optional_string(instrument.get("class_code")) or "",
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
    return dict(sorted(payload.items()))


def classify_candidate(
    candidate: dict[str, object],
    *,
    expected_ticker: str = EXPECTED_CHEP_TICKER,
    expected_figi: str = EXPECTED_CHEP_FIGI,
    expected_uid: str = EXPECTED_CHEP_UID,
    expected_class_code: str = EXPECTED_CHEP_CLASS_CODE,
) -> CandidateClassification:
    ticker = str(candidate.get("ticker") or "").upper()
    figi = str(candidate.get("figi") or "")
    uid = str(candidate.get("instrument_uid") or "")
    class_code = str(candidate.get("class_code") or "").upper()
    instrument_type = str(candidate.get("instrument_type") or "").upper()
    name = str(candidate.get("name") or "").upper()
    same_uid = uid == expected_uid
    same_figi = figi == expected_figi
    same_ticker = ticker == expected_ticker
    same_board = class_code == expected_class_code
    share_like = "SHARE" in instrument_type or instrument_type in {"", "UNKNOWN"}
    if same_uid and same_figi and same_ticker and same_board:
        return CandidateClassification.CURRENT_CONFIRMED
    if same_figi and same_ticker and same_board and share_like:
        return CandidateClassification.HISTORICAL_CONFIRMED
    if same_uid and (not same_figi or not same_ticker or not same_board):
        return CandidateClassification.AMBIGUOUS
    if same_ticker and same_board and share_like:
        return CandidateClassification.LEGACY_POSSIBLE
    if expected_ticker in name and same_board and share_like:
        return CandidateClassification.LEGACY_POSSIBLE
    if same_ticker or same_figi:
        return CandidateClassification.AMBIGUOUS
    return CandidateClassification.UNRELATED


def build_probe_windows(rows: list[dict[str, Any]]) -> tuple[ProbeWindow, ...]:
    historical = sorted(rows, key=lambda row: (_published_at(row), _event_id(row)))
    for row in historical:
        guard_no_future_probe(_published_at(row))
    if not historical:
        return ()
    indexes = [0, len(historical) // 2, len(historical) - 1]
    labels = ["earliest_historical_event", "median_historical_event", "latest_historical_event"]
    windows: list[ProbeWindow] = []
    seen: set[str] = set()
    for label, index in zip(labels, indexes, strict=True):
        row = historical[index]
        event_id = _event_id(row)
        if event_id in seen:
            continue
        seen.add(event_id)
        published = _published_at(row)
        windows.append(
            ProbeWindow(
                label=label,
                event_id=event_id,
                publication_timestamp_utc=published,
                minute_from=published - timedelta(minutes=10),
                minute_to=published + timedelta(minutes=70),
                daily_from=published.date() - timedelta(days=2),
                daily_to=published.date() + timedelta(days=2),
            )
        )
    return tuple(windows)


def guard_no_future_probe(publication_timestamp_utc: datetime) -> None:
    if publication_timestamp_utc.astimezone(UTC).date() >= FUTURE_EVENT_HOLDOUT_START:
        raise FutureHoldoutProbeError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def minute_probe_payload(
    *,
    source: str,
    label: str,
    instrument_uid: str,
    figi: str | None,
    class_code: str | None,
    date_from: datetime,
    date_to: datetime,
    candles: tuple[TInvestMinuteCandle, ...],
    rejected_reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    complete = [item for item in candles if item.is_complete]
    return {
        "source": source,
        "label": label,
        "requested_identity": {
            "ticker": EXPECTED_CHEP_TICKER,
            "figi": figi,
            "instrument_uid": instrument_uid,
            "class_code": class_code,
        },
        "interval": "1m",
        "from": date_from.astimezone(UTC).isoformat(),
        "to": date_to.astimezone(UTC).isoformat(),
        "returned_candle_count": len(candles),
        "complete_candle_count": len(complete),
        "first_returned_timestamp": min(
            (item.begin_at.astimezone(UTC).isoformat() for item in candles), default=None
        ),
        "last_returned_timestamp": max(
            (item.end_at.astimezone(UTC).isoformat() for item in candles), default=None
        ),
        "api_status": "PASS" if not rejected_reasons else "BLOCKED",
        "api_error": rejected_reasons[0] if rejected_reasons else None,
        "rejected_reasons": list(rejected_reasons),
    }


def daily_probe_payload(
    *,
    source: str,
    label: str,
    instrument_uid: str,
    figi: str | None,
    class_code: str | None,
    date_from: date,
    date_to: date,
    candles: tuple[TInvestDailyCandle, ...],
    rejected_reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    complete = [item for item in candles if item.is_complete]
    return {
        "source": source,
        "label": label,
        "requested_identity": {
            "ticker": EXPECTED_CHEP_TICKER,
            "figi": figi,
            "instrument_uid": instrument_uid,
            "class_code": class_code,
        },
        "interval": "1d",
        "from": date_from.isoformat(),
        "to": date_to.isoformat(),
        "returned_candle_count": len(candles),
        "complete_candle_count": len(complete),
        "first_returned_timestamp": min(
            (item.trade_date.isoformat() for item in candles), default=None
        ),
        "last_returned_timestamp": max(
            (item.trade_date.isoformat() for item in candles), default=None
        ),
        "api_status": "PASS" if not rejected_reasons else "BLOCKED",
        "api_error": rejected_reasons[0] if rejected_reasons else None,
        "rejected_reasons": list(rejected_reasons),
    }


def choose_primary_root_cause(
    *,
    candidate_rows: list[dict[str, Any]],
    candle_probes: list[dict[str, Any]],
    local_acquisition_logic_root_cause: bool,
    moex_event_date_trading_confirmed: bool | None,
) -> PrimaryRootCause:
    if local_acquisition_logic_root_cause:
        return PrimaryRootCause.REQUEST_IMPLEMENTATION_BUG
    current_uids = {
        str(row["instrument_uid"])
        for row in candidate_rows
        if row["classification"] == CandidateClassification.CURRENT_CONFIRMED
    }
    legacy_uids = {
        str(row["instrument_uid"])
        for row in candidate_rows
        if row["classification"]
        in {
            CandidateClassification.HISTORICAL_CONFIRMED,
            CandidateClassification.LEGACY_POSSIBLE,
        }
    }
    ambiguous_count = sum(
        row["classification"] == CandidateClassification.AMBIGUOUS for row in candidate_rows
    )
    minute_with_data = _probe_uids_with_data(candle_probes, "1m")
    if current_uids & minute_with_data:
        return PrimaryRootCause.CURRENT_IDENTITY_HAS_MINUTE_HISTORY
    if legacy_uids & minute_with_data:
        return PrimaryRootCause.LEGACY_IDENTITY_HAS_MINUTE_HISTORY
    if not current_uids and ambiguous_count:
        return PrimaryRootCause.IDENTITY_AMBIGUOUS
    if not current_uids and candidate_rows:
        return PrimaryRootCause.WRONG_INSTRUMENT_IDENTITY
    daily_event_present = any(
        str(row.get("source")) == "TINVEST_READONLY"
        and str(row.get("interval")) == "1d"
        and not str(row.get("label", "")).startswith("known_last_daily_window")
        and int(row.get("returned_candle_count", 0)) > 0
        for row in candle_probes
    )
    if daily_event_present:
        return PrimaryRootCause.TINVEST_MINUTE_HISTORY_UNAVAILABLE
    current = next(
        (
            row
            for row in candidate_rows
            if row["classification"] == CandidateClassification.CURRENT_CONFIRMED
        ),
        None,
    )
    if current is not None:
        last_daily = _date_string(current.get("last_1day_candle_date"))
        inactive = (
            current.get("api_trade_available_flag") is False
            or current.get("buy_available_flag") is False
            or current.get("sell_available_flag") is False
            or str(current.get("trading_status") or "").endswith("NOT_AVAILABLE_FOR_TRADING")
        )
        if inactive or moex_event_date_trading_confirmed is False or last_daily is not None:
            return PrimaryRootCause.HISTORICAL_SECURITY_NOT_SUPPORTED
    if ambiguous_count:
        return PrimaryRootCause.IDENTITY_AMBIGUOUS
    return PrimaryRootCause.INSUFFICIENT_EVIDENCE


def choose_recovery_feasibility(
    *,
    primary_root_cause: PrimaryRootCause,
    moex_event_date_trading_confirmed: bool | None,
    moex_minute_history_evidence: bool,
) -> RecoveryFeasibility:
    if primary_root_cause == PrimaryRootCause.CURRENT_IDENTITY_HAS_MINUTE_HISTORY:
        return RecoveryFeasibility.RECOVERY_WITH_CURRENT_TINVEST_IDENTITY
    if primary_root_cause == PrimaryRootCause.LEGACY_IDENTITY_HAS_MINUTE_HISTORY:
        return RecoveryFeasibility.RECOVERY_WITH_VERIFIED_LEGACY_TINVEST_IDENTITY
    if moex_event_date_trading_confirmed and moex_minute_history_evidence:
        return RecoveryFeasibility.RECOVERY_WITH_OFFICIAL_MOEX_MARKET_HISTORY
    if primary_root_cause == PrimaryRootCause.TINVEST_MINUTE_HISTORY_UNAVAILABLE:
        return RecoveryFeasibility.RECOVERY_REQUIRES_NEW_DATA_SOURCE
    if primary_root_cause == PrimaryRootCause.HISTORICAL_SECURITY_NOT_SUPPORTED:
        return RecoveryFeasibility.NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES
    return RecoveryFeasibility.MORE_DIAGNOSTICS_REQUIRED


def probe_metrics(candle_probes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "MINUTE_PROBES_ATTEMPTED": sum(
            row.get("source") == "TINVEST_READONLY" and row.get("interval") == "1m"
            for row in candle_probes
        ),
        "MINUTE_PROBES_WITH_DATA": sum(
            row.get("source") == "TINVEST_READONLY"
            and row.get("interval") == "1m"
            and int(row.get("returned_candle_count", 0)) > 0
            for row in candle_probes
        ),
        "DAILY_PROBES_ATTEMPTED": sum(
            row.get("source") == "TINVEST_READONLY" and row.get("interval") == "1d"
            for row in candle_probes
        ),
        "DAILY_PROBES_WITH_DATA": sum(
            row.get("source") == "TINVEST_READONLY"
            and row.get("interval") == "1d"
            and int(row.get("returned_candle_count", 0)) > 0
            for row in candle_probes
        ),
    }


def candidate_metrics(candidate_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["instrument_uid"]) for row in candidate_rows)
    confirmed = {
        str(row["instrument_uid"])
        for row in candidate_rows
        if row["classification"]
        in {
            CandidateClassification.CURRENT_CONFIRMED,
            CandidateClassification.HISTORICAL_CONFIRMED,
        }
    }
    return {
        "TINVEST_IDENTITIES_FOUND": len(counter),
        "TINVEST_IDENTITIES_CONFIRMED_SAME_SECURITY": len(confirmed),
    }


def _probe_uids_with_data(candle_probes: list[dict[str, Any]], interval: str) -> set[str]:
    result: set[str] = set()
    for row in candle_probes:
        if row.get("interval") != interval or int(row.get("returned_candle_count", 0)) <= 0:
            continue
        identity = row.get("requested_identity")
        if isinstance(identity, dict):
            typed_identity = cast("dict[str, object]", identity)
            result.add(str(typed_identity.get("instrument_uid")))
    return result


def _published_at(row: dict[str, Any]) -> datetime:
    value = row.get("publication_timestamp_utc")
    if value is None and isinstance(row.get("metadata"), dict):
        metadata = cast("dict[str, object]", row["metadata"])
        value = metadata.get("publication_timestamp_utc")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _event_id(row: dict[str, Any]) -> str:
    if "event_id" in row:
        return str(row["event_id"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        typed_metadata = cast("dict[str, object]", metadata)
        return str(typed_metadata["event_id"])
    raise ValueError("EVENT_ID_MISSING")


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
