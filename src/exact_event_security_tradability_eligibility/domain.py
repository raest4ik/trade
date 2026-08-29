from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, cast

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START

ARTIFACT_VERSION = "exact-event-security-tradability-eligibility-v1"
EXPECTED_INPUT_DIAGNOSTIC_ARTIFACT_SHA = (
    "b31fca68eccde1aa009f0a992130f8afb4ce8281cf7db0a7808eaebc81740497"
)
EXPECTED_INPUT_MATURATION_ARTIFACT_SHA = (
    "236ab1579cafda265eceeefc148b359d3ab2e5c54538d1d434bc789fc5775305"
)
EXPECTED_CHEP_HISTORICAL_EVENTS = 44
EXPECTED_FUTURE_CHEP_EVENTS = 6
EXPECTED_CHEP_TICKER = "CHEP"
EXPECTED_CHEP_UID = "b1f4f4fc-dac5-4e29-ae56-95fe441416ee"
EXPECTED_CHEP_FIGI = "BBG000Q49F45"
EXPECTED_CHEP_CLASS_CODE = "TQBR"


class EventValidity(StrEnum):
    VALID_EXACT_EVENT = "VALID_EXACT_EVENT"
    INVALID_EVENT = "INVALID_EVENT"


class InstrumentIdentityStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_EVALUATED_FUTURE_HOLDOUT = "NOT_EVALUATED_FUTURE_HOLDOUT"


class MarketReactionEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    SECURITY_NOT_TRADING_AT_EVENT_TIME = "SECURITY_NOT_TRADING_AT_EVENT_TIME"
    SECURITY_HISTORY_UNAVAILABLE = "SECURITY_HISTORY_UNAVAILABLE"
    INSTRUMENT_IDENTITY_UNRESOLVED = "INSTRUMENT_IDENTITY_UNRESOLVED"
    INSTRUMENT_IDENTITY_AMBIGUOUS = "INSTRUMENT_IDENTITY_AMBIGUOUS"
    TRADING_STATUS_UNVERIFIABLE = "TRADING_STATUS_UNVERIFIABLE"
    FUTURE_METADATA_ONLY = "FUTURE_METADATA_ONLY"


class CollectionDecision(StrEnum):
    KEEP_METADATA_ONLY = "KEEP_METADATA_ONLY"
    DISABLE_FROM_TRADING_RESEARCH_COLLECTION = "DISABLE_FROM_TRADING_RESEARCH_COLLECTION"


class FinalDecision(StrEnum):
    SOURCE_BREADTH_EXPANSION_NEXT = "SOURCE_BREADTH_EXPANSION_NEXT"
    LIVE_EXACT_ACCUMULATION_NEXT = "LIVE_EXACT_ACCUMULATION_NEXT"
    TRADABILITY_GATE_REVEALS_BROADER_DATA_QUALITY_ISSUE = (
        "TRADABILITY_GATE_REVEALS_BROADER_DATA_QUALITY_ISSUE"
    )
    ELIGIBILITY_POLICY_REVIEW_REQUIRED = "ELIGIBILITY_POLICY_REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class EligibilityPolicy:
    policy_version: str = "exact-event-security-tradability-policy-v1"
    event_validity_statuses: tuple[str, ...] = tuple(status.value for status in EventValidity)
    identity_statuses: tuple[str, ...] = tuple(status.value for status in InstrumentIdentityStatus)
    eligibility_statuses: tuple[str, ...] = tuple(
        status.value for status in MarketReactionEligibility
    )
    positive_non_trading_evidence_required: bool = True
    empty_candles_alone_prove_non_trading: bool = False
    future_event_holdout_start: str = FUTURE_EVENT_HOLDOUT_START.isoformat()
    pipeline_position: str = (
        "OFFICIAL_EVENT -> EXACT_TIMESTAMP -> CANONICAL_INSTRUMENT_MAPPING -> "
        "SECURITY_TRADABILITY_ELIGIBILITY -> MARKET_HISTORY_ONLY_IF_ELIGIBLE"
    )
    chep_collection_decision: str = CollectionDecision.KEEP_METADATA_ONLY.value
    data_cost_rub: int = 0

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TradingEvidence:
    ticker: str
    instrument_uid: str | None
    figi: str | None
    class_code: str | None
    source: str
    security_history_confirmed: bool
    event_date_trading_confirmed: bool | None
    last_confirmed_trading_date: date | None
    current_trading_status: str | None
    api_trade_available: bool | None
    buy_available: bool | None
    sell_available: bool | None
    evidence_detail: str

    def payload(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "class_code": self.class_code,
            "source": self.source,
            "security_history_confirmed": self.security_history_confirmed,
            "event_date_trading_confirmed": self.event_date_trading_confirmed,
            "last_confirmed_trading_date": (
                self.last_confirmed_trading_date.isoformat()
                if self.last_confirmed_trading_date is not None
                else None
            ),
            "current_trading_status": self.current_trading_status,
            "api_trade_available": self.api_trade_available,
            "buy_available": self.buy_available,
            "sell_available": self.sell_available,
            "evidence_detail": self.evidence_detail,
        }


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    event_id: str
    ticker: str
    published_at_utc: datetime
    event_validity: EventValidity
    instrument_identity_status: InstrumentIdentityStatus
    market_reaction_eligibility: MarketReactionEligibility
    eligibility_evidence: tuple[str, ...]
    primary_blocker: str | None
    reaction_attempt_skipped: bool
    feature_attempt_skipped: bool
    market_history_request_avoided: bool

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "published_at_utc": self.published_at_utc.astimezone(UTC).isoformat(),
            "event_validity": self.event_validity.value,
            "instrument_identity_status": self.instrument_identity_status.value,
            "market_reaction_eligibility": self.market_reaction_eligibility.value,
            "eligibility_evidence": list(self.eligibility_evidence),
            "primary_blocker": self.primary_blocker,
            "reaction_attempt_skipped": self.reaction_attempt_skipped,
            "feature_attempt_skipped": self.feature_attempt_skipped,
            "market_history_request_avoided": self.market_history_request_avoided,
            "future_outcome_fields_exposed": False,
        }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def eligibility_safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_QUALITY_GATE_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "FUTURE_CHEP_PRICE_LOOKUPS": 0,
        "FUTURE_CHEP_REACTION_ATTEMPTS": 0,
        "FUTURE_CHEP_TARGET_ATTEMPTS": 0,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
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
    }


def require_diagnostic_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "ARTIFACT_SHA": EXPECTED_INPUT_DIAGNOSTIC_ARTIFACT_SHA,
        "INPUT_MATURATION_ARTIFACT_SHA": EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
        "PRIMARY_ROOT_CAUSE": "HISTORICAL_SECURITY_NOT_SUPPORTED",
        "RECOVERY_FEASIBILITY": "NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_DIAGNOSTIC_{key}_MISMATCH")
    for key in (
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_DIAGNOSTIC_{key}_NOT_SAFE")


def evaluate_event_eligibility(
    *,
    event_id: str,
    ticker: str,
    published_at_utc: datetime,
    identity_status: InstrumentIdentityStatus,
    evidence: TradingEvidence | None,
    event_validity: EventValidity = EventValidity.VALID_EXACT_EVENT,
) -> EligibilityResult:
    published = published_at_utc.astimezone(UTC)
    if published.date() >= FUTURE_EVENT_HOLDOUT_START:
        return _blocked_result(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            identity_status=InstrumentIdentityStatus.NOT_EVALUATED_FUTURE_HOLDOUT,
            eligibility=MarketReactionEligibility.FUTURE_METADATA_ONLY,
            evidence_ids=("FUTURE_EVENT_HOLDOUT_START=2026-08-11",),
        )
    if identity_status == InstrumentIdentityStatus.UNRESOLVED:
        return _blocked_result(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            identity_status=identity_status,
            eligibility=MarketReactionEligibility.INSTRUMENT_IDENTITY_UNRESOLVED,
            evidence_ids=("identity_status=UNRESOLVED",),
        )
    if identity_status == InstrumentIdentityStatus.AMBIGUOUS:
        return _blocked_result(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            identity_status=identity_status,
            eligibility=MarketReactionEligibility.INSTRUMENT_IDENTITY_AMBIGUOUS,
            evidence_ids=("identity_status=AMBIGUOUS",),
        )
    if evidence is None:
        return _blocked_result(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            identity_status=identity_status,
            eligibility=MarketReactionEligibility.TRADING_STATUS_UNVERIFIABLE,
            evidence_ids=("no_positive_trading_evidence",),
        )
    evidence_hash = sha256_payload(evidence.payload())
    if _positive_non_trading_evidence(evidence, published.date()):
        return _blocked_result(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            identity_status=identity_status,
            eligibility=MarketReactionEligibility.SECURITY_NOT_TRADING_AT_EVENT_TIME,
            evidence_ids=(evidence_hash,),
        )
    if evidence.event_date_trading_confirmed is True:
        return EligibilityResult(
            event_id=event_id,
            ticker=ticker,
            published_at_utc=published,
            event_validity=event_validity,
            instrument_identity_status=identity_status,
            market_reaction_eligibility=MarketReactionEligibility.ELIGIBLE,
            eligibility_evidence=(evidence_hash,),
            primary_blocker=None,
            reaction_attempt_skipped=False,
            feature_attempt_skipped=False,
            market_history_request_avoided=False,
        )
    return _blocked_result(
        event_id=event_id,
        ticker=ticker,
        published_at_utc=published,
        event_validity=event_validity,
        identity_status=identity_status,
        eligibility=MarketReactionEligibility.SECURITY_HISTORY_UNAVAILABLE,
        evidence_ids=(evidence_hash,),
    )


def should_attempt_market_maturation(result: EligibilityResult) -> bool:
    return result.market_reaction_eligibility == MarketReactionEligibility.ELIGIBLE


def collection_decision_for_ticker(results: list[EligibilityResult]) -> CollectionDecision:
    if results and all(
        result.market_reaction_eligibility
        in {
            MarketReactionEligibility.SECURITY_NOT_TRADING_AT_EVENT_TIME,
            MarketReactionEligibility.FUTURE_METADATA_ONLY,
        }
        for result in results
    ):
        return CollectionDecision.KEEP_METADATA_ONLY
    return CollectionDecision.KEEP_METADATA_ONLY


def result_counts(results: list[EligibilityResult]) -> dict[str, int]:
    payloads = [result.payload() for result in results]
    return {
        "EVENTS_CHECKED": len(results),
        "EVENTS_MARKET_ELIGIBLE": sum(
            row["market_reaction_eligibility"] == MarketReactionEligibility.ELIGIBLE
            for row in payloads
        ),
        "EVENTS_MARKET_INELIGIBLE": sum(
            row["market_reaction_eligibility"]
            not in {
                MarketReactionEligibility.ELIGIBLE,
                MarketReactionEligibility.FUTURE_METADATA_ONLY,
            }
            for row in payloads
        ),
        "SECURITY_NOT_TRADING_COUNT": sum(
            row["market_reaction_eligibility"]
            == MarketReactionEligibility.SECURITY_NOT_TRADING_AT_EVENT_TIME
            for row in payloads
        ),
        "SECURITY_HISTORY_UNAVAILABLE_COUNT": sum(
            row["market_reaction_eligibility"]
            == MarketReactionEligibility.SECURITY_HISTORY_UNAVAILABLE
            for row in payloads
        ),
        "IDENTITY_UNRESOLVED_COUNT": sum(
            row["market_reaction_eligibility"]
            == MarketReactionEligibility.INSTRUMENT_IDENTITY_UNRESOLVED
            for row in payloads
        ),
        "IDENTITY_AMBIGUOUS_COUNT": sum(
            row["market_reaction_eligibility"]
            == MarketReactionEligibility.INSTRUMENT_IDENTITY_AMBIGUOUS
            for row in payloads
        ),
        "REACTION_ATTEMPTS_AVOIDED": sum(bool(row["reaction_attempt_skipped"]) for row in payloads),
        "MARKET_HISTORY_REQUESTS_AVOIDED": sum(
            bool(row["market_history_request_avoided"]) for row in payloads
        ),
    }


def _positive_non_trading_evidence(evidence: TradingEvidence, event_date: date) -> bool:
    inactive = (
        evidence.current_trading_status == "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING"
        or evidence.api_trade_available is False
        or evidence.buy_available is False
        or evidence.sell_available is False
    )
    historical_before_event = (
        evidence.security_history_confirmed
        and evidence.last_confirmed_trading_date is not None
        and evidence.last_confirmed_trading_date < event_date
    )
    exchange_confirms_no_event_trade = evidence.event_date_trading_confirmed is False
    return inactive and historical_before_event and exchange_confirms_no_event_trade


def _blocked_result(
    *,
    event_id: str,
    ticker: str,
    published_at_utc: datetime,
    event_validity: EventValidity,
    identity_status: InstrumentIdentityStatus,
    eligibility: MarketReactionEligibility,
    evidence_ids: tuple[str, ...],
) -> EligibilityResult:
    return EligibilityResult(
        event_id=event_id,
        ticker=ticker,
        published_at_utc=published_at_utc,
        event_validity=event_validity,
        instrument_identity_status=identity_status,
        market_reaction_eligibility=eligibility,
        eligibility_evidence=evidence_ids,
        primary_blocker=None
        if eligibility == MarketReactionEligibility.FUTURE_METADATA_ONLY
        else eligibility.value,
        reaction_attempt_skipped=eligibility != MarketReactionEligibility.FUTURE_METADATA_ONLY,
        feature_attempt_skipped=eligibility != MarketReactionEligibility.FUTURE_METADATA_ONLY,
        market_history_request_avoided=eligibility
        not in {
            MarketReactionEligibility.ELIGIBLE,
            MarketReactionEligibility.FUTURE_METADATA_ONLY,
        },
    )


def parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def event_id(row: dict[str, Any]) -> str:
    if "event_id" in row:
        return str(row["event_id"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(cast("dict[str, object]", metadata)["event_id"])
    raise ValueError("EVENT_ID_MISSING")
