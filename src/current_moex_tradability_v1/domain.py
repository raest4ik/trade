from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

TRADABILITY_POLICY_VERSION = "current-moex-tradability-policy-v1"
TRADABILITY_SOURCE = "MOEX_ISS_CURRENT_TQBR_BOARD"
INTENDED_BOARD = "TQBR"


class CurrentTradabilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_NO_SECURITY_ROW = "INELIGIBLE_NO_SECURITY_ROW"
    INELIGIBLE_WRONG_BOARD = "INELIGIBLE_WRONG_BOARD"
    INELIGIBLE_TRADING_DISABLED = "INELIGIBLE_TRADING_DISABLED"
    INELIGIBLE_INVALID_LOT = "INELIGIBLE_INVALID_LOT"
    INELIGIBLE_NO_CURRENT_MARKET_DATA = "INELIGIBLE_NO_CURRENT_MARKET_DATA"
    UNKNOWN = "UNKNOWN"


class CurrentMoexState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    board: str | None = None
    short_name: str | None = None
    lot_size: int | None = None
    security_status: str | None = None
    trading_status: str | None = None
    has_marketdata_row: bool = False


class CandidateClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    canonical_status: str = "CANONICAL_INSTRUMENT"
    current_moex_status: CurrentTradabilityStatus
    board: str | None = None
    lot_size: int | None = None
    security_status: str | None = None
    trading_status: str | None = None
    candidate_eligible: bool
    reason: str
    source: str = TRADABILITY_SOURCE
    fetched_at: datetime
    payload_sha: str


class CurrentUniverseResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = TRADABILITY_POLICY_VERSION
    source: str = TRADABILITY_SOURCE
    fetched_at: datetime
    source_time: datetime | None = None
    payload_sha: str
    classifications: tuple[CandidateClassification, ...]

    def by_ticker(self) -> dict[str, CandidateClassification]:
        return {row.ticker: row for row in self.classifications}

    def audit_payload(self) -> dict[str, object]:
        eligible = [row.ticker for row in self.classifications if row.candidate_eligible]
        rejected = [
            {
                "ticker": row.ticker,
                "status": row.current_moex_status.value,
                "reason": row.reason,
            }
            for row in self.classifications
            if not row.candidate_eligible
        ]
        rejection_reasons = Counter(row["reason"] for row in rejected)
        return {
            "current_eligibility_policy_version": self.policy_version,
            "current_eligibility_source": self.source,
            "current_eligibility_fetched_at": self.fetched_at.isoformat(),
            "current_eligibility_source_time": (
                None if self.source_time is None else self.source_time.isoformat()
            ),
            "current_eligibility_snapshot_sha": self.payload_sha,
            "eligible_count": len(eligible),
            "ineligible_count": sum(
                row.current_moex_status != CurrentTradabilityStatus.UNKNOWN
                and not row.candidate_eligible
                for row in self.classifications
            ),
            "unknown_count": sum(
                row.current_moex_status == CurrentTradabilityStatus.UNKNOWN
                for row in self.classifications
            ),
            "eligible_tickers": eligible,
            "rejected_candidates": rejected,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }


class TickerMigrationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    legacy_ticker: str
    current_ticker: str
    relation: str
    legacy_instrument_identity: str
    current_instrument_identity: str
    silent_substitution_allowed: bool = False
    sources: tuple[str, ...] = Field(min_length=1)
