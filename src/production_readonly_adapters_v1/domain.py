from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ADAPTER_POLICY_VERSION = "production-readonly-adapters-policy-v1"
MARKET_ADAPTER_ID = "moex-iss-fresh-market-v1"
MARKET_SOURCE = "MOEX_ISS_PUBLIC"
AGENT_ADAPTER_ID = "ollama-readonly-agent-v1"


class MarketQuoteStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    FUTURE = "FUTURE"


class MarketQuoteAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    board: str
    market_data_as_of: datetime | None = None
    fetched_at: datetime | None = None
    age_seconds: float | None = None
    source: str = MARKET_SOURCE
    status: MarketQuoteStatus
    timestamp_status: MarketQuoteStatus | None = None
    book_status: str = "VALID"
    book_reason: str | None = None
    payload_sha: str | None = None
    reason: str | None = None


class RawMarketQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    board: str
    last_price: float = Field(gt=0)
    bid: float | None = Field(default=None, gt=0)
    ask: float | None = Field(default=None, gt=0)
    lot_size: int = Field(gt=0)
    market_data_as_of: datetime
    source: str = MARKET_SOURCE
    fetched_at: datetime
    source_payload_sha: str
    book_quality_reason: str | None = None


class RawMarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market_adapter_id: str = MARKET_ADAPTER_ID
    market_source: str = MARKET_SOURCE
    market_fetch_started_at: datetime
    market_fetch_completed_at: datetime
    quotes: tuple[RawMarketQuote, ...]
    quote_audit: tuple[MarketQuoteAudit, ...]
    source_payload_sha: str


class FreshMarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market_adapter_id: str = MARKET_ADAPTER_ID
    market_source: str = MARKET_SOURCE
    market_fetch_started_at: datetime
    market_fetch_completed_at: datetime
    operation_as_of: datetime
    max_market_source_time: datetime | None = None
    market_source_clock_delta_seconds: float | None = None
    effective_max_age_seconds: float = Field(gt=0)
    quotes: list[dict[str, Any]]
    quote_audit: list[MarketQuoteAudit]
    source_payload_sha: str

    @property
    def quote_count(self) -> int:
        return len(self.quotes)

    def count(self, status: MarketQuoteStatus) -> int:
        return sum(
            row.status == status or row.timestamp_status == status for row in self.quote_audit
        )

    def audit_payload(self) -> dict[str, Any]:
        quote_audit: list[dict[str, Any]] = []
        for row in self.quote_audit:
            payload = row.model_dump(mode="json")
            payload["source_time"] = payload["market_data_as_of"]
            payload["age_at_decision_seconds"] = payload["age_seconds"]
            quote_audit.append(payload)
        return {
            "market_adapter_id": self.market_adapter_id,
            "market_source": self.market_source,
            "market_fetch_started_at": self.market_fetch_started_at.isoformat(),
            "market_fetch_completed_at": self.market_fetch_completed_at.isoformat(),
            "decision_as_of": self.operation_as_of.isoformat(),
            "max_market_source_time": (
                None
                if self.max_market_source_time is None
                else self.max_market_source_time.isoformat()
            ),
            "market_source_clock_delta_seconds": self.market_source_clock_delta_seconds,
            "quote_count": self.quote_count,
            "fresh_quote_count": self.count(MarketQuoteStatus.FRESH),
            "stale_quote_count": self.count(MarketQuoteStatus.STALE),
            "missing_quote_count": self.count(MarketQuoteStatus.MISSING),
            "invalid_quote_count": self.count(MarketQuoteStatus.INVALID),
            "future_quote_count": self.count(MarketQuoteStatus.FUTURE),
            "effective_max_age_seconds": self.effective_max_age_seconds,
            "source_payload_sha": self.source_payload_sha,
            "quotes": quote_audit,
        }
