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
    age_seconds: float | None = None
    source: str = MARKET_SOURCE
    status: MarketQuoteStatus
    payload_sha: str | None = None
    reason: str | None = None


class FreshMarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market_adapter_id: str = MARKET_ADAPTER_ID
    market_source: str = MARKET_SOURCE
    market_fetch_started_at: datetime
    market_fetch_completed_at: datetime
    operation_as_of: datetime
    effective_max_age_seconds: float = Field(gt=0)
    quotes: list[dict[str, Any]]
    quote_audit: list[MarketQuoteAudit]
    source_payload_sha: str

    @property
    def quote_count(self) -> int:
        return len(self.quotes)

    def count(self, status: MarketQuoteStatus) -> int:
        return sum(row.status == status for row in self.quote_audit)

    def audit_payload(self) -> dict[str, Any]:
        return {
            "market_adapter_id": self.market_adapter_id,
            "market_source": self.market_source,
            "market_fetch_started_at": self.market_fetch_started_at.isoformat(),
            "market_fetch_completed_at": self.market_fetch_completed_at.isoformat(),
            "quote_count": self.quote_count,
            "fresh_quote_count": self.count(MarketQuoteStatus.FRESH),
            "stale_quote_count": self.count(MarketQuoteStatus.STALE),
            "missing_quote_count": self.count(MarketQuoteStatus.MISSING),
            "invalid_quote_count": self.count(MarketQuoteStatus.INVALID),
            "future_quote_count": self.count(MarketQuoteStatus.FUTURE),
            "effective_max_age_seconds": self.effective_max_age_seconds,
            "source_payload_sha": self.source_payload_sha,
            "quotes": [row.model_dump(mode="json") for row in self.quote_audit],
        }
