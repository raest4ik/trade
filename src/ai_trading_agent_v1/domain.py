from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_AGENT_PROPOSED_WEIGHT = 0.20


class AgentMode(StrEnum):
    READ_ONLY_RESEARCH = "READ_ONLY_RESEARCH"


class TradeAction(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    AVOID = "AVOID"


class ToolCapability(StrEnum):
    READ_MARKET = "READ_MARKET"
    READ_EVENTS = "READ_EVENTS"
    READ_PORTFOLIO = "READ_PORTFOLIO"
    READ_SYSTEM = "READ_SYSTEM"


class AgentDecisionStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    DEGRADED = "DEGRADED"
    DEGRADED_STALE_DATA = "DEGRADED_STALE_DATA"
    ABORTED_LIMIT = "ABORTED_LIMIT"
    MODEL_ERROR = "MODEL_ERROR"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"


class ProposalValidationStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    DEGRADED = "DEGRADED"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    id: str


class TradeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    action: TradeAction
    agent_confidence: float = Field(ge=0.0, le=1.0)
    target_weight: float = Field(ge=0.0)
    holding_horizon: str
    thesis: list[str] = Field(min_length=1)
    risks: list[str] = Field(min_length=1)
    evidence: list[EvidenceRef]
    data_quality: str


class StructuredAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    input_snapshot_as_of: dict[str, datetime]
    proposals: list[TradeProposal]


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TickerArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)


class RecentEventsArguments(TickerArguments):
    lookback_hours: int = Field(default=168, ge=1, le=168)
    limit: int = Field(default=5, ge=1)


def _empty_tool_calls() -> list[ToolCallRequest]:
    return []


class AgentModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCallRequest] = Field(default_factory=_empty_tool_calls)
    final_output: str | None = None


class AgentModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    system_prompt: str
    allowed_universe: list[dict[str, Any]]
    transcript: list[dict[str, Any]]
    safety_policy: dict[str, Any]


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    tool_name: str
    capability: ToolCapability
    arguments_hash: str
    result_hash: str | None
    status: str
    error: str | None = None


class ProposalValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ProposalValidationStatus
    reasons: list[str]
    stale_tickers: list[str] = Field(default_factory=list)


class SafetyCounters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    BROKER_MUTATIONS: int = 0
    REAL_ORDERS_SENT: int = 0
    PAPER_ORDERS_SENT: int = 0
    PORTFOLIO_MUTATIONS: int = 0
    LIVE_OUTCOMES_READ: int = 0
    LIVE_TARGETS_COMPUTED: int = 0
    LIVE_POST_EVENT_PRICE_READS: int = 0
    MODEL_TRAINING_PERFORMED: bool = False
    BACKTEST_PERFORMED: bool = False
    OLD_FUTURE_HOLDOUT_OPENED: bool = False
    PAID_SOURCE_CALLS: int = 0
    EXTERNAL_WEB_SEARCH_ENABLED: bool = False


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    code_sha: str
    agent_model_id: str
    prompt_version: str
    AGENT_MODE: AgentMode
    AGENT_DECISION_STATUS: AgentDecisionStatus
    AGENT_RESEARCH_CAPABILITY_READY: bool
    ML_V2_DATASET_STATUS: str
    universe: list[dict[str, Any]]
    portfolio_snapshot: dict[str, Any]
    market_context_snapshot: dict[str, Any]
    event_context_snapshot: dict[str, Any]
    research_status_snapshot: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    raw_model_output: str | None
    parsed_structured_output: StructuredAgentOutput | None
    validation: ProposalValidationResult
    final_proposals: list[TradeProposal]
    safety: SafetyCounters
