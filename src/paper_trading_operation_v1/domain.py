from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPERATION_POLICY_VERSION = "paper-trading-operation-policy-v1"
OPERATION_ID_NAMESPACE = "paper-trading-operation-slot-v1"


def _dict_list() -> list[dict[str, Any]]:
    return []


def _dict() -> dict[str, Any]:
    return {}


class PaperOperationMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    PAPER_EXECUTE = "PAPER_EXECUTE"


class PaperOperationStatus(StrEnum):
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    NO_ACTION = "NO_ACTION"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    ALREADY_PROCESSED = "ALREADY_PROCESSED"


class PaperOperationStepName(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    CONTEXT = "CONTEXT"
    AGENT = "AGENT"
    RISK = "RISK"
    PAPER_EXECUTION = "PAPER_EXECUTION"
    REPLAY_VERIFY = "REPLAY_VERIFY"
    AUDIT = "AUDIT"


class PaperOperationStepStatus(StrEnum):
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class OperationAuditRecordType(StrEnum):
    PREPARED = "PREPARED"
    COMPLETED = "COMPLETED"
    RECOVERED = "RECOVERED"


class PaperOperationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = OPERATION_POLICY_VERSION
    max_operation_universe: int = Field(default=10, ge=1, le=10)
    max_agent_steps: int = Field(default=6, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    require_research_ready: bool = True
    require_operational_burnin_pass: bool = True
    require_source_failure_isolation: bool = True
    require_portfolio_replay_pass: bool = True
    paper_auto_execution_enabled: bool = False
    paper_execution_enabled: bool = False
    real_execution_enabled: bool = False
    operation_timezone: str = "Europe/Moscow"
    operation_session: str = "EOD"
    operation_schedule_enabled: bool = False
    lock_stale_after: timedelta = timedelta(hours=2)


class PaperOperationStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: PaperOperationStepName
    status: PaperOperationStepStatus
    started_at: datetime
    completed_at: datetime
    duration_ms: int = Field(default=0, ge=0)
    reason: str | None = None


class PaperOperationSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    REAL_BROKER_MUTATIONS: int = 0
    REAL_ORDERS_SENT: int = 0
    REAL_ORDERS_CANCELLED: int = 0
    REAL_POSITIONS_CHANGED: int = 0
    PAPER_OPERATION_RUNS: int = Field(default=1, ge=0)
    PAPER_RISK_PLANS: int = Field(default=0, ge=0)
    PAPER_ORDERS_PLANNED: int = Field(default=0, ge=0)
    PAPER_ORDERS_FILLED: int = Field(default=0, ge=0)
    PAPER_PORTFOLIO_MUTATIONS: int = Field(default=0, ge=0)
    LIVE_OUTCOMES_READ: int = 0
    LIVE_TARGETS_COMPUTED: int = 0
    LIVE_POST_EVENT_PRICE_READS: int = 0
    MODEL_TRAINING_PERFORMED: bool = False
    BACKTEST_PERFORMED: bool = False
    OLD_FUTURE_HOLDOUT_OPENED: bool = False
    PAID_SOURCE_CALLS: int = 0


class PaperOperationRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    operation_slot_id: str | None = None
    operation_contract_sha: str | None = None
    universe_sha: str | None = None
    research_status_sha: str | None = None
    market_adapter_id: str | None = None
    market_source: str | None = None
    market_audit: dict[str, Any] = Field(default_factory=_dict)
    operation_as_of: datetime
    mode: PaperOperationMode
    status: PaperOperationStatus
    status_code: str
    code_sha: str
    policy_version: str
    prompt_version: str
    agent_model_id: str
    research_status: dict[str, Any]
    universe: list[dict[str, Any]]
    portfolio_before_sha: str
    portfolio_before: dict[str, Any]
    market_snapshot_sha: str | None = None
    event_snapshot_sha: str | None = None
    agent_run_id: str | None = None
    agent_proposals: list[dict[str, Any]] = Field(default_factory=_dict_list)
    agent_tool_calls: list[dict[str, Any]] = Field(default_factory=_dict_list)
    risk_plan_id: str | None = None
    risk_decisions: list[dict[str, Any]] = Field(default_factory=_dict_list)
    paper_execution_status: str | None = None
    paper_order_ids: list[str] = Field(default_factory=list)
    paper_trade_ids: list[str] = Field(default_factory=list)
    portfolio_after_sha: str
    portfolio_after: dict[str, Any]
    replay_verified: bool
    day_transition_applied: bool = False
    steps: list[PaperOperationStep]
    reasons: list[str] = Field(default_factory=list)
    safety: PaperOperationSafety

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> PaperOperationRun:
        if self.operation_as_of.tzinfo is None or self.operation_as_of.utcoffset() is None:
            raise ValueError("operation_as_of must be timezone-aware")
        return self


class OperationAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    record_id: str
    operation_id: str
    record_type: OperationAuditRecordType
    occurred_at: datetime
    payload: dict[str, Any]


class PaperOperationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ARTIFACT_VERSION: str = "paper-trading-operation-v1"
    ARTIFACT_SHA: str
    BASE_MAIN_SHA: str
    HEAD_SHA: str
    PAPER_TRADING_OPERATION_READY: str
    AGENT_RESEARCH_CAPABILITY_READY: str
    RISK_ENGINE_READY: str
    PAPER_PORTFOLIO_READY: str
    REAL_EXECUTION_READY: str = "NO"
    ML_V2_DATASET_STATUS: str
    operation_policy_version: str
    operation_modes: list[PaperOperationMode]
    universe_size: int
    paper_ledger_event_count: int
    final_portfolio_sha: str
    replay_verified: bool
    safety: PaperOperationSafety
