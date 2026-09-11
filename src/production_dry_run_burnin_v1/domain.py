from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

BURNIN_POLICY_VERSION = "production-dry-run-burnin-v1"
BURNIN_ARTIFACT_VERSION = "production-dry-run-burnin-v1"
BURNIN_OBSERVATION_SCHEMA_VERSION = "production-dry-run-burnin-observation-v1"
PRIMARY_OPERATION_SLOT = "BURNIN_EOD"


class BurninAttemptType(StrEnum):
    PRIMARY = "PRIMARY"
    RETRY = "RETRY"


class BurninObservationStatus(StrEnum):
    PASS = "PASS"
    BLOCKED_EXPECTED = "BLOCKED_EXPECTED"
    FAIL = "FAIL"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"


class MoexSessionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"
    TRADING_DAY = "OPEN"
    NOT_TRADING_DAY = "CLOSED"


class BurninCollectionStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"


class BurninStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PASS = "PASS"
    FAIL = "FAIL"


class BurninPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = BURNIN_POLICY_VERSION
    min_distinct_moex_trading_days: int = Field(default=5, ge=1)
    min_primary_cycles: int = Field(default=5, ge=1)
    max_blocked_primary_cycle_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    min_market_fresh_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    min_research_ready_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    min_agent_valid_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    min_risk_completion_rate: float = Field(default=0.90, ge=0.0, le=1.0)
    max_source_clock_skew_seconds: float = Field(default=5.0, ge=0.0)
    require_zero_paper_mutations: bool = True
    require_zero_real_mutations: bool = True
    require_zero_future_data_violations: bool = True
    require_zero_holdout_violations: bool = True
    operation_timezone: str = "Europe/Moscow"
    primary_operation_slot: str = PRIMARY_OPERATION_SLOT
    paper_execution_enabled: bool = False
    real_execution_enabled: bool = False
    paper_operation_schedule_enabled: bool = False


class BurninSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    PAPER_EXECUTION_ENABLED: bool = False
    REAL_EXECUTION_ENABLED: bool = False
    PAPER_OPERATION_SCHEDULE_ENABLED: bool = False
    PAPER_ORDERS_FILLED: int = Field(default=0, ge=0)
    PAPER_PORTFOLIO_MUTATIONS: int = Field(default=0, ge=0)
    REAL_BROKER_MUTATIONS: int = Field(default=0, ge=0)
    REAL_ORDERS_SENT: int = Field(default=0, ge=0)
    REAL_ORDERS_CANCELLED: int = Field(default=0, ge=0)
    REAL_POSITIONS_CHANGED: int = Field(default=0, ge=0)
    LIVE_OUTCOMES_READ: int = Field(default=0, ge=0)
    LIVE_TARGETS_COMPUTED: int = Field(default=0, ge=0)
    LIVE_POST_EVENT_PRICE_READS: int = Field(default=0, ge=0)
    MODEL_TRAINING_PERFORMED: bool = False
    BACKTEST_PERFORMED: bool = False
    OLD_FUTURE_HOLDOUT_OPENED: bool = False
    PAID_SOURCE_CALLS: int = Field(default=0, ge=0)

    def violation_count(self) -> int:
        numeric = (
            self.PAPER_ORDERS_FILLED
            + self.PAPER_PORTFOLIO_MUTATIONS
            + self.REAL_BROKER_MUTATIONS
            + self.REAL_ORDERS_SENT
            + self.REAL_ORDERS_CANCELLED
            + self.REAL_POSITIONS_CHANGED
            + self.LIVE_OUTCOMES_READ
            + self.LIVE_TARGETS_COMPUTED
            + self.LIVE_POST_EVENT_PRICE_READS
            + self.PAID_SOURCE_CALLS
        )
        boolean = sum(
            (
                self.PAPER_EXECUTION_ENABLED,
                self.REAL_EXECUTION_ENABLED,
                self.PAPER_OPERATION_SCHEDULE_ENABLED,
                self.MODEL_TRAINING_PERFORMED,
                self.BACKTEST_PERFORMED,
                self.OLD_FUTURE_HOLDOUT_OPENED,
            )
        )
        return numeric + boolean


class MoexSessionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trading_date: str
    status: MoexSessionStatus
    source: str
    checked_at: datetime
    evidence_sha: str | None = None
    reason: str | None = None


class BurninObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = BURNIN_OBSERVATION_SCHEMA_VERSION
    sequence: int = Field(ge=1)
    previous_record_sha: str | None
    record_sha: str
    observation_id: str
    burnin_observation_id: str = ""
    trading_date: str
    market_date: str = ""
    operation_slot: str
    operation_session: str = ""
    operation_id: str = ""
    operation_slot_id: str = ""
    attempt_type: BurninAttemptType
    primary_operation_id: str
    retry_index: int = Field(default=0, ge=0)
    retry_reason: str | None = None
    cycle_started_at: datetime
    decision_as_of: datetime
    completed_at: datetime
    code_sha: str
    policy_version: str
    prompt_version: str
    agent_model_id: str
    market_adapter_id: str
    universe_sha: str
    operation_contract_sha: str
    market_snapshot_sha: str
    event_snapshot_sha: str
    research_status_sha: str
    portfolio_sha: str = ""
    market_fetch_started_at: datetime | None = None
    market_fetch_completed_at: datetime | None = None
    market_source_clock_delta_seconds: float = 0.0
    market_fetch_duration_ms: int = Field(default=0, ge=0)
    operation_duration_ms: int = Field(default=0, ge=0)
    model_latency_ms: int = Field(default=0, ge=0)
    universe_count: int = Field(default=0, ge=0)
    market_quote_count: int = Field(default=0, ge=0)
    fresh_quote_count: int = Field(default=0, ge=0)
    stale_quote_count: int = Field(default=0, ge=0)
    future_quote_count: int = Field(default=0, ge=0)
    missing_quote_count: int = Field(default=0, ge=0)
    invalid_quote_count: int = Field(default=0, ge=0)
    max_market_age_seconds: float = Field(default=0.0, ge=0.0)
    max_source_clock_delta_seconds: float = 0.0
    research_status: str
    research_operation_status: str = "UNKNOWN"
    operational_burnin_status: str = "UNKNOWN"
    source_failure_isolation: bool
    seal_verified: bool
    research_seal_verified: bool = False
    agent_decision_status: str
    agent_steps: int = Field(default=0, ge=0)
    agent_proposal_count: int = Field(default=0, ge=0)
    proposal_count: int = Field(default=0, ge=0)
    proposal_action_counts: dict[str, int] = Field(default_factory=dict)
    agent_tool_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    tool_calls: list[dict[str, object]] = Field(default_factory=lambda: [])
    validation_reasons: list[str] = Field(default_factory=list)
    model_calls: int = Field(default=0, ge=0)
    risk_plan_created: bool
    risk_decision_count: int = Field(default=0, ge=0)
    risk_approved_count: int = Field(default=0, ge=0)
    approved_count: int = Field(default=0, ge=0)
    risk_reduced_count: int = Field(default=0, ge=0)
    reduced_count: int = Field(default=0, ge=0)
    risk_rejected_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    paper_orders_planned: int = Field(default=0, ge=0)
    operation_status: str
    status: BurninObservationStatus
    status_code: str
    operation_status_code: str = ""
    reasons: list[str] = Field(default_factory=list)
    paper_ledger_event_count_before: int = Field(ge=0)
    paper_ledger_event_count_after: int = Field(ge=0)
    paper_ledger_sha_before: str
    paper_ledger_sha_after: str
    portfolio_sha_before: str
    portfolio_sha_after: str
    paper_ledger_unchanged: bool
    paper_orders_filled: int = Field(default=0, ge=0)
    paper_portfolio_mutations: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    session: MoexSessionEvidence
    safety: BurninSafety

    @model_validator(mode="after")
    def validate_invariants(self) -> BurninObservation:
        for value in (self.cycle_started_at, self.decision_as_of, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("burn-in timestamps must be timezone-aware")
        if self.cycle_started_at > self.decision_as_of or self.decision_as_of > self.completed_at:
            raise ValueError("BURNIN_TIMESTAMP_ORDER_INVALID")
        if self.attempt_type == BurninAttemptType.PRIMARY and self.retry_index != 0:
            raise ValueError("PRIMARY_RETRY_INDEX_MUST_BE_ZERO")
        if self.attempt_type == BurninAttemptType.RETRY and self.retry_index < 1:
            raise ValueError("RETRY_INDEX_REQUIRED")
        return self


class BurninRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    observation: BurninObservation | None = None
    existing_observation_id: str | None = None


class BurninReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    BURNIN_POLICY_VERSION: str
    BURNIN_COLLECTION_STATUS: BurninCollectionStatus
    BURNIN_STATUS: BurninStatus
    PRODUCTION_DRY_RUN_BURNIN_READY: str
    BURNIN_LEDGER_INTEGRITY: str
    distinct_trading_days: int
    valid_cycles: int
    cycle_count: int
    successful_cycle_count: int
    blocked_cycle_count: int
    degraded_cycle_count: int
    primary_cycles: int
    primary_pass_cycles: int
    primary_blocked_cycles: int
    primary_failed_cycles: int
    primary_pass_rate: float
    blocked_rate: float
    market_fresh_rate: float
    market_ready_rate: float
    research_ready_rate: float
    agent_valid_rate: float
    risk_completion_rate: float
    dry_run_end_to_end_rate: float
    model_timeout_count: int
    model_error_count: int
    median_cycle_duration_ms: float
    p95_cycle_duration_ms: float
    duplicate_prevented_count: int
    tool_call_total: int
    tool_failure_total: int
    tool_failure_rate: float
    mean_market_fetch_ms: float
    p95_market_fetch_ms: float
    mean_model_latency_ms: float
    p95_model_latency_ms: float
    mean_total_cycle_ms: float
    p95_total_cycle_ms: float
    future_data_violation_count: int
    paper_mutation_count: int
    real_mutation_count: int
    holdout_violation_count: int
    proposal_statistics: dict[str, int]
    safety_behavior: str
    last_observation_id: str | None = None
