from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

RISK_POLICY_VERSION = "paper-risk-policy-v1"


class RiskDecisionType(StrEnum):
    APPROVE = "APPROVE"
    REDUCE = "REDUCE"
    REJECT = "REJECT"
    NO_ACTION = "NO_ACTION"


class RiskReasonCode(StrEnum):
    OK = "OK"
    STALE_DATA = "STALE_DATA"
    RESEARCH_HEALTH_DEGRADED = "RESEARCH_HEALTH_DEGRADED"
    UNSUPPORTED_INSTRUMENT = "UNSUPPORTED_INSTRUMENT"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    POSITION_LIMIT = "POSITION_LIMIT"
    PORTFOLIO_GROSS_EXPOSURE_LIMIT = "PORTFOLIO_GROSS_EXPOSURE_LIMIT"
    PORTFOLIO_NET_EXPOSURE_LIMIT = "PORTFOLIO_NET_EXPOSURE_LIMIT"
    CASH_LIMIT = "CASH_LIMIT"
    TURNOVER_LIMIT = "TURNOVER_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    LIQUIDITY_LIMIT = "LIQUIDITY_LIMIT"
    MIN_TRADE_NOTIONAL = "MIN_TRADE_NOTIONAL"
    PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
    INVALID_PRICE = "INVALID_PRICE"
    NO_EXISTING_POSITION = "NO_EXISTING_POSITION"
    TARGET_WEIGHT_ALREADY_SATISFIED = "TARGET_WEIGHT_ALREADY_SATISFIED"
    KILL_SWITCH = "KILL_SWITCH"
    FUTURE_MARKET_SNAPSHOT = "FUTURE_MARKET_SNAPSHOT"
    LOT_SIZE_UNAVAILABLE = "LOT_SIZE_UNAVAILABLE"
    DUPLICATE_OR_CONFLICTING_PROPOSAL = "DUPLICATE_OR_CONFLICTING_PROPOSAL"


class PaperSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionPriceSource(StrEnum):
    ASK_PLUS_SLIPPAGE = "ASK_PLUS_SLIPPAGE"
    BID_MINUS_SLIPPAGE = "BID_MINUS_SLIPPAGE"
    LAST_PLUS_SLIPPAGE = "LAST_PLUS_SLIPPAGE"


class PaperOrderStatus(StrEnum):
    PLANNED = "PLANNED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"


class LedgerEventType(StrEnum):
    PORTFOLIO_CREATED = "PORTFOLIO_CREATED"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    RISK_DECISION = "RISK_DECISION"
    PAPER_ORDER_FILLED = "PAPER_ORDER_FILLED"
    POSITION_UPDATED = "POSITION_UPDATED"
    CASH_UPDATED = "CASH_UPDATED"
    DAY_CLOSED = "DAY_CLOSED"


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = RISK_POLICY_VERSION
    max_position_weight: float = Field(default=0.15, gt=0.0, le=1.0)
    max_gross_exposure: float = Field(default=0.80, gt=0.0, le=1.0)
    max_net_exposure: float = Field(default=0.80, gt=0.0, le=1.0)
    max_daily_turnover: float = Field(default=0.30, gt=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    max_portfolio_drawdown_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    min_cash_buffer_pct: float = Field(default=0.10, ge=0.0, lt=1.0)
    min_trade_notional: float = Field(default=1_000.0, gt=0.0)
    max_single_order_notional_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    max_agent_confidence_not_required: bool = True
    max_stale_market_age: timedelta = timedelta(minutes=30)
    kill_switch_enabled: bool = False
    target_weight_tolerance: float = Field(default=0.005, ge=0.0, lt=1.0)
    slippage_bps: int = Field(default=10, ge=0)
    commission_bps: int = Field(default=5, ge=0)
    reduce_to_available_cash: bool = True
    short_selling_enabled: bool = False
    margin_enabled: bool = False
    leverage_enabled: bool = False
    paper_auto_execution_enabled: bool = False


class MarketQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    as_of: datetime
    last_price: float | None
    bid: float | None = None
    ask: float | None = None
    lot_size: int | None = Field(default=None, ge=1)
    supported: bool = True


class PaperPosition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    quantity: int = Field(ge=0)
    average_cost: float = Field(ge=0.0)
    last_price: float = Field(gt=0.0)
    market_value: float = Field(ge=0.0)
    weight: float = Field(ge=0.0)
    unrealized_pnl: float


class PaperPortfolio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio_id: str
    cash: float = Field(ge=0.0)
    equity: float = Field(gt=0.0)
    positions: list[PaperPosition]
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_equity: float = Field(gt=0.0)
    drawdown: float = Field(ge=0.0)
    turnover_today: float = Field(ge=0.0)
    start_of_day_equity: float = Field(gt=0.0)
    as_of: datetime


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    proposal_id: str
    agent_run_id: str
    ticker: str
    proposal_action: str
    agent_target_weight: float = Field(ge=0.0)
    risk_decision: RiskDecisionType
    approved_target_weight: float = Field(ge=0.0)
    reason_codes: list[RiskReasonCode]
    decision_as_of: datetime


class PaperOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_order_id: str
    idempotency_key: str
    run_id: str
    proposal_id: str
    ticker: str
    side: PaperSide
    quantity: int = Field(gt=0)
    lot_size: int = Field(gt=0)
    planned_price: float = Field(gt=0.0)
    execution_price: float = Field(gt=0.0)
    price_source: ExecutionPriceSource
    slippage_bps: int = Field(ge=0)
    commission_bps: int = Field(ge=0)
    gross_notional: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    net_cash_effect: float
    created_at: datetime
    executed_at: datetime | None = None
    status: PaperOrderStatus = PaperOrderStatus.PLANNED

    @model_validator(mode="after")
    def quantity_is_whole_lots(self) -> PaperOrder:
        if self.quantity % self.lot_size:
            raise ValueError("quantity must be a whole number of lots")
        return self


class PaperTrade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_trade_id: str
    paper_order_id: str
    ticker: str
    side: PaperSide
    quantity: int = Field(gt=0)
    fill_price: float = Field(gt=0.0)
    gross_value: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    cash_delta: float
    executed_at: datetime


class RiskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    agent_run_id: str
    agent_run_sha: str
    policy_version: str
    decision_as_of: datetime
    initial_portfolio: PaperPortfolio
    market_snapshot: list[MarketQuote]
    decisions: list[RiskDecision]
    paper_orders: list[PaperOrder]


class LedgerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    event_id: str
    event_type: LedgerEventType
    portfolio_id: str
    idempotency_key: str | None = None
    occurred_at: datetime
    payload: dict[str, object]


class ReplayVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_matches: bool
    expected_sha: str
    replayed_sha: str
    event_count: int = Field(ge=0)


class PipelineSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    REAL_BROKER_MUTATIONS: int = 0
    REAL_ORDERS_SENT: int = 0
    REAL_ORDERS_CANCELLED: int = 0
    REAL_POSITIONS_CHANGED: int = 0
    PAPER_ORDERS_PLANNED: int = Field(ge=0)
    PAPER_ORDERS_FILLED: int = Field(ge=0)
    PAPER_PORTFOLIO_MUTATIONS: int = Field(ge=0)
    LIVE_OUTCOMES_READ: int = 0
    LIVE_TARGETS_COMPUTED: int = 0
    LIVE_POST_EVENT_PRICE_READS: int = 0
    MODEL_TRAINING_PERFORMED: bool = False
    BACKTEST_PERFORMED: bool = False
    OLD_FUTURE_HOLDOUT_OPENED: bool = False
    PAID_SOURCE_CALLS: int = 0


class PaperExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: RiskPlan
    filled_orders: list[PaperOrder]
    paper_trades: list[PaperTrade]
    final_portfolio: PaperPortfolio
    replay_verification: ReplayVerification
    duplicate_executions_skipped: int = Field(ge=0)
    safety: PipelineSafety
