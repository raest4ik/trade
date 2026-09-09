from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

from src.ai_trading_agent_v1.application import (
    PROMPT_VERSION,
    AgentDataContext,
    AgentModel,
    AgentRunConfig,
    build_allowed_universe,
    existing_market_context,
    recent_event_context,
    research_status,
    run_read_only_research_agent_v1,
)
from src.ai_trading_agent_v1.domain import AgentDecisionStatus
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.domain import (
    OPERATION_ID_NAMESPACE,
    OperationAuditEvent,
    OperationAuditRecordType,
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationRun,
    PaperOperationSafety,
    PaperOperationStatus,
    PaperOperationStep,
    PaperOperationStepName,
    PaperOperationStepStatus,
)
from src.paper_trading_operation_v1.repository import (
    OperationAuditRepository,
    operation_lock,
)
from src.risk_engine_paper_v1.application import (
    close_paper_day,
    evaluate_agent_run,
    execute_paper_plan,
    initial_paper_portfolio,
    market_snapshot_sha,
    portfolio_state_sha,
    replay_portfolio,
)
from src.risk_engine_paper_v1.domain import (
    LedgerEventType,
    MarketQuote,
    PaperExecutionResult,
    PaperExecutionStatus,
    PaperPortfolio,
    RiskPlan,
    RiskPolicy,
)
from src.risk_engine_paper_v1.repository import LedgerIntegrityError, PaperLedgerRepository

DEFAULT_STATE_ROOT = Path("state/paper-operation-v1")
DEFAULT_PAPER_STATE_ROOT = Path("state/paper-portfolio-v1")


class SimulatedCrashAfterPaperFillError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PaperOperationContext:
    universe: list[dict[str, Any]]
    market_quotes: list[MarketQuote]
    market_context: dict[str, Any]
    event_context: dict[str, Any]
    research_status: dict[str, Any]


class PaperOperationContextProvider(Protocol):
    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext: ...


@dataclass(frozen=True, slots=True)
class StaticPaperOperationContextProvider:
    context: PaperOperationContext

    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext:
        return self.context


@dataclass(frozen=True, slots=True)
class ExistingArtifactContextProvider:
    agent_config: AgentRunConfig

    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext:
        canonical = build_allowed_universe(self.agent_config)
        by_ticker = {str(row["ticker"]).upper(): row for row in canonical}
        held = [position.ticker.upper() for position in portfolio.positions]
        selected_tickers = list(dict.fromkeys([*held, *sorted(by_ticker)]))[
            : policy.max_operation_universe
        ]
        universe = [by_ticker[ticker] for ticker in selected_tickers if ticker in by_ticker]
        market = existing_market_context(
            self.agent_config.market_features_path,
            universe,
            operation_as_of,
        )
        return PaperOperationContext(
            universe=universe,
            market_quotes=market_quotes_from_context(market, universe),
            market_context=market,
            event_context=recent_event_context(
                self.agent_config.live_root,
                selected_tickers,
                operation_as_of,
                self.agent_config,
            ),
            research_status=research_status(
                self.agent_config.operation_root,
                self.agent_config.operational_proof_path,
                operation_as_of,
            ),
        )


def run_paper_operation(
    *,
    operation_as_of: datetime,
    mode: PaperOperationMode,
    model: AgentModel,
    context_provider: PaperOperationContextProvider,
    paper_repository: PaperLedgerRepository,
    audit_repository: OperationAuditRepository,
    state_root: Path,
    code_sha: str,
    policy: PaperOperationPolicy | None = None,
    risk_policy: RiskPolicy | None = None,
    operation_slot: str | None = None,
    simulate_crash_after_fill: bool = False,
) -> PaperOperationRun:
    operation_policy = policy or PaperOperationPolicy()
    risk = risk_policy or RiskPolicy()
    as_of = _utc(operation_as_of)
    ledger_failure: str | None = None
    try:
        raw_portfolio = _replay_or_initial(paper_repository, as_of)
    except (LedgerIntegrityError, ValueError):
        raw_portfolio = initial_paper_portfolio(as_of)
        ledger_failure = "PAPER_LEDGER_INTEGRITY_FAILED"
    context_failure: str | None = None
    try:
        context = context_provider.load(
            operation_as_of=as_of,
            portfolio=raw_portfolio,
            policy=operation_policy,
        )
    except Exception as exc:
        context_failure = f"CONTEXT_UNAVAILABLE:{type(exc).__name__}"
        context = PaperOperationContext(
            universe=[],
            market_quotes=[],
            market_context={},
            event_context={"events": []},
            research_status={"research_status_as_of": as_of.isoformat()},
        )
    universe = context.universe[: operation_policy.max_operation_universe]
    universe_sha = sha256_payload(universe)
    session = operation_slot or operation_policy.operation_session
    operation_slot_id = build_operation_slot_id(
        operation_as_of=as_of,
        session=session,
        timezone_name=operation_policy.operation_timezone,
    )
    operation_id = build_operation_id(
        operation_as_of=as_of,
        session=session,
        timezone_name=operation_policy.operation_timezone,
    )
    operation_contract_sha = build_operation_contract_sha(
        model_id=model.model_id,
        universe_sha=universe_sha,
        policy_version=operation_policy.policy_version,
        code_sha=code_sha,
    )
    lock_path = state_root / "operation.lock"
    with operation_lock(
        lock_path,
        operation_id=operation_id,
        stale_after=operation_policy.lock_stale_after,
    ):
        completed = audit_repository.completed_run(operation_id)
        if completed is not None:
            return completed.model_copy(
                update={
                    "status": PaperOperationStatus.ALREADY_PROCESSED,
                    "status_code": "ALREADY_PROCESSED",
                    "safety": completed.safety.model_copy(
                        update={
                            "PAPER_OPERATION_RUNS": 0,
                            "PAPER_RISK_PLANS": 0,
                            "PAPER_ORDERS_PLANNED": 0,
                            "PAPER_ORDERS_FILLED": 0,
                            "PAPER_PORTFOLIO_MUTATIONS": 0,
                        }
                    ),
                }
            )
        pending = audit_repository.latest(operation_id)
        if pending is not None and pending.record_type == OperationAuditRecordType.PREPARED:
            return recover_prepared_operation(
                pending=pending,
                paper_repository=paper_repository,
                audit_repository=audit_repository,
                operation_as_of=as_of,
            )

        model_failure = (
            "AGENT_MODEL_UNAVAILABLE" if model.model_id == "unconfigured-agent-model" else None
        )
        initial_reasons = [
            reason for reason in (ledger_failure, context_failure, model_failure) if reason
        ]
        preflight_reasons = initial_reasons + preflight_failures(
            operation_as_of=as_of,
            portfolio=raw_portfolio,
            context=context,
            policy=operation_policy,
        )
        if preflight_reasons:
            blocked = _base_run(
                operation_id=operation_id,
                operation_slot_id=operation_slot_id,
                operation_contract_sha=operation_contract_sha,
                universe_sha=universe_sha,
                operation_as_of=as_of,
                mode=mode,
                status=PaperOperationStatus.BLOCKED,
                status_code=preflight_reasons[0],
                code_sha=code_sha,
                model_id=model.model_id,
                policy=operation_policy,
                context=context,
                portfolio=raw_portfolio,
                reasons=preflight_reasons,
                steps=_blocked_steps(as_of, PaperOperationStepName.PREFLIGHT, preflight_reasons[0]),
            )
            return _complete(audit_repository, blocked, OperationAuditRecordType.COMPLETED)

        portfolio, transitioned = _day_transition(
            paper_repository,
            raw_portfolio,
            as_of,
        )
        portfolio = _mark_portfolio(
            portfolio,
            context.market_quotes,
            as_of,
            risk,
        )
        agent_context = AgentDataContext(
            as_of=as_of,
            allowed_universe=universe,
            portfolio_snapshot=portfolio.model_dump(mode="json"),
            market_context_snapshot=context.market_context,
            event_context_snapshot=context.event_context,
            research_status_snapshot=context.research_status,
        )
        agent_root = state_root / "agent-runs" / operation_id
        agent_result = run_read_only_research_agent_v1(
            config=AgentRunConfig(
                output_root=agent_root,
                code_sha=code_sha,
                run_id=operation_id,
                created_at=as_of,
                max_agent_steps=operation_policy.max_agent_steps,
                max_tool_calls=operation_policy.max_tool_calls,
                max_tickers_per_run=operation_policy.max_operation_universe,
            ),
            model=model,
            deterministic_context=agent_context,
        )
        if agent_result.AGENT_DECISION_STATUS != AgentDecisionStatus.VALID:
            reason = (
                "AGENT_MODEL_UNAVAILABLE"
                if model.model_id == "unconfigured-agent-model"
                else f"AGENT_{agent_result.AGENT_DECISION_STATUS.value}"
            )
            blocked = _base_run(
                operation_id=operation_id,
                operation_slot_id=operation_slot_id,
                operation_contract_sha=operation_contract_sha,
                universe_sha=universe_sha,
                operation_as_of=as_of,
                mode=mode,
                status=PaperOperationStatus.BLOCKED,
                status_code=reason,
                code_sha=code_sha,
                model_id=model.model_id,
                policy=operation_policy,
                context=context,
                portfolio=portfolio,
                reasons=[reason],
                steps=_blocked_steps(as_of, PaperOperationStepName.AGENT, reason),
                day_transition_applied=transitioned,
            )
            return _complete(audit_repository, blocked, OperationAuditRecordType.COMPLETED)

        risk_plan = evaluate_agent_run(
            agent_run=agent_result.model_dump(mode="json"),
            portfolio=portfolio,
            market_snapshot=context.market_quotes,
            policy=risk,
            decision_as_of=as_of,
            ledger_event_count=paper_repository.last_sequence() + int(transitioned),
        )
        prepared_run = _run_from_agent_and_risk(
            operation_id=operation_id,
            operation_slot_id=operation_slot_id,
            operation_contract_sha=operation_contract_sha,
            universe_sha=universe_sha,
            operation_as_of=as_of,
            mode=mode,
            code_sha=code_sha,
            policy=operation_policy,
            context=context,
            portfolio=portfolio,
            agent_result=agent_result.model_dump(mode="json"),
            risk_plan=risk_plan,
            day_transition_applied=transitioned,
        )
        if mode == PaperOperationMode.DRY_RUN:
            status = (
                PaperOperationStatus.DEGRADED
                if _risk_data_degraded([row.model_dump(mode="json") for row in risk_plan.decisions])
                else PaperOperationStatus.NO_ACTION
                if not risk_plan.paper_orders
                else PaperOperationStatus.SUCCESS
            )
            completed_run = prepared_run.model_copy(
                update={
                    "status": status,
                    "status_code": status.value,
                    "paper_execution_status": "SKIPPED_DRY_RUN",
                    "steps": _successful_steps(as_of, execute=False),
                }
            )
            return _complete(audit_repository, completed_run, OperationAuditRecordType.COMPLETED)
        if not operation_policy.paper_execution_enabled:
            blocked = prepared_run.model_copy(
                update={
                    "status": PaperOperationStatus.BLOCKED,
                    "status_code": "PAPER_EXECUTION_DISABLED",
                    "reasons": ["PAPER_EXECUTION_DISABLED"],
                    "steps": _blocked_steps(
                        as_of, PaperOperationStepName.PAPER_EXECUTION, "PAPER_EXECUTION_DISABLED"
                    ),
                }
            )
            return _complete(audit_repository, blocked, OperationAuditRecordType.COMPLETED)

        audit_repository.append(
            _audit_event(
                audit_repository,
                operation_id,
                OperationAuditRecordType.PREPARED,
                as_of,
                {
                    "run": prepared_run.model_dump(mode="json"),
                    "risk_plan": risk_plan.model_dump(mode="json"),
                },
            )
        )
        if transitioned:
            close_paper_day(paper_repository, next_day_as_of=as_of)
        execution = execute_paper_plan(risk_plan, paper_repository, execution_as_of=as_of)
        if simulate_crash_after_fill:
            raise SimulatedCrashAfterPaperFillError("SIMULATED_CRASH_AFTER_PAPER_FILL")
        completed_run = _finish_execution(prepared_run, execution)
        return _complete(audit_repository, completed_run, OperationAuditRecordType.COMPLETED)


def recover_prepared_operation(
    *,
    pending: OperationAuditEvent,
    paper_repository: PaperLedgerRepository,
    audit_repository: OperationAuditRepository,
    operation_as_of: datetime,
) -> PaperOperationRun:
    prepared = PaperOperationRun.model_validate(cast("dict[str, Any]", pending.payload["run"]))
    plan = RiskPlan.model_validate(cast("dict[str, Any]", pending.payload["risk_plan"]))
    if prepared.day_transition_applied:
        current = _replay_or_initial(paper_repository, operation_as_of)
        if paper_repository.events() and _market_date(current.as_of) < _market_date(
            operation_as_of
        ):
            close_paper_day(paper_repository, next_day_as_of=operation_as_of)
    execution = execute_paper_plan(plan, paper_repository, execution_as_of=operation_as_of)
    trade_ids = [
        str(cast("dict[str, Any]", event.payload.get("trade", {})).get("paper_trade_id"))
        for event in paper_repository.events()
        if event.event_type == LedgerEventType.PAPER_ORDER_FILLED
        and cast("dict[str, Any]", event.payload.get("order", {})).get("paper_order_id")
        in prepared.paper_order_ids
    ]
    recovered = _finish_execution(prepared, execution).model_copy(
        update={
            "status_code": "RECOVERED_AFTER_COMMITTED_FILL",
            "paper_trade_ids": trade_ids,
            "steps": _successful_steps(operation_as_of, execute=True),
        }
    )
    return _complete(audit_repository, recovered, OperationAuditRecordType.RECOVERED)


def preflight_failures(
    *,
    operation_as_of: datetime,
    portfolio: PaperPortfolio,
    context: PaperOperationContext,
    policy: PaperOperationPolicy,
) -> list[str]:
    reasons = pit_failures(operation_as_of, portfolio, context)
    research = context.research_status
    if policy.real_execution_enabled:
        reasons.append("REAL_EXECUTION_MUST_BE_DISABLED")
    if len(context.universe) > policy.max_operation_universe:
        reasons.append("MAX_OPERATION_UNIVERSE_EXCEEDED")
    if not context.market_quotes:
        reasons.append("MARKET_CONTEXT_UNAVAILABLE")
    held = {position.ticker.upper() for position in portfolio.positions}
    allowed = {str(row.get("ticker", "")).upper() for row in context.universe}
    if not held.issubset(allowed):
        reasons.append("HELD_POSITION_OUTSIDE_ALLOWED_UNIVERSE")
    if policy.require_research_ready and research.get("LIVE_RESEARCH_OPERATION_STATUS") != "READY":
        reasons.append("LIVE_RESEARCH_NOT_READY")
    if policy.require_operational_burnin_pass and research.get("OPERATIONAL_BURN_IN") != "PASS":
        reasons.append("OPERATIONAL_BURN_IN_NOT_PASS")
    if policy.require_source_failure_isolation and (
        research.get("SOURCE_FAILURE_ISOLATION") is not True
        or research.get("SOURCE_FAILURE_ISOLATION_PROOF_LEVEL")
        not in {"APPLICATION_PROOF", "REAL_BURNIN_PROOF"}
    ):
        reasons.append("SOURCE_FAILURE_ISOLATION_NOT_PROVEN")
    seal = cast("dict[str, Any]", research.get("seal", {}))
    if seal.get("sealed_epoch_verified") is not True or int(seal.get("violations", 0)) != 0:
        reasons.append("RESEARCH_SEAL_NOT_VERIFIED")
    return list(dict.fromkeys(reasons))


def pit_failures(
    operation_as_of: datetime,
    portfolio: PaperPortfolio,
    context: PaperOperationContext,
) -> list[str]:
    reasons: list[str] = []
    if portfolio.as_of > operation_as_of or any(
        position.mark_as_of > operation_as_of for position in portfolio.positions
    ):
        reasons.append("FUTURE_PORTFOLIO_MARK")
    research_as_of = _parse_time(context.research_status.get("research_status_as_of"))
    if research_as_of is None:
        reasons.append("RESEARCH_STATUS_TIMESTAMP_INVALID")
    elif research_as_of > operation_as_of:
        reasons.append("FUTURE_RESEARCH_STATUS")
    if any(quote.as_of > operation_as_of for quote in context.market_quotes):
        reasons.append("FUTURE_MARKET_QUOTE")
    events = cast("list[dict[str, Any]]", context.event_context.get("events", []))
    for event in events:
        published = _parse_time(event.get("published_at"))
        if published is None:
            reasons.append("EVENT_TIMESTAMP_INVALID")
        elif published > operation_as_of:
            reasons.append("FUTURE_EVENT")
    return reasons


def build_operation_id(
    *,
    operation_as_of: datetime,
    session: str,
    timezone_name: str = "Europe/Moscow",
) -> str:
    operation_slot_id = build_operation_slot_id(
        operation_as_of=operation_as_of,
        session=session,
        timezone_name=timezone_name,
    )
    contract = {
        "namespace": OPERATION_ID_NAMESPACE,
        "operation_slot_id": operation_slot_id,
    }
    return f"paper-operation-{sha256_payload(contract)[:24]}"


def build_operation_contract_sha(
    *,
    model_id: str,
    universe_sha: str,
    policy_version: str,
    code_sha: str,
) -> str:
    return sha256_payload(
        {
            "agent_model_id": model_id,
            "prompt_version": PROMPT_VERSION,
            "universe_sha": universe_sha,
            "policy_version": policy_version,
            "code_sha": code_sha,
        }
    )


def build_operation_slot_id(
    *,
    operation_as_of: datetime,
    session: str,
    timezone_name: str = "Europe/Moscow",
) -> str:
    trading_date = operation_as_of.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    return f"{trading_date}:{session}"


def market_quotes_from_context(
    market: dict[str, Any], universe: Sequence[dict[str, Any]]
) -> list[MarketQuote]:
    mapping = {str(row["ticker"]).upper(): row for row in universe}
    by_ticker = cast("dict[str, dict[str, Any]]", market.get("by_ticker", {}))
    quotes: list[MarketQuote] = []
    for ticker, row in sorted(by_ticker.items()):
        timestamp = _parse_time(row.get("market_data_as_of"))
        if timestamp is None:
            continue
        instrument = mapping.get(ticker.upper(), {})
        quotes.append(
            MarketQuote(
                ticker=ticker.upper(),
                as_of=timestamp,
                last_price=_number(row.get("last_price")),
                bid=_number(row.get("bid")),
                ask=_number(row.get("ask")),
                lot_size=cast("int | None", instrument.get("lot_size")),
                supported=instrument.get("supported") is True,
            )
        )
    return quotes


def operation_history(repository: OperationAuditRepository) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in repository.events():
        if event.record_type == OperationAuditRecordType.PREPARED:
            continue
        run = PaperOperationRun.model_validate(event.payload)
        rows.append(
            {
                "operation_id": run.operation_id,
                "as_of": run.operation_as_of.isoformat(),
                "status": run.status.value,
                "proposal_count": len(run.agent_proposals),
                "risk_approved_count": sum(
                    row.get("risk_decision") in {"APPROVE", "REDUCE"} for row in run.risk_decisions
                ),
                "paper_fills": len(run.paper_trade_ids),
                "portfolio_equity_after": run.portfolio_after.get("equity"),
            }
        )
    return rows


def _finish_execution(
    prepared: PaperOperationRun, execution: PaperExecutionResult
) -> PaperOperationRun:
    success = (
        execution.execution_status == PaperExecutionStatus.SUCCESS
        and execution.replay_verification.replay_matches
    )
    incomplete = _risk_data_degraded(prepared.risk_decisions)
    status = (
        PaperOperationStatus.DEGRADED
        if success and incomplete
        else PaperOperationStatus.SUCCESS
        if success and prepared.paper_order_ids
        else PaperOperationStatus.NO_ACTION
        if success
        else PaperOperationStatus.FAILED
    )
    portfolio = execution.final_portfolio
    return prepared.model_copy(
        update={
            "status": status,
            "status_code": execution.status_code,
            "paper_execution_status": execution.execution_status.value,
            "paper_trade_ids": [row.paper_trade_id for row in execution.paper_trades],
            "portfolio_after_sha": portfolio_state_sha(portfolio),
            "portfolio_after": portfolio.model_dump(mode="json"),
            "replay_verified": execution.replay_verification.replay_matches,
            "steps": _successful_steps(prepared.operation_as_of, execute=True),
            "safety": prepared.safety.model_copy(
                update={
                    "PAPER_ORDERS_FILLED": len(execution.filled_orders),
                    "PAPER_PORTFOLIO_MUTATIONS": len(execution.filled_orders),
                }
            ),
        }
    )


def _run_from_agent_and_risk(
    *,
    operation_id: str,
    operation_slot_id: str,
    operation_contract_sha: str,
    universe_sha: str,
    operation_as_of: datetime,
    mode: PaperOperationMode,
    code_sha: str,
    policy: PaperOperationPolicy,
    context: PaperOperationContext,
    portfolio: PaperPortfolio,
    agent_result: dict[str, Any],
    risk_plan: RiskPlan,
    day_transition_applied: bool,
) -> PaperOperationRun:
    return PaperOperationRun(
        operation_id=operation_id,
        operation_slot_id=operation_slot_id,
        operation_contract_sha=operation_contract_sha,
        universe_sha=universe_sha,
        operation_as_of=operation_as_of,
        mode=mode,
        status=PaperOperationStatus.STARTED,
        status_code="PREPARED",
        code_sha=code_sha,
        policy_version=policy.policy_version,
        prompt_version=PROMPT_VERSION,
        agent_model_id=str(agent_result["agent_model_id"]),
        research_status=context.research_status,
        universe=context.universe,
        portfolio_before_sha=portfolio_state_sha(portfolio),
        portfolio_before=portfolio.model_dump(mode="json"),
        market_snapshot_sha=market_snapshot_sha(context.market_quotes),
        event_snapshot_sha=sha256_payload(context.event_context),
        agent_run_id=str(agent_result["run_id"]),
        agent_proposals=cast("list[dict[str, Any]]", agent_result["final_proposals"]),
        agent_tool_calls=cast("list[dict[str, Any]]", agent_result["tool_calls"]),
        risk_plan_id=risk_plan.plan_id,
        risk_decisions=[row.model_dump(mode="json") for row in risk_plan.decisions],
        paper_order_ids=[row.paper_order_id for row in risk_plan.paper_orders],
        portfolio_after_sha=portfolio_state_sha(portfolio),
        portfolio_after=portfolio.model_dump(mode="json"),
        replay_verified=True,
        day_transition_applied=day_transition_applied,
        steps=_successful_steps(operation_as_of, execute=False),
        safety=PaperOperationSafety(
            PAPER_RISK_PLANS=1,
            PAPER_ORDERS_PLANNED=len(risk_plan.paper_orders),
        ),
    )


def _base_run(
    *,
    operation_id: str,
    operation_slot_id: str,
    operation_contract_sha: str,
    universe_sha: str,
    operation_as_of: datetime,
    mode: PaperOperationMode,
    status: PaperOperationStatus,
    status_code: str,
    code_sha: str,
    model_id: str,
    policy: PaperOperationPolicy,
    context: PaperOperationContext,
    portfolio: PaperPortfolio,
    reasons: list[str],
    steps: list[PaperOperationStep],
    day_transition_applied: bool = False,
) -> PaperOperationRun:
    state = portfolio.model_dump(mode="json")
    state_sha = portfolio_state_sha(portfolio)
    return PaperOperationRun(
        operation_id=operation_id,
        operation_slot_id=operation_slot_id,
        operation_contract_sha=operation_contract_sha,
        universe_sha=universe_sha,
        operation_as_of=operation_as_of,
        mode=mode,
        status=status,
        status_code=status_code,
        code_sha=code_sha,
        policy_version=policy.policy_version,
        prompt_version=PROMPT_VERSION,
        agent_model_id=model_id,
        research_status=context.research_status,
        universe=context.universe,
        portfolio_before_sha=state_sha,
        portfolio_before=state,
        market_snapshot_sha=market_snapshot_sha(context.market_quotes),
        event_snapshot_sha=sha256_payload(context.event_context),
        portfolio_after_sha=state_sha,
        portfolio_after=state,
        replay_verified=True,
        day_transition_applied=day_transition_applied,
        steps=steps,
        reasons=reasons,
        safety=PaperOperationSafety(),
    )


def _complete(
    repository: OperationAuditRepository,
    run: PaperOperationRun,
    record_type: OperationAuditRecordType,
) -> PaperOperationRun:
    repository.append(
        _audit_event(
            repository,
            run.operation_id,
            record_type,
            run.operation_as_of,
            run.model_dump(mode="json"),
        )
    )
    return run


def _audit_event(
    repository: OperationAuditRepository,
    operation_id: str,
    record_type: OperationAuditRecordType,
    occurred_at: datetime,
    payload: dict[str, Any],
) -> OperationAuditEvent:
    sequence = len(repository.events()) + 1
    return OperationAuditEvent(
        sequence=sequence,
        record_id=f"operation-audit-{sha256_payload([operation_id, record_type, sequence])[:24]}",
        operation_id=operation_id,
        record_type=record_type,
        occurred_at=occurred_at,
        payload=payload,
    )


def _day_transition(
    repository: PaperLedgerRepository,
    portfolio: PaperPortfolio,
    as_of: datetime,
) -> tuple[PaperPortfolio, bool]:
    if not repository.events() or _market_date(portfolio.as_of) >= _market_date(as_of):
        return portfolio, False
    from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

    clone = InMemoryPaperLedgerRepository(repository.events())
    return close_paper_day(clone, next_day_as_of=as_of), True


def _mark_portfolio(
    portfolio: PaperPortfolio,
    quotes: Sequence[MarketQuote],
    as_of: datetime,
    risk: RiskPolicy,
) -> PaperPortfolio:
    from src.risk_engine_paper_v1.application import mark_to_market

    return mark_to_market(
        portfolio,
        quotes,
        as_of,
        max_stale_market_age=risk.max_stale_market_age,
        allow_incomplete=True,
    )


def _replay_or_initial(repository: PaperLedgerRepository, as_of: datetime) -> PaperPortfolio:
    events = repository.events()
    return replay_portfolio(events) if events else initial_paper_portfolio(as_of)


def _successful_steps(as_of: datetime, *, execute: bool) -> list[PaperOperationStep]:
    return [
        _step(
            name,
            PaperOperationStepStatus.SUCCESS
            if execute or name != PaperOperationStepName.PAPER_EXECUTION
            else PaperOperationStepStatus.SKIPPED,
            as_of,
        )
        for name in PaperOperationStepName
    ]


def _blocked_steps(
    as_of: datetime, blocked_at: PaperOperationStepName, reason: str
) -> list[PaperOperationStep]:
    result: list[PaperOperationStep] = []
    blocked_seen = False
    for name in PaperOperationStepName:
        if name == blocked_at:
            blocked_seen = True
            result.append(_step(name, PaperOperationStepStatus.BLOCKED, as_of, reason))
        elif blocked_seen:
            result.append(_step(name, PaperOperationStepStatus.SKIPPED, as_of))
        else:
            result.append(_step(name, PaperOperationStepStatus.SUCCESS, as_of))
    return result


def _step(
    name: PaperOperationStepName,
    status: PaperOperationStepStatus,
    as_of: datetime,
    reason: str | None = None,
) -> PaperOperationStep:
    return PaperOperationStep(
        name=name,
        status=status,
        started_at=as_of,
        completed_at=as_of,
        duration_ms=0,
        reason=reason,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("OPERATION_AS_OF_MUST_BE_TIMEZONE_AWARE")
    return value.astimezone(UTC)


def _market_date(value: datetime) -> Any:
    return value.astimezone(ZoneInfo("Europe/Moscow")).date()


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _risk_data_degraded(decisions: Sequence[dict[str, Any]]) -> bool:
    data_reasons = {
        "PORTFOLIO_MARK_INCOMPLETE",
        "STALE_DATA",
        "FUTURE_MARKET_SNAPSHOT",
        "PRICE_UNAVAILABLE",
        "INVALID_PRICE",
    }
    return any(data_reasons.intersection(row.get("reason_codes", [])) for row in decisions)
