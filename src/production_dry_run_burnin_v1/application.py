from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, cast
from zoneinfo import ZoneInfo

from src.ai_trading_agent_v1.application import AgentModel, AgentModelRequest, git_sha
from src.ai_trading_agent_v1.domain import AgentModelResponse
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import build_operation_id, run_paper_operation
from src.paper_trading_operation_v1.domain import (
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationRun,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.repository import OperationAuditRepository
from src.production_dry_run_burnin_v1.domain import (
    BurninAttemptType,
    BurninCollectionStatus,
    BurninObservation,
    BurninObservationStatus,
    BurninPolicy,
    BurninReport,
    BurninRunResult,
    BurninSafety,
    BurninStatus,
    MoexSessionEvidence,
    MoexSessionStatus,
)
from src.production_dry_run_burnin_v1.policy import MoexSessionVerifier
from src.production_dry_run_burnin_v1.repository import BurninObservationRepository
from src.risk_engine_paper_v1.application import (
    initial_paper_portfolio,
    portfolio_state_sha,
    replay_portfolio,
)
from src.risk_engine_paper_v1.domain import PaperPortfolio, RiskPolicy
from src.risk_engine_paper_v1.repository import PaperLedgerRepository

DEFAULT_BIN_STATE_ROOT = Path("state/production-dry-run-burnin-v1")
EXPECTED_EXTERNAL_BLOCKERS = {
    "MARKET_CONTEXT_UNAVAILABLE",
    "STALE_MARKET",
    "BID_ABOVE_ASK",
    "MARKET_SOURCE_CLOCK_SKEW",
    "SOURCE_CLOCK_SKEW",
    "LIVE_RESEARCH_NOT_READY",
    "OPERATIONAL_BURN_IN_NOT_PASS",
    "SOURCE_FAILURE_ISOLATION_NOT_PROVEN",
    "RESEARCH_SEAL_NOT_VERIFIED",
    "AGENT_MODEL_UNAVAILABLE",
}


class TimedAgentModel:
    def __init__(self, model: AgentModel, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._model = model
        self._monotonic = monotonic
        self.model_id = model.model_id
        self.model_calls = 0
        self.latency_ms = 0

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        started = self._monotonic()
        self.model_calls += 1
        try:
            return self._model.complete(request)
        finally:
            self.latency_ms += max(0, round((self._monotonic() - started) * 1000))


def run_burnin_once(
    *,
    cycle_started_at: datetime,
    model: AgentModel,
    context_provider: Any,
    paper_repository: PaperLedgerRepository,
    audit_repository: OperationAuditRepository,
    observation_repository: BurninObservationRepository,
    session_verifier: MoexSessionVerifier,
    operation_state_root: Path,
    code_sha: str | None = None,
    policy: BurninPolicy | None = None,
    operation_policy: PaperOperationPolicy | None = None,
    risk_policy: RiskPolicy | None = None,
    retry_index: int = 0,
    retry_reason: str | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    operation_runner: Callable[..., PaperOperationRun] = run_paper_operation,
) -> BurninRunResult:
    fixed_policy = policy or BurninPolicy()
    paper_policy = operation_policy or PaperOperationPolicy()
    _assert_safety_policy(fixed_policy, paper_policy)
    now = clock or (lambda: datetime.now(UTC))
    cycle_start = _utc(cycle_started_at)
    trading_date = cycle_start.astimezone(ZoneInfo(fixed_policy.operation_timezone)).date()
    attempt_type = BurninAttemptType.RETRY if retry_index else BurninAttemptType.PRIMARY
    if attempt_type == BurninAttemptType.RETRY and not retry_reason:
        raise ValueError("RETRY_REASON_REQUIRED")
    operation_slot = (
        fixed_policy.primary_operation_slot
        if retry_index == 0
        else f"{fixed_policy.primary_operation_slot}_RETRY_{retry_index}"
    )
    primary_operation_id = build_operation_id(
        operation_as_of=cycle_start,
        session=fixed_policy.primary_operation_slot,
        timezone_name=fixed_policy.operation_timezone,
    )
    operation_id = build_operation_id(
        operation_as_of=cycle_start,
        session=operation_slot,
        timezone_name=fixed_policy.operation_timezone,
    )
    existing = observation_repository.primary(primary_operation_id)
    if existing is not None and attempt_type == BurninAttemptType.PRIMARY:
        return BurninRunResult(
            status="ALREADY_OBSERVED",
            existing_observation_id=existing.observation_id,
        )

    before = _paper_snapshot(paper_repository, cycle_start)
    recovered = audit_repository.completed_run(operation_id)
    if recovered is not None:
        after = _paper_snapshot(paper_repository, cycle_start)
        observation = _observation_from_run(
            run=recovered,
            attempt_type=attempt_type,
            retry_index=retry_index,
            retry_reason=retry_reason,
            primary_operation_id=primary_operation_id,
            operation_slot=operation_slot,
            trading_date=trading_date.isoformat(),
            completed_at=_completion_time(now(), recovered.operation_as_of),
            operation_duration_ms=0,
            model_calls=0,
            model_latency_ms=0,
            before=before,
            after=after,
            session=_recovered_session(trading_date.isoformat(), cycle_start),
            policy=fixed_policy,
            status_code_override="RECOVERED_FROM_OPERATION_AUDIT",
        )
        appended = observation_repository.append(observation)
        return BurninRunResult(status=observation.status.value, observation=appended)

    session = session_verifier.verify(trading_date)
    if session.status != MoexSessionStatus.OPEN:
        reason = (
            "MARKET_SESSION_CLOSED"
            if session.status == MoexSessionStatus.CLOSED
            else "MARKET_SESSION_UNKNOWN"
        )
        completed = _completion_time(now(), cycle_start)
        observation = _empty_observation(
            operation_id=operation_id,
            primary_operation_id=primary_operation_id,
            operation_slot=operation_slot,
            attempt_type=attempt_type,
            retry_index=retry_index,
            retry_reason=retry_reason,
            trading_date=trading_date.isoformat(),
            cycle_started_at=cycle_start,
            completed_at=completed,
            code_sha=code_sha or git_sha(),
            model_id=model.model_id,
            before=before,
            session=session,
            reason=reason,
            policy=fixed_policy,
        )
        appended = observation_repository.append(observation)
        return BurninRunResult(status=observation.status.value, observation=appended)

    timed_model = TimedAgentModel(model, monotonic)
    started = monotonic()
    try:
        run = operation_runner(
            operation_as_of=cycle_start,
            mode=PaperOperationMode.DRY_RUN,
            model=timed_model,
            context_provider=context_provider,
            paper_repository=paper_repository,
            audit_repository=audit_repository,
            state_root=operation_state_root,
            code_sha=code_sha or git_sha(),
            policy=paper_policy,
            risk_policy=risk_policy or RiskPolicy(),
            operation_slot=operation_slot,
        )
        operation_duration_ms = max(0, round((monotonic() - started) * 1000))
        after = _paper_snapshot(paper_repository, cycle_start)
        completed = _completion_time(now(), run.operation_as_of)
        observation = _observation_from_run(
            run=run,
            attempt_type=attempt_type,
            retry_index=retry_index,
            retry_reason=retry_reason,
            primary_operation_id=primary_operation_id,
            operation_slot=operation_slot,
            trading_date=trading_date.isoformat(),
            completed_at=completed,
            operation_duration_ms=operation_duration_ms,
            model_calls=timed_model.model_calls,
            model_latency_ms=timed_model.latency_ms,
            before=before,
            after=after,
            session=session,
            policy=fixed_policy,
        )
    except Exception as exc:
        operation_duration_ms = max(0, round((monotonic() - started) * 1000))
        after = _paper_snapshot(paper_repository, cycle_start)
        completed = _completion_time(now(), cycle_start)
        observation = _empty_observation(
            operation_id=operation_id,
            primary_operation_id=primary_operation_id,
            operation_slot=operation_slot,
            attempt_type=attempt_type,
            retry_index=retry_index,
            retry_reason=retry_reason,
            trading_date=trading_date.isoformat(),
            cycle_started_at=cycle_start,
            completed_at=completed,
            code_sha=code_sha or git_sha(),
            model_id=model.model_id,
            before=before,
            after=after,
            session=session,
            reason=f"BURNIN_INTERNAL_FAILURE:{type(exc).__name__}",
            policy=fixed_policy,
            operation_duration_ms=operation_duration_ms,
            model_calls=timed_model.model_calls,
            model_latency_ms=timed_model.latency_ms,
            status=BurninObservationStatus.FAIL,
        )
    appended = observation_repository.append(observation)
    return BurninRunResult(status=appended.status.value, observation=appended)


def build_burnin_report(
    observations: Sequence[BurninObservation],
    policy: BurninPolicy | None = None,
) -> BurninReport:
    fixed = policy or BurninPolicy()
    primary = [row for row in observations if row.attempt_type == BurninAttemptType.PRIMARY]
    distinct_days = len(
        {row.market_date for row in primary if row.session.status == MoexSessionStatus.OPEN}
    )
    passes = sum(_valid_cycle(row) for row in primary)
    blocked = sum(row.status == BurninObservationStatus.BLOCKED_EXPECTED for row in primary)
    failed = len(primary) - passes - blocked
    total_quotes = sum(
        row.fresh_quote_count
        + row.stale_quote_count
        + row.future_quote_count
        + row.missing_quote_count
        + row.invalid_quote_count
        for row in primary
    )
    market_rate = _rate(sum(row.fresh_quote_count for row in primary), total_quotes)
    research_rate = _rate(
        sum(row.research_operation_status == "READY" for row in primary), len(primary)
    )
    agent_rate = _rate(sum(row.agent_decision_status == "VALID" for row in primary), len(primary))
    risk_rate = _rate(sum(row.risk_plan_created for row in primary), len(primary))
    future = sum(row.future_quote_count for row in observations)
    paper_mutations = sum(row.safety.PAPER_PORTFOLIO_MUTATIONS for row in observations)
    real_mutations = sum(
        row.safety.REAL_BROKER_MUTATIONS
        + row.safety.REAL_ORDERS_SENT
        + row.safety.REAL_ORDERS_CANCELLED
        + row.safety.REAL_POSITIONS_CHANGED
        for row in observations
    )
    holdout = sum(row.safety.OLD_FUTURE_HOLDOUT_OPENED for row in observations)
    complete = (
        distinct_days >= fixed.min_distinct_moex_trading_days
        and len(primary) >= fixed.min_primary_cycles
    )
    blocked_rate = _rate(blocked, len(primary))
    zero_safety = future == 0 and paper_mutations == 0 and real_mutations == 0 and holdout == 0
    source_clock_pass = all(
        row.max_source_clock_delta_seconds <= fixed.max_source_clock_skew_seconds
        for row in observations
    )
    ready = (
        complete
        and _rate(passes, len(primary)) > 0
        and blocked_rate <= fixed.max_blocked_primary_cycle_rate
        and market_rate >= fixed.min_market_fresh_rate
        and research_rate >= fixed.min_research_ready_rate
        and agent_rate >= fixed.min_agent_valid_rate
        and risk_rate >= fixed.min_risk_completion_rate
        and zero_safety
        and source_clock_pass
        and all(row.paper_ledger_unchanged for row in observations)
    )
    safety_violation = not zero_safety or any(
        row.status == BurninObservationStatus.SAFETY_VIOLATION for row in observations
    )
    burnin_status = (
        BurninStatus.NOT_STARTED
        if not observations
        else BurninStatus.FAIL
        if safety_violation
        else BurninStatus.PASS
        if ready
        else BurninStatus.FAIL
        if complete
        else BurninStatus.IN_PROGRESS
    )
    actions: Counter[str] = Counter()
    for row in observations:
        actions.update(row.proposal_action_counts)
    actions["NO_ACTION"] = sum(
        row.operation_status == PaperOperationStatus.NO_ACTION.value for row in observations
    )
    actions["risk_rejection_count"] = sum(row.risk_rejected_count for row in observations)
    actions["risk_reduction_count"] = sum(row.risk_reduced_count for row in observations)
    tool_total = sum(row.tool_call_count for row in observations)
    tool_failures = sum(
        call.get("status") not in {"SUCCESS", "OK"}
        for row in observations
        for call in row.tool_calls
    )
    degraded = sum(row.operation_status == PaperOperationStatus.DEGRADED.value for row in primary)
    timeout_count = sum("TIMEOUT" in row.status_code for row in observations)
    model_error_count = sum(
        row.status_code.startswith("AGENT_MODEL_") and "TIMEOUT" not in row.status_code
        for row in observations
    )
    return BurninReport(
        BURNIN_POLICY_VERSION=fixed.policy_version,
        BURNIN_COLLECTION_STATUS=(
            BurninCollectionStatus.COMPLETE if complete else BurninCollectionStatus.IN_PROGRESS
        ),
        BURNIN_STATUS=burnin_status,
        PRODUCTION_DRY_RUN_BURNIN_READY="YES",
        BURNIN_LEDGER_INTEGRITY="PASS",
        distinct_trading_days=distinct_days,
        valid_cycles=passes,
        cycle_count=len(primary),
        successful_cycle_count=passes,
        blocked_cycle_count=blocked,
        degraded_cycle_count=degraded,
        primary_cycles=len(primary),
        primary_pass_cycles=passes,
        primary_blocked_cycles=blocked,
        primary_failed_cycles=failed,
        primary_pass_rate=_rate(passes, len(primary)),
        blocked_rate=blocked_rate,
        market_fresh_rate=market_rate,
        market_ready_rate=market_rate,
        research_ready_rate=research_rate,
        agent_valid_rate=agent_rate,
        risk_completion_rate=risk_rate,
        dry_run_end_to_end_rate=_rate(passes, len(primary)),
        model_timeout_count=timeout_count,
        model_error_count=model_error_count,
        median_cycle_duration_ms=_median(row.duration_ms for row in primary),
        p95_cycle_duration_ms=_p95(row.duration_ms for row in primary),
        duplicate_prevented_count=0,
        tool_call_total=tool_total,
        tool_failure_total=tool_failures,
        tool_failure_rate=_rate(tool_failures, tool_total),
        mean_market_fetch_ms=_mean(row.market_fetch_duration_ms for row in primary),
        p95_market_fetch_ms=_p95(row.market_fetch_duration_ms for row in primary),
        mean_model_latency_ms=_mean(row.model_latency_ms for row in primary),
        p95_model_latency_ms=_p95(row.model_latency_ms for row in primary),
        mean_total_cycle_ms=_mean(row.operation_duration_ms for row in primary),
        p95_total_cycle_ms=_p95(row.operation_duration_ms for row in primary),
        future_data_violation_count=future,
        paper_mutation_count=paper_mutations,
        real_mutation_count=real_mutations,
        holdout_violation_count=holdout,
        proposal_statistics=dict(sorted(actions.items())),
        safety_behavior="PASS" if zero_safety else "FAIL",
        last_observation_id=observations[-1].observation_id if observations else None,
    )


def _observation_from_run(
    *,
    run: PaperOperationRun,
    attempt_type: BurninAttemptType,
    retry_index: int,
    retry_reason: str | None,
    primary_operation_id: str,
    operation_slot: str,
    trading_date: str,
    completed_at: datetime,
    operation_duration_ms: int,
    model_calls: int,
    model_latency_ms: int,
    before: tuple[int, str, str],
    after: tuple[int, str, str],
    session: MoexSessionEvidence,
    policy: BurninPolicy,
    status_code_override: str | None = None,
) -> BurninObservation:
    market = run.market_audit
    research = run.research_status
    unchanged = before == after
    safety = _safety(run, policy)
    if not unchanged:
        safety = safety.model_copy(
            update={"PAPER_PORTFOLIO_MUTATIONS": max(1, safety.PAPER_PORTFOLIO_MUTATIONS)}
        )
    status = _classify(run, unchanged, safety)
    decisions = run.risk_decisions
    actions = Counter(str(row.get("action", "UNKNOWN")) for row in run.agent_proposals)
    ages = [
        float(row["age_seconds"])
        for row in cast("list[dict[str, Any]]", market.get("quotes", []))
        if row.get("age_seconds") is not None and float(row["age_seconds"]) >= 0
    ]
    seal = cast("dict[str, Any]", research.get("seal", {}))
    fetch_started = _optional_time(market.get("market_fetch_started_at"))
    fetch_completed = _optional_time(market.get("market_fetch_completed_at"))
    market_duration = (
        max(0, round((fetch_completed - fetch_started).total_seconds() * 1000))
        if fetch_started is not None and fetch_completed is not None
        else 0
    )
    reasons = list(run.reasons)
    if not unchanged:
        reasons.append("BURNIN_PAPER_LEDGER_MUTATED")
    if safety.violation_count():
        reasons.append("BURNIN_SAFETY_VIOLATION")
    status_code = status_code_override or run.status_code
    observation_id = _observation_id(run.operation_id, retry_index)
    tool_calls: list[dict[str, object]] = [
        {
            "tool_name": str(row.get("tool_name", "unknown")),
            "status": str(row.get("status", "UNKNOWN")),
            "latency_ms": 0,
            "result_hash": row.get("result_hash"),
        }
        for row in run.agent_tool_calls
    ]
    approved = _decision_count(decisions, "APPROVE")
    reduced = _decision_count(decisions, "REDUCE")
    rejected = _decision_count(decisions, "REJECT")
    research_operation_status = str(research.get("LIVE_RESEARCH_OPERATION_STATUS", "UNKNOWN"))
    operational_burnin_status = str(research.get("OPERATIONAL_BURN_IN", "UNKNOWN"))
    return BurninObservation(
        sequence=1,
        previous_record_sha=None,
        record_sha="PENDING",
        observation_id=observation_id,
        burnin_observation_id=observation_id,
        trading_date=trading_date,
        market_date=trading_date,
        operation_slot=operation_slot,
        operation_session=operation_slot,
        operation_id=run.operation_id,
        operation_slot_id=run.operation_slot_id or f"{trading_date}:{operation_slot}",
        attempt_type=attempt_type,
        primary_operation_id=primary_operation_id,
        retry_index=retry_index,
        retry_reason=retry_reason,
        cycle_started_at=run.cycle_started_at or run.operation_as_of,
        decision_as_of=run.decision_as_of or run.operation_as_of,
        completed_at=completed_at,
        code_sha=run.code_sha,
        policy_version=policy.policy_version,
        prompt_version=run.prompt_version,
        agent_model_id=run.agent_model_id,
        market_adapter_id=run.market_adapter_id or "unknown",
        universe_sha=run.universe_sha or sha256_payload(run.universe),
        operation_contract_sha=run.operation_contract_sha or "unknown",
        market_snapshot_sha=run.market_snapshot_sha or sha256_payload([]),
        event_snapshot_sha=run.event_snapshot_sha or sha256_payload({}),
        research_status_sha=run.research_status_sha or sha256_payload(research),
        portfolio_sha=run.portfolio_before_sha,
        market_fetch_started_at=fetch_started,
        market_fetch_completed_at=fetch_completed,
        market_source_clock_delta_seconds=float(
            market.get("market_source_clock_delta_seconds") or 0.0
        ),
        market_fetch_duration_ms=market_duration,
        operation_duration_ms=operation_duration_ms,
        model_latency_ms=model_latency_ms,
        universe_count=len(run.universe),
        market_quote_count=int(market.get("quote_count", 0)),
        fresh_quote_count=int(market.get("fresh_quote_count", 0)),
        stale_quote_count=int(market.get("stale_quote_count", 0)),
        future_quote_count=int(market.get("future_quote_count", 0)),
        missing_quote_count=int(market.get("missing_quote_count", 0)),
        invalid_quote_count=int(market.get("invalid_quote_count", 0)),
        max_market_age_seconds=max(ages, default=0.0),
        max_source_clock_delta_seconds=float(
            market.get("market_source_clock_delta_seconds") or 0.0
        ),
        research_status=research_operation_status,
        research_operation_status=research_operation_status,
        operational_burnin_status=operational_burnin_status,
        source_failure_isolation=research.get("SOURCE_FAILURE_ISOLATION") is True,
        seal_verified=(
            seal.get("sealed_epoch_verified") is True and int(seal.get("violations", 0)) == 0
        ),
        research_seal_verified=(
            seal.get("sealed_epoch_verified") is True and int(seal.get("violations", 0)) == 0
        ),
        agent_decision_status="VALID" if run.agent_run_id and run.risk_plan_id else "NOT_VALID",
        agent_steps=sum(step.name.value == "AGENT" for step in run.steps),
        agent_proposal_count=len(run.agent_proposals),
        proposal_count=len(run.agent_proposals),
        proposal_action_counts=dict(sorted(actions.items())),
        agent_tool_call_count=len(run.agent_tool_calls),
        tool_call_count=len(run.agent_tool_calls),
        tool_calls=tool_calls,
        validation_reasons=list(run.reasons),
        model_calls=model_calls,
        risk_plan_created=run.risk_plan_id is not None,
        risk_decision_count=len(decisions),
        risk_approved_count=approved,
        approved_count=approved,
        risk_reduced_count=reduced,
        reduced_count=reduced,
        risk_rejected_count=rejected,
        rejected_count=rejected,
        paper_orders_planned=len(run.paper_order_ids),
        operation_status=run.status.value,
        status=status,
        status_code=status_code,
        operation_status_code=run.status_code,
        reasons=list(dict.fromkeys(reasons)),
        paper_ledger_event_count_before=before[0],
        paper_ledger_event_count_after=after[0],
        paper_ledger_sha_before=before[1],
        paper_ledger_sha_after=after[1],
        portfolio_sha_before=before[2],
        portfolio_sha_after=after[2],
        paper_ledger_unchanged=unchanged,
        paper_orders_filled=safety.PAPER_ORDERS_FILLED,
        paper_portfolio_mutations=safety.PAPER_PORTFOLIO_MUTATIONS,
        duration_ms=operation_duration_ms,
        session=session,
        safety=safety,
    )


def _empty_observation(
    *,
    operation_id: str,
    primary_operation_id: str,
    operation_slot: str,
    attempt_type: BurninAttemptType,
    retry_index: int,
    retry_reason: str | None,
    trading_date: str,
    cycle_started_at: datetime,
    completed_at: datetime,
    code_sha: str,
    model_id: str,
    before: tuple[int, str, str],
    session: MoexSessionEvidence,
    reason: str,
    policy: BurninPolicy,
    after: tuple[int, str, str] | None = None,
    operation_duration_ms: int = 0,
    model_calls: int = 0,
    model_latency_ms: int = 0,
    status: BurninObservationStatus = BurninObservationStatus.BLOCKED_EXPECTED,
) -> BurninObservation:
    final = after or before
    unchanged = before == final
    if not unchanged:
        status = BurninObservationStatus.SAFETY_VIOLATION
    observation_id = _observation_id(operation_id, retry_index)
    return BurninObservation(
        sequence=1,
        previous_record_sha=None,
        record_sha="PENDING",
        observation_id=observation_id,
        burnin_observation_id=observation_id,
        trading_date=trading_date,
        market_date=trading_date,
        operation_slot=operation_slot,
        operation_session=operation_slot,
        operation_id=operation_id,
        operation_slot_id=f"{trading_date}:{operation_slot}",
        attempt_type=attempt_type,
        primary_operation_id=primary_operation_id,
        retry_index=retry_index,
        retry_reason=retry_reason,
        cycle_started_at=cycle_started_at,
        decision_as_of=cycle_started_at,
        completed_at=completed_at,
        code_sha=code_sha,
        policy_version=policy.policy_version,
        prompt_version="not-called",
        agent_model_id=model_id,
        market_adapter_id="not-called",
        universe_sha=sha256_payload([]),
        operation_contract_sha="not-created",
        market_snapshot_sha=sha256_payload([]),
        event_snapshot_sha=sha256_payload({}),
        research_status_sha=sha256_payload({}),
        portfolio_sha=before[2],
        operation_duration_ms=operation_duration_ms,
        model_latency_ms=model_latency_ms,
        research_status="NOT_LOADED",
        research_operation_status="NOT_LOADED",
        operational_burnin_status="NOT_LOADED",
        source_failure_isolation=False,
        seal_verified=False,
        agent_decision_status="NOT_CALLED",
        model_calls=model_calls,
        risk_plan_created=False,
        operation_status="NOT_STARTED",
        status=status,
        status_code=reason,
        operation_status_code=reason,
        reasons=[reason],
        paper_ledger_event_count_before=before[0],
        paper_ledger_event_count_after=final[0],
        paper_ledger_sha_before=before[1],
        paper_ledger_sha_after=final[1],
        portfolio_sha_before=before[2],
        portfolio_sha_after=final[2],
        paper_ledger_unchanged=unchanged,
        duration_ms=operation_duration_ms,
        session=session,
        safety=BurninSafety(PAPER_PORTFOLIO_MUTATIONS=0 if unchanged else 1),
    )


def _paper_snapshot(repository: PaperLedgerRepository, as_of: datetime) -> tuple[int, str, str]:
    events = repository.events()
    portfolio: PaperPortfolio = (
        replay_portfolio(events) if events else initial_paper_portfolio(as_of)
    )
    payload = [row.model_dump(mode="json") for row in events]
    return len(events), sha256_payload(payload), portfolio_state_sha(portfolio)


def _safety(run: PaperOperationRun, policy: BurninPolicy) -> BurninSafety:
    source = run.safety
    return BurninSafety(
        PAPER_EXECUTION_ENABLED=policy.paper_execution_enabled,
        REAL_EXECUTION_ENABLED=policy.real_execution_enabled,
        PAPER_OPERATION_SCHEDULE_ENABLED=policy.paper_operation_schedule_enabled,
        PAPER_ORDERS_FILLED=source.PAPER_ORDERS_FILLED,
        PAPER_PORTFOLIO_MUTATIONS=source.PAPER_PORTFOLIO_MUTATIONS,
        REAL_BROKER_MUTATIONS=source.REAL_BROKER_MUTATIONS,
        REAL_ORDERS_SENT=source.REAL_ORDERS_SENT,
        REAL_ORDERS_CANCELLED=source.REAL_ORDERS_CANCELLED,
        REAL_POSITIONS_CHANGED=source.REAL_POSITIONS_CHANGED,
        LIVE_OUTCOMES_READ=source.LIVE_OUTCOMES_READ,
        LIVE_TARGETS_COMPUTED=source.LIVE_TARGETS_COMPUTED,
        LIVE_POST_EVENT_PRICE_READS=source.LIVE_POST_EVENT_PRICE_READS,
        MODEL_TRAINING_PERFORMED=source.MODEL_TRAINING_PERFORMED,
        BACKTEST_PERFORMED=source.BACKTEST_PERFORMED,
        OLD_FUTURE_HOLDOUT_OPENED=source.OLD_FUTURE_HOLDOUT_OPENED,
        PAID_SOURCE_CALLS=source.PAID_SOURCE_CALLS,
    )


def _classify(
    run: PaperOperationRun,
    paper_unchanged: bool,
    safety: BurninSafety,
) -> BurninObservationStatus:
    future_quotes = int(run.market_audit.get("future_quote_count", 0))
    if not paper_unchanged or safety.violation_count() or future_quotes:
        return BurninObservationStatus.SAFETY_VIOLATION
    if run.mode != PaperOperationMode.DRY_RUN or run.paper_execution_status not in {
        None,
        "SKIPPED_DRY_RUN",
    }:
        return BurninObservationStatus.SAFETY_VIOLATION
    if (
        run.agent_run_id
        and run.risk_plan_id
        and run.status
        in {
            PaperOperationStatus.SUCCESS,
            PaperOperationStatus.NO_ACTION,
            PaperOperationStatus.DEGRADED,
        }
    ):
        return BurninObservationStatus.PASS
    if run.status == PaperOperationStatus.BLOCKED and any(
        _expected_blocker(reason) for reason in [run.status_code, *run.reasons]
    ):
        return BurninObservationStatus.BLOCKED_EXPECTED
    return BurninObservationStatus.FAIL


def _assert_safety_policy(policy: BurninPolicy, operation_policy: PaperOperationPolicy) -> None:
    if (
        policy.paper_execution_enabled
        or policy.real_execution_enabled
        or policy.paper_operation_schedule_enabled
        or operation_policy.paper_execution_enabled
        or operation_policy.real_execution_enabled
        or operation_policy.operation_schedule_enabled
    ):
        raise ValueError("BURNIN_EXECUTION_CAPABILITY_FORBIDDEN")


def _expected_blocker(reason: str) -> bool:
    return any(
        reason == candidate or reason.startswith(f"{candidate}:")
        for candidate in EXPECTED_EXTERNAL_BLOCKERS
    )


def _decision_count(rows: list[dict[str, Any]], value: str) -> int:
    return sum(row.get("risk_decision") == value for row in rows)


def _valid_cycle(row: BurninObservation) -> bool:
    return (
        row.status == BurninObservationStatus.PASS
        and row.session.status == MoexSessionStatus.OPEN
        and row.paper_ledger_unchanged
        and row.safety.violation_count() == 0
        and bool(row.operation_id and row.operation_slot_id and row.operation_contract_sha)
        and row.operation_status != "NOT_STARTED"
    )


def _observation_id(operation_id: str, retry_index: int) -> str:
    return f"burnin-observation-{sha256_payload([operation_id, retry_index])[:24]}"


def _optional_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _completion_time(value: datetime, floor: datetime) -> datetime:
    parsed = _utc(value)
    return max(parsed, _utc(floor))


def _recovered_session(trading_date: str, checked_at: datetime) -> MoexSessionEvidence:
    return MoexSessionEvidence(
        trading_date=trading_date,
        status=MoexSessionStatus.OPEN,
        source="RECOVERED_OPERATION_AUDIT",
        checked_at=checked_at,
        reason="RECOVERED_FROM_OPERATION_AUDIT",
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _mean(values: Any) -> float:
    rows = list(values)
    return 0.0 if not rows else round(mean(rows), 3)


def _median(values: Any) -> float:
    rows = list(values)
    return 0.0 if not rows else round(median(rows), 3)


def _p95(values: Any) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    index = max(0, round(0.95 * len(rows) + 0.5) - 1)
    return round(rows[min(index, len(rows) - 1)], 3)
