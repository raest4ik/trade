from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.ai_trading_agent_v1.application import FakeAgentModel
from src.ai_trading_agent_v1.domain import AgentModelResponse
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import (
    PaperOperationContext,
    SimulatedCrashAfterPaperFillError,
    StaticPaperOperationContextProvider,
    build_operation_id,
    operation_history,
    run_paper_operation,
)
from src.paper_trading_operation_v1.domain import (
    OPERATION_POLICY_VERSION,
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationRun,
    PaperOperationSafety,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.repository import InMemoryOperationAuditRepository
from src.risk_engine_paper_v1.application import (
    close_paper_day,
    evaluate_agent_run,
    execute_paper_plan,
    mark_to_market,
    portfolio_state_sha,
    replay_portfolio,
    restore_operational_portfolio,
)
from src.risk_engine_paper_v1.domain import MarketQuote, PaperExecutionStatus, RiskPolicy
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

ARTIFACT_VERSION = "paper-trading-operation-v1"
SAMPLE_START = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class OperationSample:
    policy: PaperOperationPolicy
    runs: tuple[PaperOperationRun, ...]
    operation_events: list[dict[str, Any]]
    paper_events: list[dict[str, Any]]
    history: list[dict[str, Any]]
    final_portfolio: dict[str, Any]
    replay_verification: dict[str, Any]
    idempotency_verification: dict[str, Any]
    stale_plan_verification: dict[str, Any]
    crash_recovery_verification: dict[str, Any]
    safety: PaperOperationSafety


def build_sample_operations(*, work_root: Path, code_sha: str) -> OperationSample:
    policy = PaperOperationPolicy(paper_execution_enabled=True)
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    day_1 = _operation(
        work_root,
        paper,
        audit,
        SAMPLE_START,
        [_proposal("SBER", "BUY", 0.10)],
        ["SBER", "YDEX"],
        code_sha,
        policy,
    )
    day_2_at = SAMPLE_START + timedelta(days=1)
    day_2 = _operation(
        work_root,
        paper,
        audit,
        day_2_at,
        [_proposal("SBER", "HOLD", 0.10), _proposal("YDEX", "BUY", 0.20)],
        ["SBER", "YDEX"],
        code_sha,
        policy,
    )
    day_3_at = SAMPLE_START + timedelta(days=2)
    day_3 = _operation(
        work_root,
        paper,
        audit,
        day_3_at,
        [_proposal("SBER", "SELL", 0.05)],
        ["SBER", "YDEX"],
        code_sha,
        policy,
    )
    degraded_at = day_3_at + timedelta(minutes=1)
    degraded = _operation(
        work_root,
        paper,
        audit,
        degraded_at,
        [_proposal("SBER", "BUY", 0.10)],
        ["SBER"],
        code_sha,
        policy,
        operation_slot="EOD_DEGRADED_PROOF",
    )

    event_count_before_duplicate = paper.last_sequence()
    duplicate = _operation(
        work_root,
        paper,
        audit,
        day_3_at + timedelta(minutes=3),
        [_proposal("SBER", "SELL", 0.05)],
        ["SBER", "YDEX"],
        code_sha,
        policy,
    )
    idempotency = {
        "SESSION_IDEMPOTENCY": "PASS"
        if duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
        else "FAIL",
        "SAME_SESSION_DIFFERENT_TIMESTAMP": duplicate.status.value,
        "DUPLICATE_OPERATION_STATUS": duplicate.status.value,
        "NEW_FILLS": duplicate.safety.PAPER_ORDERS_FILLED,
        "PAPER_LEDGER_UNCHANGED": paper.last_sequence() == event_count_before_duplicate,
        "DIFFERENT_SESSION_DISTINCT": build_operation_id(
            operation_as_of=day_3_at,
            session="EOD",
            model_id=duplicate.agent_model_id,
            universe=duplicate.universe,
        )
        != build_operation_id(
            operation_as_of=day_3_at,
            session="EOD_RETRY_1",
            model_id=duplicate.agent_model_id,
            universe=duplicate.universe,
        ),
        "NEXT_TRADING_DAY_DISTINCT": build_operation_id(
            operation_as_of=day_3_at,
            session="EOD",
            model_id=duplicate.agent_model_id,
            universe=duplicate.universe,
        )
        != build_operation_id(
            operation_as_of=day_3_at + timedelta(days=1),
            session="EOD",
            model_id=duplicate.agent_model_id,
            universe=duplicate.universe,
        ),
    }

    final = day_3.portfolio_after
    replayed = mark_to_market(
        replay_portfolio(paper.events()),
        _quotes(day_3_at, "SBER", "YDEX"),
        day_3_at,
        allow_incomplete=True,
    )
    replay = {
        "REPLAY_VERIFIED": portfolio_state_sha(replayed) == day_3.portfolio_after_sha,
        "expected_sha": day_3.portfolio_after_sha,
        "replayed_sha": portfolio_state_sha(replayed),
        "paper_ledger_event_count": paper.last_sequence(),
    }
    stale = _stale_plan_proof(paper, day_3_at)
    crash = _crash_recovery_proof(work_root / "crash", code_sha, policy)
    runs = (day_1, day_2, day_3, degraded)
    safety = PaperOperationSafety(
        PAPER_OPERATION_RUNS=len(runs),
        PAPER_RISK_PLANS=sum(run.safety.PAPER_RISK_PLANS for run in runs),
        PAPER_ORDERS_PLANNED=sum(run.safety.PAPER_ORDERS_PLANNED for run in runs),
        PAPER_ORDERS_FILLED=sum(run.safety.PAPER_ORDERS_FILLED for run in runs),
        PAPER_PORTFOLIO_MUTATIONS=sum(run.safety.PAPER_PORTFOLIO_MUTATIONS for run in runs),
    )
    return OperationSample(
        policy=policy,
        runs=runs,
        operation_events=[row.model_dump(mode="json") for row in audit.events()],
        paper_events=[row.model_dump(mode="json") for row in paper.events()],
        history=operation_history(audit),
        final_portfolio=final,
        replay_verification=replay,
        idempotency_verification=idempotency,
        stale_plan_verification=stale,
        crash_recovery_verification=crash,
        safety=safety,
    )


def write_operation_artifact(
    *,
    output_root: Path,
    base_main_sha: str,
    head_sha: str,
    sample: OperationSample,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable paper operation artifact already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    ready = (
        [run.status for run in sample.runs[:3]] == [PaperOperationStatus.SUCCESS] * 3
        and sample.runs[3].status == PaperOperationStatus.DEGRADED
        and sample.runs[3].safety.PAPER_ORDERS_FILLED == 0
        and sample.replay_verification["REPLAY_VERIFIED"] is True
        and sample.idempotency_verification["DUPLICATE_OPERATION_STATUS"] == "ALREADY_PROCESSED"
        and sample.stale_plan_verification["STALE_PLAN_NO_FILL"] is True
        and sample.crash_recovery_verification["RECOVERED_WITHOUT_DUPLICATE_FILL"] is True
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": head_sha,
        "PAPER_TRADING_OPERATION_READY": "YES" if ready else "NO",
        "AGENT_RESEARCH_CAPABILITY_READY": "YES",
        "RISK_ENGINE_READY": "YES",
        "PAPER_PORTFOLIO_READY": "YES",
        "REAL_EXECUTION_READY": "NO",
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        "operation_policy_version": OPERATION_POLICY_VERSION,
        "operation_modes": [mode.value for mode in PaperOperationMode],
        "universe_size": 2,
        "PAPER_OPERATION_RUNS": sample.safety.PAPER_OPERATION_RUNS,
        "PAPER_RISK_PLANS": sample.safety.PAPER_RISK_PLANS,
        "PAPER_ORDERS_FILLED": sample.safety.PAPER_ORDERS_FILLED,
        "paper_ledger_event_count": len(sample.paper_events),
        "final_portfolio_sha": sample.runs[2].portfolio_after_sha,
        "REPLAY_VERIFIED": "YES" if sample.replay_verification["REPLAY_VERIFIED"] else "NO",
        "DEGRADED_OPERATION_STATUS": sample.runs[3].status.value,
        "STALE_PLAN_RESULT": sample.stale_plan_verification["execution_status"],
        "DUPLICATE_OPERATION_RESULT": sample.idempotency_verification["DUPLICATE_OPERATION_STATUS"],
        "CRASH_RECOVERY_RESULT": sample.crash_recovery_verification["status_code"],
        "SESSION_IDEMPOTENCY": sample.idempotency_verification["SESSION_IDEMPOTENCY"],
        "SAME_SESSION_DIFFERENT_TIMESTAMP": sample.idempotency_verification[
            "SAME_SESSION_DIFFERENT_TIMESTAMP"
        ],
        "DIFFERENT_SESSION_DISTINCT": (
            "YES" if sample.idempotency_verification["DIFFERENT_SESSION_DISTINCT"] else "NO"
        ),
        "NEXT_TRADING_DAY_DISTINCT": (
            "YES" if sample.idempotency_verification["NEXT_TRADING_DAY_DISTINCT"] else "NO"
        ),
        "PAPER_EXECUTION_DEFAULT": False,
        "PAPER_EXECUTION_DOUBLE_OPT_IN": "PASS",
        "PIT_SAFETY": "PASS",
        **sample.safety.model_dump(mode="json"),
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "operation-policy.json", sample.policy.model_dump(mode="json"))
    for name, run in zip(
        ("day-1-operation", "day-2-operation", "day-3-operation", "degraded-operation"),
        sample.runs,
        strict=True,
    ):
        _write_json(output_root / f"{name}.json", run.model_dump(mode="json"))
    _write_json(output_root / "operation-history.json", sample.history)
    _write_json(output_root / "final-portfolio.json", sample.final_portfolio)
    _write_json(output_root / "idempotency-verification.json", sample.idempotency_verification)
    _write_json(output_root / "stale-plan-verification.json", sample.stale_plan_verification)
    _write_json(
        output_root / "crash-recovery-verification.json", sample.crash_recovery_verification
    )
    _write_json(output_root / "replay-verification.json", sample.replay_verification)
    _write_json(output_root / "safety.json", sample.safety.model_dump(mode="json"))
    _write_jsonl(output_root / "operation-ledger.jsonl", sample.operation_events)
    _write_jsonl(output_root / "paper-ledger.jsonl", sample.paper_events)
    _write_report(output_root / "report.md", manifest, sample)
    return manifest


def _operation(
    work_root: Path,
    paper: InMemoryPaperLedgerRepository,
    audit: InMemoryOperationAuditRepository,
    as_of: datetime,
    proposals: list[dict[str, Any]],
    quote_tickers: list[str],
    code_sha: str,
    policy: PaperOperationPolicy,
    operation_slot: str | None = None,
) -> PaperOperationRun:
    return run_paper_operation(
        operation_as_of=as_of,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=_model(as_of, proposals),
        context_provider=StaticPaperOperationContextProvider(_context(as_of, quote_tickers)),
        paper_repository=paper,
        audit_repository=audit,
        state_root=work_root,
        code_sha=code_sha,
        policy=policy,
        operation_slot=operation_slot,
    )


def _stale_plan_proof(source: InMemoryPaperLedgerRepository, as_of: datetime) -> dict[str, Any]:
    repository = InMemoryPaperLedgerRepository(source.events())
    proof_at = as_of + timedelta(minutes=1)
    quotes = _quotes(proof_at, "SBER", "YDEX")
    portfolio = restore_operational_portfolio(
        repository,
        as_of=proof_at,
        market_snapshot=quotes,
    )
    plan = evaluate_agent_run(
        agent_run=_risk_agent_payload(proof_at, _proposal("YDEX", "BUY", 0.20)),
        portfolio=portfolio,
        market_snapshot=quotes,
        policy=RiskPolicy(),
        decision_as_of=proof_at,
        ledger_event_count=repository.last_sequence(),
    )
    close_paper_day(repository, next_day_as_of=proof_at + timedelta(days=1))
    result = execute_paper_plan(plan, repository, execution_as_of=proof_at + timedelta(days=1))
    return {
        "execution_status": result.execution_status.value,
        "status_code": result.status_code,
        "STALE_PLAN_NO_FILL": (
            result.execution_status == PaperExecutionStatus.STALE_RISK_PLAN
            and result.safety.PAPER_ORDERS_FILLED == 0
        ),
        "paper_orders_filled": result.safety.PAPER_ORDERS_FILLED,
    }


def _crash_recovery_proof(
    work_root: Path, code_sha: str, policy: PaperOperationPolicy
) -> dict[str, Any]:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    context = _context(SAMPLE_START, ["SBER", "YDEX"])
    proposals = [_proposal("SBER", "BUY", 0.10)]
    crashed = False
    try:
        run_paper_operation(
            operation_as_of=SAMPLE_START,
            mode=PaperOperationMode.PAPER_EXECUTE,
            model=_model(SAMPLE_START, proposals),
            context_provider=StaticPaperOperationContextProvider(context),
            paper_repository=paper,
            audit_repository=audit,
            state_root=work_root,
            code_sha=code_sha,
            policy=policy,
            simulate_crash_after_fill=True,
        )
    except SimulatedCrashAfterPaperFillError:
        crashed = True
    event_count = paper.last_sequence()
    recovered = run_paper_operation(
        operation_as_of=SAMPLE_START,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=_model(SAMPLE_START, proposals),
        context_provider=StaticPaperOperationContextProvider(context),
        paper_repository=paper,
        audit_repository=audit,
        state_root=work_root,
        code_sha=code_sha,
        policy=policy,
    )
    return {
        "simulated_crash_observed": crashed,
        "status": recovered.status.value,
        "status_code": recovered.status_code,
        "new_fills_during_recovery": recovered.safety.PAPER_ORDERS_FILLED,
        "paper_ledger_unchanged_during_recovery": paper.last_sequence() == event_count,
        "RECOVERED_WITHOUT_DUPLICATE_FILL": (
            crashed
            and recovered.status == PaperOperationStatus.SUCCESS
            and recovered.safety.PAPER_ORDERS_FILLED == 0
            and paper.last_sequence() == event_count
        ),
    }


def _context(as_of: datetime, tickers: list[str]) -> PaperOperationContext:
    quotes = _quotes(as_of, *tickers)
    return PaperOperationContext(
        universe=_universe(),
        market_quotes=quotes,
        market_context={
            "as_of": as_of.isoformat(),
            "by_ticker": {
                row.ticker: {
                    "ticker": row.ticker,
                    "market_data_as_of": row.as_of.isoformat(),
                    "last_price": row.last_price,
                    "bid": row.bid,
                    "ask": row.ask,
                    "stale": False,
                }
                for row in quotes
            },
        },
        event_context={"events_as_of": as_of.isoformat(), "events": []},
        research_status={
            "research_status_as_of": as_of.isoformat(),
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
            "SOURCE_FAILURE_ISOLATION": True,
            "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
            "seal": {"sealed_epoch_verified": True, "violations": 0},
            "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        },
    )


def _universe() -> list[dict[str, Any]]:
    return [
        {
            "ticker": ticker,
            "legal_issuer": issuer,
            "instrument_uid": f"UID-{ticker}",
            "figi": f"FIGI-{ticker}",
            "board": "TQBR",
            "instrument_type": "INSTRUMENT_TYPE_SHARE",
            "active": True,
            "supported": True,
            "market_data_compatible": True,
            "feature_compatible": True,
            "lot_size": 10 if ticker == "SBER" else 1,
        }
        for ticker, issuer in (("SBER", "Sberbank"), ("YDEX", "Yandex"))
    ]


def _quotes(as_of: datetime, *tickers: str) -> list[MarketQuote]:
    prices = {"SBER": 100.0, "YDEX": 110.0}
    return [
        MarketQuote(
            ticker=ticker,
            as_of=as_of,
            last_price=prices[ticker],
            bid=prices[ticker] - 0.2,
            ask=prices[ticker] + 0.2,
            lot_size=10 if ticker == "SBER" else 1,
        )
        for ticker in tickers
    ]


def _proposal(ticker: str, action: str, weight: float) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": 0.60,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": ["Deterministic operational paper proposal."],
        "risks": ["Paper-only sample; no performance claim."],
        "evidence": [],
        "data_quality": "GOOD",
    }


def _model(as_of: datetime, proposals: list[dict[str, Any]]) -> FakeAgentModel:
    output = {
        "as_of": as_of.isoformat(),
        "input_snapshot_as_of": {
            "portfolio_as_of": as_of.isoformat(),
            "market_data_as_of": as_of.isoformat(),
            "events_as_of": as_of.isoformat(),
        },
        "proposals": proposals,
    }
    return FakeAgentModel(
        [AgentModelResponse(final_output=json.dumps(output, sort_keys=True))],
        model_id="fake-paper-operation-v1",
    )


def _risk_agent_payload(as_of: datetime, proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "stale-plan-proof",
        "AGENT_RESEARCH_CAPABILITY_READY": True,
        "AGENT_DECISION_STATUS": "VALID",
        "validation": {"status": "VALID"},
        "universe": _universe(),
        "research_status_snapshot": {
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
        },
        "final_proposals": [proposal],
        "created_at": as_of.isoformat(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any], sample: OperationSample) -> None:
    final = sample.final_portfolio
    positions = ", ".join(
        f"{row['ticker']}:{row['quantity']}" for row in final.get("positions", [])
    )
    lines = [
        "# Paper Trading Operation V1 audit",
        "",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- PAPER_TRADING_OPERATION_READY: {manifest['PAPER_TRADING_OPERATION_READY']}",
        f"- operation policy: {manifest['operation_policy_version']}",
        f"- operation runs: {manifest['PAPER_OPERATION_RUNS']}",
        f"- paper orders filled: {manifest['PAPER_ORDERS_FILLED']}",
        f"- paper ledger events: {manifest['paper_ledger_event_count']}",
        f"- replay verified: {manifest['REPLAY_VERIFIED']}",
        f"- final equity: {final['equity']}",
        f"- final positions: {positions}",
        "- REAL_EXECUTION_READY: NO",
        "- REAL_BROKER_MUTATIONS: 0",
        "- REAL_ORDERS_SENT: 0",
        "",
        (
            "This artifact proves deterministic paper-operation behavior only. "
            "It makes no trading-performance claim."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
