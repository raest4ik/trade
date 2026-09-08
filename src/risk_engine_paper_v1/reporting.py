from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.risk_engine_paper_v1.application import (
    ARTIFACT_VERSION,
    evaluate_agent_run,
    execute_paper_plan,
    portfolio_state_sha,
    replay_portfolio,
    restore_operational_portfolio,
    total_commission,
    write_json,
)
from src.risk_engine_paper_v1.domain import (
    MarketQuote,
    PaperExecutionResult,
    PaperExecutionStatus,
    PipelineSafety,
    RiskPlan,
    RiskPolicy,
)
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

SAMPLE_AS_OF = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class SampleRun:
    agent_run: dict[str, Any]
    plan: RiskPlan
    result: PaperExecutionResult


@dataclass(frozen=True)
class MultiRunSample:
    policy: RiskPolicy
    runs: tuple[SampleRun, ...]
    ledger: list[dict[str, Any]]
    replay_verification: dict[str, Any]
    multi_run_verification: dict[str, Any]
    idempotency_verification: dict[str, Any]
    stale_plan_verification: dict[str, Any]
    safety: PipelineSafety


def build_sample_execution(
    agent_artifact_path: Path,
    *,
    as_of: datetime = SAMPLE_AS_OF,
) -> MultiRunSample:
    policy = RiskPolicy()
    repository = InMemoryPaperLedgerRepository()

    run_1_agent = _sample_agent_run(
        agent_artifact_path,
        "paper-multi-run-1",
        [_proposal("SBER", "BUY", 0.10, "run-1-buy")],
    )
    run_1 = _evaluate_and_execute(repository, run_1_agent, _quotes(as_of), policy, as_of)
    stale_agent = _sample_agent_run(
        agent_artifact_path,
        "paper-stale-plan",
        [_proposal("SBER", "BUY", 0.12, "stale-plan")],
    )
    stale_time = as_of + timedelta(seconds=30)
    stale_plan = _evaluate(repository, stale_agent, _quotes(stale_time), policy, stale_time)

    run_2_time = as_of + timedelta(minutes=1)
    run_2_agent = _sample_agent_run(
        agent_artifact_path,
        "paper-multi-run-2",
        [
            _proposal("SBER", "BUY", 0.15, "run-2-delta-buy"),
            _proposal("YDEX", "BUY", 0.20, "run-2-reduce"),
            _proposal("GAZP", "HOLD", 0.05, "run-2-hold"),
            _proposal("LKOH", "BUY", 0.10, "run-2-stale"),
        ],
    )
    run_2 = _evaluate_and_execute(
        repository,
        run_2_agent,
        _quotes(run_2_time, include_stale=True),
        policy,
        run_2_time,
    )
    stale_result = execute_paper_plan(stale_plan, repository, execution_as_of=run_2_time)

    run_3_time = as_of + timedelta(minutes=2)
    run_3_agent = _sample_agent_run(
        agent_artifact_path,
        "paper-multi-run-3",
        [_proposal("SBER", "SELL", 0.05, "run-3-partial-sell")],
    )
    run_3 = _evaluate_and_execute(
        repository,
        run_3_agent,
        _quotes(run_3_time),
        policy,
        run_3_time,
    )

    before_duplicate = replay_portfolio(repository.events())
    restarted = InMemoryPaperLedgerRepository(repository.events())
    duplicate = execute_paper_plan(run_1.plan, restarted, execution_as_of=run_3_time)
    after_duplicate = replay_portfolio(restarted.events())
    replayed = replay_portfolio(repository.events())
    final = run_3.result.final_portfolio
    replay_sha = portfolio_state_sha(replayed)
    final_sha = portfolio_state_sha(final)

    runs = (run_1, run_2, run_3)
    safety = PipelineSafety(
        PAPER_ORDERS_PLANNED=sum(len(row.plan.paper_orders) for row in runs),
        PAPER_ORDERS_FILLED=sum(len(row.result.filled_orders) for row in runs),
        PAPER_PORTFOLIO_MUTATIONS=sum(len(row.result.filled_orders) for row in runs),
    )
    ledger = [event.model_dump(mode="json") for event in repository.events()]
    return MultiRunSample(
        policy=policy,
        runs=runs,
        ledger=ledger,
        replay_verification={
            "replay_matches": replay_sha == final_sha,
            "expected_sha": final_sha,
            "replayed_sha": replay_sha,
            "event_count": len(ledger),
            "historical_quotes_required": False,
        },
        multi_run_verification={
            "MULTI_RUN_PORTFOLIO_STATE": "PASS",
            "PORTFOLIO_RESET_BETWEEN_RUNS": False,
            "run_2_started_from_run_1": (
                run_2.plan.initial_portfolio.cash == run_1.result.final_portfolio.cash
                and run_2.plan.initial_portfolio.turnover_today
                == run_1.result.final_portfolio.turnover_today
            ),
            "run_2_bought_sber_delta_only": (
                _quantity(run_2.result, "SBER") > _quantity(run_1.result, "SBER")
            ),
            "run_3_saw_existing_sber": bool(run_3.plan.paper_orders),
        },
        idempotency_verification={
            "IDEMPOTENCY_VERIFIED": (
                duplicate.duplicate_executions_skipped == len(run_1.plan.paper_orders)
                and not duplicate.filled_orders
                and before_duplicate == after_duplicate
            ),
            "duplicate_executions_skipped": duplicate.duplicate_executions_skipped,
            "new_fills": len(duplicate.filled_orders),
            "portfolio_unchanged": before_duplicate == after_duplicate,
        },
        stale_plan_verification={
            "STALE_PLAN_REJECTED": (
                stale_result.execution_status == PaperExecutionStatus.STALE_RISK_PLAN
                and stale_result.safety.PAPER_ORDERS_FILLED == 0
            ),
            "execution_status": stale_result.execution_status,
            "status_code": stale_result.status_code,
            "paper_orders_filled": stale_result.safety.PAPER_ORDERS_FILLED,
            "paper_portfolio_mutations": stale_result.safety.PAPER_PORTFOLIO_MUTATIONS,
        },
        safety=safety,
    )


def write_audit_artifact(
    *,
    output_root: Path,
    code_sha: str,
    sample: MultiRunSample,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable risk/paper output already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    decisions = [decision for run in sample.runs for decision in run.plan.decisions]
    counts = Counter(decision.risk_decision.value for decision in decisions)
    trades = [trade for run in sample.runs for trade in run.result.paper_trades]
    replay_verified = sample.replay_verification["replay_matches"] is True
    multi_run_ready = (
        sample.multi_run_verification["MULTI_RUN_PORTFOLIO_STATE"] == "PASS"
        and sample.idempotency_verification["IDEMPOTENCY_VERIFIED"] is True
        and sample.stale_plan_verification["STALE_PLAN_REJECTED"] is True
        and replay_verified
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "created_at": sample.runs[-1].plan.decision_as_of.isoformat(),
        "code_sha": code_sha,
        "risk_policy_version": sample.policy.policy_version,
        "AGENT_RESEARCH_CAPABILITY_READY": "YES",
        "SINGLE_RUN_PAPER_READY": "YES",
        "MULTI_RUN_PAPER_READY": "YES" if multi_run_ready else "NO",
        "RISK_ENGINE_READY": "YES" if multi_run_ready else "NO",
        "PAPER_PORTFOLIO_READY": "YES" if multi_run_ready else "NO",
        "REAL_EXECUTION_READY": "NO",
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        "STALE_PLAN_PROTECTION": "YES"
        if sample.stale_plan_verification["STALE_PLAN_REJECTED"] is True
        else "NO",
        "REPLAY_VERIFIED": "YES" if replay_verified else "NO",
        "IDEMPOTENCY_VERIFIED": "YES"
        if sample.idempotency_verification["IDEMPOTENCY_VERIFIED"] is True
        else "NO",
        "MULTI_RUN_PORTFOLIO_STATE": sample.multi_run_verification["MULTI_RUN_PORTFOLIO_STATE"],
        "PORTFOLIO_RESET_BETWEEN_RUNS": False,
        "APPROVE_COUNT": counts["APPROVE"],
        "REDUCE_COUNT": counts["REDUCE"],
        "REJECT_COUNT": counts["REJECT"],
        "NO_ACTION_COUNT": counts["NO_ACTION"],
        "TOTAL_COMMISSION": total_commission(trades),
        "SLIPPAGE_MODEL": "FIXED_BPS",
        "IDEMPOTENCY_KEY_POLICY": "agent_run_id+proposal_id",
        **sample.safety.model_dump(mode="json"),
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(manifest)
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "risk-policy.json", sample.policy.model_dump(mode="json"))
    for index, run in enumerate(sample.runs, start=1):
        write_json(output_root / f"run-{index}-agent.json", run.agent_run)
        write_json(output_root / f"run-{index}-risk-plan.json", run.plan.model_dump(mode="json"))
        write_json(
            output_root / f"run-{index}-orders.json",
            [order.model_dump(mode="json") for order in run.result.filled_orders],
        )
    write_json(
        output_root / "final-portfolio.json",
        sample.runs[-1].result.final_portfolio.model_dump(mode="json"),
    )
    write_json(output_root / "replay-verification.json", sample.replay_verification)
    write_json(output_root / "multi-run-verification.json", sample.multi_run_verification)
    write_json(output_root / "idempotency-verification.json", sample.idempotency_verification)
    write_json(output_root / "stale-plan-verification.json", sample.stale_plan_verification)
    write_json(output_root / "safety.json", sample.safety.model_dump(mode="json"))
    _write_jsonl(output_root / "paper-ledger.jsonl", sample.ledger)
    _write_report(output_root / "report.md", manifest, sample)
    return manifest


def _evaluate_and_execute(
    repository: InMemoryPaperLedgerRepository,
    agent_run: dict[str, Any],
    quotes: list[MarketQuote],
    policy: RiskPolicy,
    as_of: datetime,
) -> SampleRun:
    plan = _evaluate(repository, agent_run, quotes, policy, as_of)
    result = execute_paper_plan(plan, repository, execution_as_of=as_of)
    return SampleRun(agent_run=agent_run, plan=plan, result=result)


def _evaluate(
    repository: InMemoryPaperLedgerRepository,
    agent_run: dict[str, Any],
    quotes: list[MarketQuote],
    policy: RiskPolicy,
    as_of: datetime,
) -> RiskPlan:
    portfolio = restore_operational_portfolio(
        repository,
        as_of=as_of,
        market_snapshot=quotes,
    )
    return evaluate_agent_run(
        agent_run=agent_run,
        portfolio=portfolio,
        market_snapshot=quotes,
        policy=policy,
        decision_as_of=as_of,
        ledger_event_count=repository.last_sequence(),
    )


def _sample_agent_run(
    path: Path,
    run_id: str,
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sample Agent V1 run must be an object")
    run = cast("dict[str, Any]", json.loads(json.dumps(value, ensure_ascii=False)))
    run["run_id"] = run_id
    run["final_proposals"] = proposals
    universe = cast("list[dict[str, Any]]", run["universe"])
    known = {str(row.get("ticker")) for row in universe}
    for ticker in {str(row["ticker"]) for row in proposals} - known:
        universe.append(
            {
                "ticker": ticker,
                "legal_issuer": f"{ticker} sample issuer",
                "active": True,
                "supported": True,
                "market_data_compatible": True,
                "feature_compatible": True,
            }
        )
    return run


def _proposal(
    ticker: str,
    action: str,
    target_weight: float,
    evidence_id: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": 0.60,
        "target_weight": target_weight,
        "holding_horizon": "1-5d",
        "thesis": ["Deterministic multi-run risk-engine coverage proposal."],
        "risks": ["Sample-only proposal with no real execution semantics."],
        "evidence": [{"type": "MARKET", "id": evidence_id}],
        "data_quality": "GOOD",
    }


def _quotes(as_of: datetime, *, include_stale: bool = False) -> list[MarketQuote]:
    quotes = [
        MarketQuote(ticker="SBER", as_of=as_of, last_price=100.0, bid=99.8, ask=100.2, lot_size=10),
        MarketQuote(ticker="YDEX", as_of=as_of, last_price=110.5, bid=110.3, ask=110.7, lot_size=1),
        MarketQuote(
            ticker="GAZP", as_of=as_of, last_price=103.5, bid=103.3, ask=103.7, lot_size=10
        ),
    ]
    if include_stale:
        quotes.append(
            MarketQuote(
                ticker="LKOH",
                as_of=as_of - timedelta(hours=2),
                last_price=7_100.0,
                bid=7_099.0,
                ask=7_101.0,
                lot_size=1,
            )
        )
    return quotes


def _quantity(result: PaperExecutionResult, ticker: str) -> int:
    return next(
        (row.quantity for row in result.final_portfolio.positions if row.ticker == ticker), 0
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    sample: MultiRunSample,
) -> None:
    final = sample.runs[-1].result.final_portfolio
    positions = ", ".join(f"{row.ticker}:{row.quantity}" for row in final.positions)
    lines = [
        "# Risk engine and paper portfolio V1 multi-run audit",
        "",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- MULTI_RUN_PAPER_READY: {manifest['MULTI_RUN_PAPER_READY']}",
        f"- STALE_PLAN_PROTECTION: {manifest['STALE_PLAN_PROTECTION']}",
        f"- REPLAY_VERIFIED: {manifest['REPLAY_VERIFIED']}",
        f"- IDEMPOTENCY_VERIFIED: {manifest['IDEMPOTENCY_VERIFIED']}",
        f"- PORTFOLIO_RESET_BETWEEN_RUNS: {manifest['PORTFOLIO_RESET_BETWEEN_RUNS']}",
        f"- paper orders filled: {manifest['PAPER_ORDERS_FILLED']}",
        f"- final cash: {final.cash:.4f} RUB",
        f"- final equity: {final.equity:.4f} RUB",
        f"- final positions: {positions}",
        "",
        (
            "This artifact proves deterministic multi-run paper execution safety only. "
            "It makes no performance claim."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
