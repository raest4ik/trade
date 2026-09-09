from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src.ai_trading_agent_v1.application import AgentRunConfig, UnconfiguredAgentModel, git_sha
from src.paper_trading_operation_v1.application import (
    DEFAULT_PAPER_STATE_ROOT,
    DEFAULT_STATE_ROOT,
    ExistingArtifactContextProvider,
    operation_history,
    preflight_failures,
    run_paper_operation,
)
from src.paper_trading_operation_v1.domain import (
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.repository import (
    JsonlOperationAuditRepository,
    OperationAlreadyRunningError,
)
from src.risk_engine_paper_v1.application import (
    initial_paper_portfolio,
    portfolio_state_sha,
    replay_portfolio,
)
from src.risk_engine_paper_v1.repository import JsonlPaperLedgerRepository


def run(args: argparse.Namespace) -> int:
    state_root = Path(args.state_root)
    paper_root = Path(args.paper_state_root)
    audit = JsonlOperationAuditRepository(state_root / "operation-ledger.jsonl")
    paper = JsonlPaperLedgerRepository(paper_root / "ledger.jsonl")
    if args.command == "history":
        print(json.dumps(operation_history(audit), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "inspect":
        completed = audit.completed_run(args.operation_id)
        if completed is None:
            raise SystemExit(f"operation not found or incomplete: {args.operation_id}")
        print(completed.model_dump_json(indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(_status(audit, paper), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "health":
        payload = _health(state_root, paper)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if payload["status"] == "READY" else 2

    as_of = _datetime(args.as_of) if args.as_of else datetime.now(UTC)
    policy = _policy_from_env()
    model = UnconfiguredAgentModel()
    provider = ExistingArtifactContextProvider(
        AgentRunConfig(
            output_root=state_root / "agent-runs" / "pending",
            code_sha=git_sha(),
            created_at=as_of,
            max_agent_steps=policy.max_agent_steps,
            max_tool_calls=policy.max_tool_calls,
            max_tickers_per_run=policy.max_operation_universe,
        )
    )
    mode = PaperOperationMode.PAPER_EXECUTE if args.execute_paper else PaperOperationMode.DRY_RUN
    try:
        result = run_paper_operation(
            operation_as_of=as_of,
            mode=mode,
            model=model,
            context_provider=provider,
            paper_repository=paper,
            audit_repository=audit,
            state_root=state_root,
            code_sha=git_sha(),
            policy=policy,
            operation_slot=args.operation_slot,
        )
    except OperationAlreadyRunningError:
        print(json.dumps({"status": "BLOCKED", "status_code": "OPERATION_ALREADY_RUNNING"}))
        return 2
    print(result.model_dump_json(indent=2))
    return _exit_code(result.status)


def _status(
    audit: JsonlOperationAuditRepository,
    paper: JsonlPaperLedgerRepository,
) -> dict[str, object]:
    history = operation_history(audit)
    events = paper.events()
    portfolio = replay_portfolio(events) if events else initial_paper_portfolio(datetime.now(UTC))
    last = history[-1] if history else {}
    return {
        "PAPER_TRADING_OPERATION_READY": True,
        "last_operation_id": last.get("operation_id"),
        "last_operation_status": last.get("status"),
        "last_operation_as_of": last.get("as_of"),
        "paper_portfolio_sha": portfolio_state_sha(portfolio),
        "ledger_event_count": len(events),
        "agent_capability": "READ_ONLY_RESEARCH",
        "risk_capability": "RISK_ENGINE_V1",
        "paper_capability": "MULTI_RUN_PAPER_PORTFOLIO_V1",
        "REAL_EXECUTION_READY": "NO",
        "REAL_BROKER_MUTATIONS": 0,
        "REAL_ORDERS_SENT": 0,
    }


def _health(state_root: Path, paper: JsonlPaperLedgerRepository) -> dict[str, object]:
    as_of = datetime.now(UTC)
    policy = _policy_from_env()
    provider = ExistingArtifactContextProvider(
        AgentRunConfig(output_root=state_root / "health", code_sha=git_sha(), created_at=as_of)
    )
    try:
        events = paper.events()
        portfolio = replay_portfolio(events) if events else initial_paper_portfolio(as_of)
        context = provider.load(operation_as_of=as_of, portfolio=portfolio, policy=policy)
        reasons = preflight_failures(
            operation_as_of=as_of,
            portfolio=portfolio,
            context=context,
            policy=policy,
        )
    except Exception as exc:
        reasons = [f"CONTEXT_UNAVAILABLE:{type(exc).__name__}"]
    reasons.append("AGENT_MODEL_UNAVAILABLE")
    return {
        "status": "READY" if not reasons else "BLOCKED",
        "reasons": reasons,
        "REAL_EXECUTION_READY": "NO",
    }


def _exit_code(status: PaperOperationStatus) -> int:
    if status in {
        PaperOperationStatus.SUCCESS,
        PaperOperationStatus.NO_ACTION,
        PaperOperationStatus.ALREADY_PROCESSED,
    }:
        return 0
    if status == PaperOperationStatus.BLOCKED:
        return 2
    if status == PaperOperationStatus.DEGRADED:
        return 3
    return 4


def _policy_from_env() -> PaperOperationPolicy:
    return PaperOperationPolicy(
        max_operation_universe=int(os.getenv("MAX_OPERATION_UNIVERSE", "10")),
        operation_schedule_enabled=_boolean("PAPER_OPERATION_SCHEDULE_ENABLED", False),
        operation_timezone=os.getenv("PAPER_OPERATION_TIMEZONE", "Europe/Moscow"),
        operation_session=os.getenv("PAPER_OPERATION_SESSION", "EOD"),
        paper_execution_enabled=_boolean("PAPER_EXECUTION_ENABLED", True),
        real_execution_enabled=_boolean("REAL_EXECUTION_ENABLED", False),
    )


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="paper-trading-operation-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    operation = subparsers.add_parser("run")
    operation.add_argument("--execute-paper", action="store_true")
    operation.add_argument("--operation-slot", default=None)
    operation.add_argument("--as-of", default=None)
    operation.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    operation.add_argument("--paper-state-root", default=str(DEFAULT_PAPER_STATE_ROOT))
    for name in ("status", "history", "health"):
        command = subparsers.add_parser(name)
        command.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
        command.add_argument("--paper-state-root", default=str(DEFAULT_PAPER_STATE_ROOT))
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("operation_id")
    inspect.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    inspect.add_argument("--paper-state-root", default=str(DEFAULT_PAPER_STATE_ROOT))
    return parser


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("as-of must include timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
