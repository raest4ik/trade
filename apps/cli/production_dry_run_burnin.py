from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.ai_trading_agent_v1.application import AgentRunConfig, build_allowed_universe, git_sha
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import DEFAULT_PAPER_STATE_ROOT
from src.paper_trading_operation_v1.domain import PaperOperationPolicy
from src.paper_trading_operation_v1.repository import JsonlOperationAuditRepository
from src.production_dry_run_burnin_v1.application import (
    DEFAULT_BIN_STATE_ROOT,
    build_burnin_report,
    run_burnin_once,
)
from src.production_dry_run_burnin_v1.domain import BurninObservationStatus, BurninPolicy
from src.production_dry_run_burnin_v1.policy import MoexIssSessionVerifier
from src.production_dry_run_burnin_v1.repository import (
    BurninAlreadyRunningError,
    BurninLedgerIntegrityError,
    BurninSingleFlightLock,
    JsonlBurninObservationRepository,
    validate_observations,
)
from src.production_readonly_adapters_v1.factory import (
    create_current_tradability_resolver,
    create_production_agent_model,
    create_production_context_provider,
)
from src.risk_engine_paper_v1.domain import RiskPolicy
from src.risk_engine_paper_v1.repository import JsonlPaperLedgerRepository
from src.shared.config.settings import get_settings


def run(args: argparse.Namespace) -> int:
    if args.command == "universe-status":
        settings = get_settings()
        config = AgentRunConfig(output_root=Path("unused"), code_sha=git_sha())
        resolution = create_current_tradability_resolver(settings).resolve(
            build_allowed_universe(config)
        )
        selected = sorted(
            row.ticker for row in resolution.classifications if row.candidate_eligible
        )[: args.max_universe]
        print(
            json.dumps(
                {
                    **resolution.audit_payload(),
                    "selected_tickers": selected,
                    "selected_universe_sha": sha256_payload(selected),
                    "classifications": [
                        row.model_dump(mode="json") for row in resolution.classifications
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "calendar-status":
        settings = get_settings()
        requested = (
            date.fromisoformat(args.date)
            if args.date
            else datetime.now(ZoneInfo("Europe/Moscow")).date()
        )
        evidence = MoexIssSessionVerifier(
            base_url=settings.moex_iss_base_url,
            timeout_seconds=settings.moex_http_timeout_seconds,
            max_retries=settings.moex_http_max_retries,
            user_agent=settings.moex_http_user_agent,
        ).verify(requested)
        print(evidence.model_dump_json(indent=2))
        return 0
    if args.command == "calendar-range":
        settings = get_settings()
        first = date.fromisoformat(args.from_date)
        last = date.fromisoformat(args.to_date)
        if last < first or (last - first).days > 366:
            raise SystemExit("calendar range must be ordered and at most 367 days")
        verifier = MoexIssSessionVerifier(
            base_url=settings.moex_iss_base_url,
            timeout_seconds=settings.moex_http_timeout_seconds,
            max_retries=settings.moex_http_max_retries,
            user_agent=settings.moex_http_user_agent,
        )
        dates: list[date] = []
        current = first
        while current <= last:
            dates.append(current)
            current += timedelta(days=1)
        rows = [row.model_dump(mode="json") for row in verifier.verify_many(dates)]
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    state_root = Path(args.state_root)
    observations = JsonlBurninObservationRepository(state_root / "observations.jsonl")
    if args.command == "verify":
        try:
            rows = observations.observations()
            validate_observations(rows)
        except BurninLedgerIntegrityError as exc:
            print(json.dumps({"BURNIN_LEDGER_INTEGRITY": "FAIL", "reason": str(exc)}))
            return 4
        print(json.dumps({"BURNIN_LEDGER_INTEGRITY": "PASS", "records": len(rows)}))
        return 0
    if args.command == "inspect":
        row = observations.get(args.observation_id)
        if row is None:
            raise SystemExit(f"observation not found: {args.observation_id}")
        print(row.model_dump_json(indent=2))
        return 0
    if args.command == "history":
        payload = [
            {
                "burnin_observation_id": row.burnin_observation_id,
                "market_date": row.market_date,
                "operation_session": row.operation_session,
                "status": row.status.value,
                "operation_status_code": row.operation_status_code,
                "burnin_epoch": row.burnin_epoch,
            }
            for row in observations.observations()
        ]
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command in {"status", "report"}:
        rows = observations.observations()
        report = build_burnin_report(rows)
        current_rows = [row for row in rows if row.burnin_epoch == report.BURNIN_EPOCH]
        _write_status_cache(state_root, report.model_dump(mode="json"))
        payload = report.model_dump(mode="json")
        if args.command == "status":
            payload = {
                "BURNIN_STATUS": report.BURNIN_STATUS.value,
                "BURNIN_EPOCH": report.BURNIN_EPOCH,
                "valid_cycles": report.valid_cycles,
                "distinct_trading_days": report.distinct_trading_days,
                "first_observation_at": (
                    current_rows[0].cycle_started_at.isoformat() if current_rows else None
                ),
                "last_observation_at": (
                    current_rows[-1].completed_at.isoformat() if current_rows else None
                ),
                "safety_violations": sum(
                    row.status == BurninObservationStatus.SAFETY_VIOLATION for row in current_rows
                ),
                "current_blockers": [
                    row.operation_status_code
                    for row in current_rows
                    if row.status != BurninObservationStatus.PASS
                ],
                "last_observation_id": report.last_observation_id,
                "safety": report.safety_behavior,
                "readiness": report.PRODUCTION_DRY_RUN_BURNIN_READY,
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    try:
        with BurninSingleFlightLock(state_root / "burnin.lock"):
            exit_code = _run_once(args, state_root, observations)
            _write_status_cache(
                state_root,
                build_burnin_report(observations.observations()).model_dump(mode="json"),
            )
            return exit_code
    except BurninAlreadyRunningError:
        print(json.dumps({"status": "BURNIN_ALREADY_RUNNING"}))
        return 2


def _run_once(
    args: argparse.Namespace,
    state_root: Path,
    observations: JsonlBurninObservationRepository,
) -> int:
    cycle_started_at = _datetime(args.as_of) if args.as_of else datetime.now(UTC)
    settings = replace(get_settings(), ollama_think=False)
    burnin_policy = BurninPolicy()
    operation_policy = PaperOperationPolicy(
        operation_session=burnin_policy.primary_operation_slot,
        operation_schedule_enabled=False,
        paper_auto_execution_enabled=False,
        paper_execution_enabled=False,
        real_execution_enabled=False,
    )
    risk_policy = RiskPolicy()
    operation_root = state_root / "paper-operation"
    paper_root = Path(args.paper_state_root)
    agent_config = AgentRunConfig(
        output_root=operation_root / "agent-runs" / "pending",
        code_sha=git_sha(),
        created_at=cycle_started_at,
        max_agent_steps=operation_policy.max_agent_steps,
        max_tool_calls=operation_policy.max_tool_calls,
        max_tickers_per_run=operation_policy.max_operation_universe,
    )
    result = run_burnin_once(
        cycle_started_at=cycle_started_at,
        model=create_production_agent_model(settings),
        context_provider=create_production_context_provider(settings, agent_config, risk_policy),
        paper_repository=JsonlPaperLedgerRepository(paper_root / "ledger.jsonl"),
        audit_repository=JsonlOperationAuditRepository(operation_root / "operation-ledger.jsonl"),
        observation_repository=observations,
        session_verifier=MoexIssSessionVerifier(
            base_url=settings.moex_iss_base_url,
            timeout_seconds=settings.moex_http_timeout_seconds,
            max_retries=settings.moex_http_max_retries,
            user_agent=settings.moex_http_user_agent,
        ),
        operation_state_root=operation_root,
        code_sha=git_sha(),
        policy=burnin_policy,
        operation_policy=operation_policy,
        risk_policy=risk_policy,
        retry_index=args.retry,
        retry_reason=args.retry_reason,
    )
    print(result.model_dump_json(indent=2))
    if result.status in {BurninObservationStatus.PASS.value, "ALREADY_OBSERVED"}:
        return 0
    if result.status == BurninObservationStatus.BLOCKED_EXPECTED.value:
        return 2
    return 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production-dry-run-burnin-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_once = subparsers.add_parser("run", aliases=["run-once"])
    run_once.add_argument("--as-of", default=None)
    run_once.add_argument("--retry", type=int, default=0)
    run_once.add_argument("--retry-reason", default=None)
    for command in (run_once,):
        command.add_argument("--state-root", default=str(DEFAULT_BIN_STATE_ROOT))
        command.add_argument("--paper-state-root", default=str(DEFAULT_PAPER_STATE_ROOT))
    for name in ("status", "history", "report", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--state-root", default=str(DEFAULT_BIN_STATE_ROOT))
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("observation_id")
    inspect.add_argument("--state-root", default=str(DEFAULT_BIN_STATE_ROOT))
    calendar = subparsers.add_parser("calendar-status")
    calendar.add_argument("--date", default=None)
    calendar_range = subparsers.add_parser("calendar-range")
    calendar_range.add_argument("--from", dest="from_date", required=True)
    calendar_range.add_argument("--to", dest="to_date", required=True)
    universe = subparsers.add_parser("universe-status")
    universe.add_argument("--max-universe", type=int, default=10)
    return parser


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("as-of must include timezone")
    return parsed.astimezone(UTC)


def _write_status_cache(state_root: Path, payload: dict[str, object]) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    temporary = state_root / "status.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(state_root / "status.json")


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
