from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.risk_engine_paper_v1.application import (
    ARTIFACT_VERSION,
    evaluate_agent_run,
    execute_paper_plan,
    initial_paper_portfolio,
    total_commission,
    write_json,
)
from src.risk_engine_paper_v1.domain import (
    MarketQuote,
    PaperExecutionResult,
    RiskPolicy,
)
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

SAMPLE_AS_OF = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def build_sample_execution(
    agent_artifact_path: Path,
    *,
    as_of: datetime = SAMPLE_AS_OF,
) -> tuple[dict[str, Any], RiskPolicy, PaperExecutionResult, list[dict[str, Any]]]:
    agent_run = _sample_agent_run(agent_artifact_path)
    policy = RiskPolicy()
    portfolio = initial_paper_portfolio(as_of)
    quotes = [
        MarketQuote(ticker="SBER", as_of=as_of, last_price=100.0, bid=99.8, ask=100.2, lot_size=10),
        MarketQuote(
            ticker="GAZP", as_of=as_of, last_price=103.5, bid=103.3, ask=103.7, lot_size=10
        ),
        MarketQuote(ticker="ROSN", as_of=as_of, last_price=107.0, bid=106.8, ask=107.2, lot_size=1),
        MarketQuote(ticker="YDEX", as_of=as_of, last_price=110.5, bid=110.3, ask=110.7, lot_size=1),
        MarketQuote(
            ticker="LKOH",
            as_of=as_of - timedelta(hours=2),
            last_price=7_100.0,
            bid=7_099.0,
            ask=7_101.0,
            lot_size=1,
        ),
    ]
    plan = evaluate_agent_run(
        agent_run=agent_run,
        portfolio=portfolio,
        market_snapshot=quotes,
        policy=policy,
        decision_as_of=as_of,
    )
    repository = InMemoryPaperLedgerRepository()
    result = execute_paper_plan(plan, repository)
    ledger = [event.model_dump(mode="json") for event in repository.events()]
    return agent_run, policy, result, ledger


def write_audit_artifact(
    *,
    output_root: Path,
    code_sha: str,
    agent_run: dict[str, Any],
    policy: RiskPolicy,
    result: PaperExecutionResult,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable risk/paper output already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    counts = Counter(decision.risk_decision.value for decision in result.plan.decisions)
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "created_at": result.plan.decision_as_of.isoformat(),
        "code_sha": code_sha,
        "agent_run_id": result.plan.agent_run_id,
        "agent_run_sha": result.plan.agent_run_sha,
        "risk_policy_version": policy.policy_version,
        "AGENT_RESEARCH_CAPABILITY_READY": "YES",
        "RISK_ENGINE_READY": "YES",
        "PAPER_PORTFOLIO_READY": "YES" if result.replay_verification.replay_matches else "NO",
        "REAL_EXECUTION_READY": "NO",
        "ML_V2_DATASET_STATUS": agent_run.get("ML_V2_DATASET_STATUS"),
        "APPROVE_COUNT": counts["APPROVE"],
        "REDUCE_COUNT": counts["REDUCE"],
        "REJECT_COUNT": counts["REJECT"],
        "NO_ACTION_COUNT": counts["NO_ACTION"],
        "TOTAL_COMMISSION": total_commission(result.paper_trades),
        "SLIPPAGE_MODEL": "FIXED_BPS",
        "REPLAY_VERIFIED": result.replay_verification.replay_matches,
        "IDEMPOTENCY_KEY_POLICY": "agent_run_id+proposal_id",
        **result.safety.model_dump(mode="json"),
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(manifest)
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "risk-policy.json", policy.model_dump(mode="json"))
    write_json(
        output_root / "initial-portfolio.json",
        result.plan.initial_portfolio.model_dump(mode="json"),
    )
    write_json(output_root / "agent-run-input.json", agent_run)
    write_json(
        output_root / "risk-decisions.json",
        [row.model_dump(mode="json") for row in result.plan.decisions],
    )
    write_json(
        output_root / "paper-orders.json",
        [row.model_dump(mode="json") for row in result.filled_orders],
    )
    write_json(
        output_root / "paper-trades.json",
        [row.model_dump(mode="json") for row in result.paper_trades],
    )
    write_json(
        output_root / "final-portfolio.json",
        result.final_portfolio.model_dump(mode="json"),
    )
    write_json(
        output_root / "replay-verification.json",
        result.replay_verification.model_dump(mode="json"),
    )
    write_json(output_root / "safety.json", result.safety.model_dump(mode="json"))
    _write_jsonl(output_root / "paper-ledger.jsonl", ledger)
    _write_report(output_root / "report.md", manifest, result)
    return manifest


def _sample_agent_run(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sample Agent V1 run must be an object")
    run = cast("dict[str, Any]", value)
    run = json.loads(json.dumps(run, ensure_ascii=False))
    run["run_id"] = "sample-risk-agent-run-v1"
    universe = cast("list[dict[str, Any]]", run["universe"])
    universe.append(
        {
            "ticker": "LKOH",
            "legal_issuer": "PJSC LUKOIL",
            "instrument_uid": "UID-LKOH-SAMPLE",
            "figi": "FIGI-LKOH-SAMPLE",
            "board": "TQBR",
            "instrument_type": "INSTRUMENT_TYPE_SHARE",
            "active": True,
            "supported": True,
            "market_data_compatible": True,
            "feature_compatible": True,
        }
    )
    proposals = cast("list[dict[str, Any]]", run["final_proposals"])
    proposals.extend(
        [
            _proposal("YDEX", 0.20, "sample-position-limit"),
            _proposal("LKOH", 0.10, "sample-stale-quote"),
        ]
    )
    return run


def _proposal(ticker: str, target_weight: float, evidence_id: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": "BUY",
        "agent_confidence": 0.60,
        "target_weight": target_weight,
        "holding_horizon": "1-5d",
        "thesis": ["Deterministic risk-engine coverage proposal."],
        "risks": ["Sample-only proposal with no real execution semantics."],
        "evidence": [{"type": "MARKET", "id": evidence_id}],
        "data_quality": "GOOD",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any], result: PaperExecutionResult) -> None:
    positions = ", ".join(
        f"{row.ticker}:{row.quantity}" for row in result.final_portfolio.positions
    )
    lines = [
        "# Risk engine and paper portfolio V1",
        "",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- RISK_ENGINE_READY: {manifest['RISK_ENGINE_READY']}",
        f"- PAPER_PORTFOLIO_READY: {manifest['PAPER_PORTFOLIO_READY']}",
        f"- REAL_EXECUTION_READY: {manifest['REAL_EXECUTION_READY']}",
        f"- APPROVE / REDUCE / REJECT / NO_ACTION: "
        f"{manifest['APPROVE_COUNT']} / {manifest['REDUCE_COUNT']} / "
        f"{manifest['REJECT_COUNT']} / {manifest['NO_ACTION_COUNT']}",
        f"- paper orders filled: {manifest['PAPER_ORDERS_FILLED']}",
        f"- final cash: {result.final_portfolio.cash:.4f} RUB",
        f"- final equity: {result.final_portfolio.equity:.4f} RUB",
        f"- final positions: {positions}",
        f"- replay verified: {manifest['REPLAY_VERIFIED']}",
        "",
        (
            "This artifact proves deterministic paper execution safety only. "
            "It makes no performance claim."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
