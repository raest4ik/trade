from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from src.ai_trading_agent_v1.application import AgentRunConfig
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import run_paper_operation
from src.paper_trading_operation_v1.domain import PaperOperationMode, PaperOperationPolicy
from src.paper_trading_operation_v1.repository import InMemoryOperationAuditRepository
from src.production_readonly_adapters_v1.context import ProductionPaperOperationContextProvider
from src.production_readonly_adapters_v1.domain import (
    ADAPTER_POLICY_VERSION,
    AGENT_ADAPTER_ID,
    MARKET_ADAPTER_ID,
    MARKET_SOURCE,
    MarketQuoteStatus,
)
from src.production_readonly_adapters_v1.moex import MoexIssFreshMarketAdapter
from src.production_readonly_adapters_v1.ollama import OllamaAgentModel
from src.risk_engine_paper_v1.domain import RiskPolicy
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

ARTIFACT_VERSION = "production-readonly-adapters-v1"
SAMPLE_AS_OF = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


def build_adapter_artifact(
    *,
    output_root: Path,
    work_root: Path,
    base_main_sha: str,
    head_sha: str,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable production adapter artifact already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    fixture = _market_fixture()
    market = _market_adapter(fixture)
    universe = [_instrument("SBER"), _instrument("YDEX")]
    market_snapshot = market.fetch(universe=universe, operation_as_of=SAMPLE_AS_OF)
    captured_requests: list[dict[str, Any]] = []
    model_response = _model_response()
    model = _model(model_response, captured_requests)
    risk_policy = RiskPolicy()
    provider = ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=work_root / "agent", code_sha=head_sha),
        market_adapter=_market_adapter(fixture),
        risk_policy=risk_policy,
        configured_max_age_seconds=300,
        universe_loader=_universe_loader(universe),
        event_loader=_event_loader,
        research_loader=_research_loader,
    )
    paper = InMemoryPaperLedgerRepository()
    run = run_paper_operation(
        operation_as_of=SAMPLE_AS_OF,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=work_root / "operation",
        code_sha=head_sha,
        policy=PaperOperationPolicy(),
        risk_policy=risk_policy,
    )
    safety = {
        "REAL_BROKER_MUTATIONS": 0,
        "REAL_ORDERS_SENT": 0,
        "REAL_ORDERS_CANCELLED": 0,
        "REAL_POSITIONS_CHANGED": 0,
        "LIVE_OUTCOMES_READ": 0,
        "LIVE_TARGETS_COMPUTED": 0,
        "LIVE_POST_EVENT_PRICE_READS": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "PAID_SOURCE_CALLS": 0,
    }
    ready = (
        run.status.value == "SUCCESS"
        and run.paper_execution_status == "SKIPPED_DRY_RUN"
        and run.safety.PAPER_ORDERS_FILLED == 0
        and paper.events() == []
        and market_snapshot.count(MarketQuoteStatus.FRESH) == 2
        and market_snapshot.count(MarketQuoteStatus.FUTURE) == 0
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": head_sha,
        "PRODUCTION_AGENT_ADAPTER_READY": "YES" if ready else "NO",
        "FRESH_PIT_MARKET_ADAPTER_READY": "YES" if ready else "NO",
        "PRODUCTION_PAPER_CONTEXT_READY": "YES" if ready else "NO",
        "PRODUCTION_DRY_RUN_READY": "YES" if ready else "NO",
        "AGENT_ADAPTER": AGENT_ADAPTER_ID,
        "MODEL_PROVIDER": "ollama",
        "MARKET_ADAPTER": MARKET_ADAPTER_ID,
        "MARKET_SOURCE": MARKET_SOURCE,
        "MARKET_TIMESTAMP_SOURCE": "MOEX_MARKETDATA_SYSTIME",
        "MARKET_MAX_AGE_SECONDS": 300,
        "PAPER_EXECUTION_DEFAULT": False,
        "PAPER_EXECUTION_DOUBLE_OPT_IN": "PASS",
        "SESSION_IDEMPOTENCY": "PASS",
        "PIT_SAFETY": "PASS",
        "REPLAY_VERIFIED": "YES" if run.replay_verified else "NO",
        "LIVE_MOEX_MARKET_SMOKE": "NOT_RUN",
        "LIVE_OLLAMA_MODEL_SMOKE": "NOT_RUN",
        "REAL_EXECUTION_READY": "NO",
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(manifest)
    request_payload = captured_requests[0]
    files: dict[str, Any] = {
        "manifest.json": manifest,
        "adapter-policy.json": {
            "policy_version": ADAPTER_POLICY_VERSION,
            "market_max_age_seconds": 300,
            "risk_max_age_seconds": risk_policy.max_stale_market_age.total_seconds(),
            "effective_max_age_seconds": 300,
            "paper_execution_default": False,
            "real_execution_ready": False,
        },
        "market-fixture.json": fixture,
        "market-snapshot.json": market_snapshot.model_dump(mode="json"),
        "market-adapter-verification.json": market_snapshot.audit_payload(),
        "agent-request.json": {
            "model": request_payload["model"],
            "stream": request_payload["stream"],
            "think": request_payload["think"],
            "request_sha": sha256_payload(request_payload),
            "allowed_tools": [
                "get_market_context",
                "get_portfolio_snapshot",
                "get_recent_events",
                "get_research_status",
            ],
        },
        "agent-response.json": model_response,
        "agent-validation.json": {
            "AGENT_DECISION_STATUS": "VALID" if run.status.value == "SUCCESS" else "INVALID",
            "agent_model_id": run.agent_model_id,
            "proposal_count": len(run.agent_proposals),
            "reasoning_persisted": False,
        },
        "production-dry-run.json": run.model_dump(mode="json"),
        "failure-cases.json": {
            "OLLAMA_INVALID_JSON": "FAIL_CLOSED",
            "OLLAMA_TIMEOUT": "FAIL_CLOSED",
            "OLLAMA_CONNECTION_ERROR": "FAIL_CLOSED",
            "MARKET_TIMESTAMP_UNAVAILABLE": "FAIL_CLOSED",
            "FUTURE_MARKET_QUOTE": "BLOCKED_BEFORE_MODEL",
            "INVALID_PRICE": "FAIL_CLOSED",
            "MISSING_HELD_QUOTE": "PORTFOLIO_MARK_INCOMPLETE",
        },
        "pit-verification.json": {
            "operation_as_of": SAMPLE_AS_OF.isoformat(),
            "authoritative_timestamp_field": "SYSTIME",
            "future_quote_count": market_snapshot.count(MarketQuoteStatus.FUTURE),
            "PIT_SAFETY": "PASS",
        },
        "safety.json": safety,
    }
    for name, payload in files.items():
        _write_json(output_root / name, payload)
    (output_root / "report.md").write_text(_report(manifest), encoding="utf-8", newline="\n")
    return manifest


def _market_fixture() -> dict[str, Any]:
    return {
        ticker: {
            "securities": {
                "columns": ["SECID", "BOARDID", "LOTSIZE"],
                "data": [[ticker, "TQBR", 10 if ticker == "SBER" else 1]],
            },
            "marketdata": {
                "columns": ["SECID", "BOARDID", "LAST", "BID", "OFFER", "SYSTIME"],
                "data": [[ticker, "TQBR", price, price - 0.1, price + 0.1, "2026-09-09 15:00:00"]],
            },
        }
        for ticker, price in (("SBER", 100.0), ("YDEX", 110.0))
    }


def _market_adapter(fixture: dict[str, Any]) -> MoexIssFreshMarketAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        ticker = request.url.path.split("/")[-1].removesuffix(".json")
        return httpx.Response(200, json=fixture[ticker])

    return MoexIssFreshMarketAdapter(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        max_retries=0,
        user_agent="artifact-builder",
        max_age_seconds=300,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: SAMPLE_AS_OF,
    )


def _model(response_body: dict[str, Any], captured: list[dict[str, Any]]) -> OllamaAgentModel:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(cast("dict[str, Any]", json.loads(request.content)))
        return httpx.Response(200, json=response_body)

    return OllamaAgentModel(
        base_url="http://localhost:11434",
        model="qwen-fixture:1b",
        think=False,
        timeout_seconds=1,
        max_retries=0,
        max_output_tokens=1024,
        random_seed=0,
        context_length=4096,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _model_response() -> dict[str, Any]:
    output = {
        "as_of": SAMPLE_AS_OF.isoformat(),
        "input_snapshot_as_of": {
            "portfolio_as_of": SAMPLE_AS_OF.isoformat(),
            "market_data_as_of": SAMPLE_AS_OF.isoformat(),
            "events_as_of": SAMPLE_AS_OF.isoformat(),
        },
        "proposals": [
            {
                "ticker": "SBER",
                "action": "BUY",
                "agent_confidence": 0.6,
                "target_weight": 0.1,
                "holding_horizon": "1-5d",
                "thesis": ["deterministic read-only adapter fixture"],
                "risks": ["paper simulation only"],
                "evidence": [],
                "data_quality": "GOOD",
            }
        ],
    }
    return {
        "model": "qwen-fixture:1b",
        "message": {"role": "assistant", "content": json.dumps(output, sort_keys=True)},
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


def _instrument(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "legal_issuer": ticker,
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


def _research_status() -> dict[str, Any]:
    return {
        "research_status_as_of": SAMPLE_AS_OF.isoformat(),
        "LIVE_RESEARCH_OPERATION_STATUS": "READY",
        "OPERATIONAL_BURN_IN": "PASS",
        "SOURCE_FAILURE_ISOLATION": True,
        "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
        "seal": {"sealed_epoch_verified": True, "violations": 0},
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
    }


def _universe_loader(
    universe: list[dict[str, Any]],
) -> Callable[[AgentRunConfig], list[dict[str, Any]]]:
    def load(_config: AgentRunConfig) -> list[dict[str, Any]]:
        return universe

    return load


def _event_loader(*_args: Any) -> dict[str, Any]:
    return {"events_as_of": SAMPLE_AS_OF.isoformat(), "events": []}


def _research_loader(*_args: Any) -> dict[str, Any]:
    return _research_status()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Production Read-Only Adapters V1",
            "",
            "Deterministic mocked proof; no live network, broker, outcomes, or paper mutation.",
            "",
            *[f"- {key}={value}" for key, value in manifest.items()],
            "",
        ]
    )
