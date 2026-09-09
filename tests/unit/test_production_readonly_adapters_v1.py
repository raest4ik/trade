from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from apps.cli import paper_operation as paper_operation_cli
from src.ai_trading_agent_v1.application import (
    AgentDataContext,
    AgentRunConfig,
    run_read_only_research_agent_v1,
)
from src.ai_trading_agent_v1.domain import AgentDecisionStatus, AgentModelRequest
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import run_paper_operation
from src.paper_trading_operation_v1.domain import (
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.repository import InMemoryOperationAuditRepository
from src.production_readonly_adapters_v1.context import ProductionPaperOperationContextProvider
from src.production_readonly_adapters_v1.domain import (
    MARKET_ADAPTER_ID,
    MARKET_SOURCE,
    MarketQuoteStatus,
)
from src.production_readonly_adapters_v1.moex import (
    FreshMarketAdapterError,
    MarketResponseInvalidError,
    MoexIssFreshMarketAdapter,
)
from src.production_readonly_adapters_v1.ollama import (
    OllamaAgentInvalidResponseError,
    OllamaAgentModel,
    OllamaAgentTimeoutError,
    OllamaAgentUnavailableError,
)
from src.production_readonly_adapters_v1.reporting import build_adapter_artifact
from src.risk_engine_paper_v1.application import initial_paper_portfolio
from src.risk_engine_paper_v1.domain import PaperPosition, RiskPolicy
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
HttpHandler = Callable[[httpx.Request], httpx.Response]


def test_market_smoke_dispatch_does_not_require_state_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def smoke_result(tickers: list[str]) -> int:
        return len(tickers) - 2

    monkeypatch.setattr(paper_operation_cli, "_market_smoke", smoke_result)
    args = paper_operation_cli.build_parser().parse_args(["market-smoke", "SBER", "YDEX"])

    assert paper_operation_cli.run(args) == 0


def _instrument(ticker: str, *, board: str = "TQBR", supported: bool = True) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "legal_issuer": ticker,
        "instrument_uid": f"UID-{ticker}",
        "figi": f"FIGI-{ticker}",
        "board": board,
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
        "active": True,
        "supported": supported,
        "market_data_compatible": True,
        "feature_compatible": True,
        "lot_size": 10,
    }


def _moex_payload(
    ticker: str = "SBER",
    *,
    board: str = "TQBR",
    last: object = "100.5",
    bid: object = "100.4",
    ask: object = "100.6",
    lot: object = 10,
    systime: object = "2026-09-09 15:00:00",
    market_rows: int = 1,
) -> dict[str, Any]:
    market_row = [ticker, board, last, bid, ask, systime]
    return {
        "securities": {
            "columns": ["SECID", "BOARDID", "LOTSIZE"],
            "data": [[ticker, board, lot]],
        },
        "marketdata": {
            "columns": ["SECID", "BOARDID", "LAST", "BID", "OFFER", "SYSTIME"],
            "data": [market_row for _ in range(market_rows)],
        },
    }


def _market_adapter(
    handler: HttpHandler,
    *,
    max_age_seconds: float = 300,
    fetched_at: datetime = NOW + timedelta(seconds=2),
) -> MoexIssFreshMarketAdapter:
    return MoexIssFreshMarketAdapter(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        max_retries=0,
        user_agent="tests",
        max_age_seconds=max_age_seconds,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: fetched_at,
    )


def _ollama_model(handler: HttpHandler, *, max_retries: int = 0) -> OllamaAgentModel:
    return OllamaAgentModel(
        base_url="http://localhost:11434",
        model="qwen-test:1b",
        think=False,
        timeout_seconds=1,
        max_retries=max_retries,
        max_output_tokens=1024,
        random_seed=0,
        context_length=4096,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _agent_request() -> AgentModelRequest:
    return AgentModelRequest(
        prompt_version="agent-research-v1",
        system_prompt="read only",
        allowed_universe=[_instrument("SBER")],
        transcript=[{"role": "system", "content": "read only"}],
        safety_policy={"allowed_tool_names": ["get_market_context"]},
    )


def _proposal_output(as_of: datetime = NOW) -> dict[str, Any]:
    return {
        "as_of": as_of.isoformat(),
        "input_snapshot_as_of": {
            "portfolio_as_of": as_of.isoformat(),
            "market_data_as_of": as_of.isoformat(),
            "events_as_of": as_of.isoformat(),
        },
        "proposals": [
            {
                "ticker": "SBER",
                "action": "BUY",
                "agent_confidence": 0.6,
                "target_weight": 0.1,
                "holding_horizon": "1-5d",
                "thesis": ["deterministic fixture"],
                "risks": ["paper only"],
                "evidence": [],
                "data_quality": "GOOD",
            }
        ],
    }


def _ollama_body(content: object, *, thinking: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {"model": "qwen-test:1b", "message": message, "prompt_eval_count": 10, "eval_count": 20}


def test_ollama_agent_model_valid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["options"]["seed"] == 0
        return httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))

    response = _ollama_model(handler).complete(_agent_request())

    assert json.loads(response.final_output or "{}")["proposals"][0]["ticker"] == "SBER"


def test_ollama_agent_model_invalid_json_fails_closed() -> None:
    model = _ollama_model(lambda _request: httpx.Response(200, json=_ollama_body("not-json")))

    with pytest.raises(OllamaAgentInvalidResponseError):
        model.complete(_agent_request())


def test_ollama_agent_model_schema_violation_fails_closed(tmp_path: Path) -> None:
    model = _ollama_model(
        lambda _request: httpx.Response(
            200, json=_ollama_body(json.dumps({"as_of": NOW.isoformat()}))
        )
    )
    context = AgentDataContext(
        as_of=NOW,
        allowed_universe=[_instrument("SBER")],
        portfolio_snapshot={"as_of": NOW.isoformat()},
        market_context_snapshot={"as_of": NOW.isoformat(), "by_ticker": {}},
        event_context_snapshot={"events_as_of": NOW.isoformat(), "events": []},
        research_status_snapshot={"LIVE_RESEARCH_OPERATION_STATUS": "READY"},
    )
    result = run_read_only_research_agent_v1(
        config=AgentRunConfig(output_root=tmp_path / "agent", code_sha="a" * 40, created_at=NOW),
        model=model,
        deterministic_context=context,
    )

    assert result.AGENT_DECISION_STATUS == AgentDecisionStatus.INVALID_MODEL_OUTPUT
    assert result.final_proposals == []


def test_ollama_agent_model_timeout_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(OllamaAgentTimeoutError):
        _ollama_model(handler, max_retries=1).complete(_agent_request())


def test_ollama_agent_model_connection_error_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaAgentUnavailableError):
        _ollama_model(handler).complete(_agent_request())


def test_ollama_reasoning_field_not_persisted() -> None:
    model = _ollama_model(
        lambda _request: httpx.Response(
            200,
            json=_ollama_body(json.dumps(_proposal_output()), thinking="hidden reasoning"),
        )
    )

    response = model.complete(_agent_request())

    assert "thinking" not in response.model_dump_json()
    assert "thinking" not in json.dumps(model.last_metadata)


def test_ollama_agent_model_never_registers_broker_tools() -> None:
    source = Path("src/production_readonly_adapters_v1/ollama.py").read_text(encoding="utf-8")

    assert "tinvest" not in source.lower()
    assert "place_order" not in source
    assert "subprocess" not in source


def test_ollama_agent_model_rejects_arbitrary_remote_host() -> None:
    with pytest.raises(ValueError, match="localhost"):
        OllamaAgentModel(
            base_url="https://example.com",
            model="qwen-test:1b",
            think=False,
            timeout_seconds=1,
            max_retries=0,
            max_output_tokens=1024,
            random_seed=0,
            context_length=4096,
        )


def test_ollama_agent_model_accepts_compose_host_bridge() -> None:
    model = OllamaAgentModel(
        base_url="http://host.docker.internal:11434",
        model="qwen-test:1b",
        think=False,
        timeout_seconds=1,
        max_retries=0,
        max_output_tokens=1024,
        random_seed=0,
        context_length=4096,
    )

    assert model.model_id == "ollama:qwen-test:1b"


def test_no_fake_model_fallback_in_production() -> None:
    source = Path("src/production_readonly_adapters_v1/factory.py").read_text(encoding="utf-8")

    assert "FakeAgentModel" not in source


def test_moex_market_adapter_builds_valid_quote() -> None:
    snapshot = _market_adapter(lambda _request: httpx.Response(200, json=_moex_payload())).fetch(
        universe=[_instrument("SBER")], operation_as_of=NOW
    )

    quote = snapshot.quotes[0]
    assert quote["ticker"] == "SBER"
    assert quote["last_price"] == 100.5
    assert quote["bid"] == 100.4
    assert quote["ask"] == 100.6
    assert quote["lot_size"] == 10
    assert quote["source"] == MARKET_SOURCE
    assert snapshot.quote_audit[0].status == MarketQuoteStatus.FRESH


def test_market_timestamp_comes_from_source_not_local_now() -> None:
    fetched = NOW + timedelta(hours=2)
    snapshot = _market_adapter(
        lambda _request: httpx.Response(200, json=_moex_payload()),
        fetched_at=fetched,
    ).fetch(universe=[_instrument("SBER")], operation_as_of=NOW)

    assert snapshot.quotes[0]["market_data_as_of"] == NOW.isoformat()
    assert snapshot.quotes[0]["fetched_at"] == fetched.isoformat()


def test_future_market_quote_blocked() -> None:
    snapshot = _market_adapter(
        lambda _request: httpx.Response(200, json=_moex_payload(systime="2026-09-09 15:00:01"))
    ).fetch(universe=[_instrument("SBER")], operation_as_of=NOW)

    assert snapshot.quote_audit[0].status == MarketQuoteStatus.FUTURE


def test_stale_market_quote_degraded_or_blocked() -> None:
    snapshot = _market_adapter(
        lambda _request: httpx.Response(200, json=_moex_payload(systime="2026-09-09 14:50:00"))
    ).fetch(universe=[_instrument("SBER")], operation_as_of=NOW)

    assert snapshot.quote_audit[0].status == MarketQuoteStatus.STALE


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_moex_payload(last=0), "INVALID_LAST_PRICE"),
        (_moex_payload(lot=None), "MISSING_OR_INVALID_LOT_SIZE"),
        (_moex_payload(board="TQTF"), "WRONG_BOARD"),
        (_moex_payload(bid=-1), "INVALID_BID"),
        (_moex_payload(ask=0), "INVALID_ASK"),
        (_moex_payload(bid=101, ask=100), "BID_ABOVE_ASK"),
        (_moex_payload(systime=None), "MARKET_TIMESTAMP_UNAVAILABLE"),
    ],
)
def test_invalid_market_rows_are_rejected(payload: dict[str, Any], reason: str) -> None:
    snapshot = _market_adapter(lambda _request: httpx.Response(200, json=payload)).fetch(
        universe=[_instrument("SBER")], operation_as_of=NOW
    )

    assert snapshot.quotes == []
    assert snapshot.quote_audit[0].reason == reason


def test_wrong_canonical_board_rejected_without_request() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_moex_payload())

    snapshot = _market_adapter(handler).fetch(
        universe=[_instrument("SBER", board="TQTF")], operation_as_of=NOW
    )

    assert calls == 0
    assert snapshot.quote_audit[0].reason == "WRONG_BOARD"


def test_duplicate_ticker_rejected() -> None:
    with pytest.raises(MarketResponseInvalidError, match="DUPLICATE"):
        _market_adapter(lambda _request: httpx.Response(200, json={})).fetch(
            universe=[_instrument("SBER"), _instrument("SBER")], operation_as_of=NOW
        )


def test_source_payload_sha_is_deterministic() -> None:
    first = _market_adapter(lambda _request: httpx.Response(200, json=_moex_payload())).fetch(
        universe=[_instrument("SBER")], operation_as_of=NOW
    )
    second = _market_adapter(lambda _request: httpx.Response(200, json=_moex_payload())).fetch(
        universe=[_instrument("SBER")], operation_as_of=NOW
    )

    assert first.source_payload_sha == second.source_payload_sha
    assert first.quotes[0]["source_payload_sha"] == second.quotes[0]["source_payload_sha"]


def test_missing_held_quote_preserves_incomplete_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch

    def handler(request: httpx.Request) -> httpx.Response:
        ticker = request.url.path.split("/")[-1].removesuffix(".json")
        return httpx.Response(
            200,
            json=_moex_payload(ticker, market_rows=0)
            if ticker == "YDEX"
            else _moex_payload(ticker),
        )

    provider = ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=Path("unused"), code_sha="a" * 40),
        market_adapter=_market_adapter(handler),
        risk_policy=RiskPolicy(),
        configured_max_age_seconds=300,
        universe_loader=_fixture_universe,
        event_loader=_fixture_events,
        research_loader=_fixture_research,
    )
    portfolio = initial_paper_portfolio(NOW).model_copy(
        update={
            "positions": [
                PaperPosition(
                    ticker="YDEX",
                    quantity=1,
                    average_cost=100,
                    last_price=100,
                    market_value=100,
                    weight=0.0001,
                    unrealized_pnl=0,
                    mark_as_of=NOW - timedelta(hours=1),
                )
            ]
        }
    )
    context = provider.load(
        operation_as_of=NOW,
        portfolio=portfolio,
        policy=PaperOperationPolicy(max_operation_universe=2),
    )

    assert context.universe[0]["ticker"] == "YDEX"
    assert {quote.ticker for quote in context.market_quotes} == {"GAZP"}
    assert context.market_context["missing_quote_count"] == 1
    assert context.market_context["quotes"][0]["ticker"] == "YDEX"
    assert context.market_context["quotes"][0]["status"] == "MISSING"


def _fixture_universe(_config: AgentRunConfig) -> list[dict[str, Any]]:
    return [_instrument("GAZP"), _instrument("SBER"), _instrument("YDEX")]


def _fixture_events(*_args: Any) -> dict[str, Any]:
    return {"events_as_of": NOW.isoformat(), "events": []}


def _fixture_research(*_args: Any) -> dict[str, Any]:
    return {
        "research_status_as_of": NOW.isoformat(),
        "LIVE_RESEARCH_OPERATION_STATUS": "READY",
        "OPERATIONAL_BURN_IN": "PASS",
        "SOURCE_FAILURE_ISOLATION": True,
        "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
        "seal": {"sealed_epoch_verified": True, "violations": 0},
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
    }


def _production_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    research_loader: Callable[..., dict[str, Any]] = _fixture_research,
) -> ProductionPaperOperationContextProvider:
    del monkeypatch

    def handler(request: httpx.Request) -> httpx.Response:
        ticker = request.url.path.split("/")[-1].removesuffix(".json")
        return httpx.Response(200, json=_moex_payload(ticker))

    return ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=Path("unused"), code_sha="a" * 40),
        market_adapter=_market_adapter(handler),
        risk_policy=RiskPolicy(),
        configured_max_age_seconds=300,
        universe_loader=_fixture_universe,
        event_loader=_fixture_events,
        research_loader=research_loader,
    )


def test_production_context_includes_and_prioritizes_all_held_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _production_provider(monkeypatch)
    portfolio = initial_paper_portfolio(NOW).model_copy(
        update={
            "positions": [
                PaperPosition(
                    ticker="YDEX",
                    quantity=1,
                    average_cost=100,
                    last_price=100,
                    market_value=100,
                    weight=0.0001,
                    unrealized_pnl=0,
                    mark_as_of=NOW,
                )
            ]
        }
    )
    context = provider.load(
        operation_as_of=NOW,
        portfolio=portfolio,
        policy=PaperOperationPolicy(max_operation_universe=2),
    )

    assert [row["ticker"] for row in context.universe] == ["YDEX", "GAZP"]
    assert context.market_context["market_adapter_id"] == MARKET_ADAPTER_ID
    assert context.market_context["fresh_quote_count"] == 2


def test_operation_dry_run_uses_production_model_and_market_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _production_provider(monkeypatch)
    model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    paper = InMemoryPaperLedgerRepository()
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
        policy=PaperOperationPolicy(),
    )

    assert run.status == PaperOperationStatus.SUCCESS
    assert run.market_adapter_id == MARKET_ADAPTER_ID
    assert run.market_audit["fresh_quote_count"] == 3
    assert run.paper_execution_status == "SKIPPED_DRY_RUN"
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_execute_still_requires_double_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _production_provider(monkeypatch)
    model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    paper = InMemoryPaperLedgerRepository()
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
        policy=PaperOperationPolicy(),
    )

    assert run.status_code == "PAPER_EXECUTION_DISABLED"
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_future_market_data_zero_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    provider = ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=tmp_path / "unused", code_sha="a" * 40),
        market_adapter=_market_adapter(
            lambda _request: httpx.Response(200, json=_moex_payload(systime="2026-09-09 15:00:01"))
        ),
        risk_policy=RiskPolicy(),
        configured_max_age_seconds=300,
        universe_loader=_fixture_universe,
        event_loader=_fixture_events,
        research_loader=_fixture_research,
    )
    model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert "FUTURE_MARKET_QUOTE" in run.reasons
    assert model.last_metadata == {}


def test_market_failure_zero_model_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch

    def market_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    provider = ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=tmp_path / "unused", code_sha="a" * 40),
        market_adapter=_market_adapter(market_failure),
        risk_policy=RiskPolicy(),
        configured_max_age_seconds=300,
        universe_loader=_fixture_universe,
        event_loader=_fixture_events,
        research_loader=_fixture_research,
    )
    model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code.startswith("CONTEXT_UNAVAILABLE")
    assert model.last_metadata == {}
    assert run.safety.PAPER_ORDERS_FILLED == 0


def test_same_slot_idempotency_survives_market_context_and_model_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _production_provider(monkeypatch)
    first_model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=first_model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
    )
    changed_model = OllamaAgentModel(
        base_url="http://localhost:11434",
        model="changed-model:2b",
        think=False,
        timeout_seconds=1,
        max_retries=0,
        max_output_tokens=1024,
        random_seed=1,
        context_length=2048,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, json=_ollama_body(json.dumps(_proposal_output(NOW + timedelta(minutes=2))))
                )
            )
        ),
    )
    duplicate = run_paper_operation(
        operation_as_of=NOW + timedelta(minutes=2),
        mode=PaperOperationMode.DRY_RUN,
        model=changed_model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path / "operation",
        code_sha="b" * 40,
    )

    assert first.status == PaperOperationStatus.SUCCESS
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert changed_model.last_metadata == {}
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_agent_failure_zero_fills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _production_provider(monkeypatch)
    model = _ollama_model(lambda _request: httpx.Response(200, json=_ollama_body("not-json")))
    paper = InMemoryPaperLedgerRepository()
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "AGENT_MODEL_ERROR"
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_research_degraded_zero_exposure_increase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def degraded(*_args: Any) -> dict[str, Any]:
        return {
            "research_status_as_of": NOW.isoformat(),
            "LIVE_RESEARCH_OPERATION_STATUS": "DEGRADED",
            "OPERATIONAL_BURN_IN": "PARTIAL",
            "SOURCE_FAILURE_ISOLATION": False,
            "seal": {"sealed_epoch_verified": True, "violations": 0},
        }

    provider = _production_provider(monkeypatch, research_loader=degraded)
    model = _ollama_model(
        lambda _request: httpx.Response(200, json=_ollama_body(json.dumps(_proposal_output())))
    )
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=model,
        context_provider=provider,
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path / "operation",
        code_sha="a" * 40,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "LIVE_RESEARCH_NOT_READY"
    assert model.last_metadata == {}
    assert run.safety.PAPER_ORDERS_FILLED == 0


def test_market_adapter_failure_is_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(FreshMarketAdapterError, match="MOEX_REQUEST_FAILED"):
        _market_adapter(handler).fetch(universe=[_instrument("SBER")], operation_as_of=NOW)


def test_effective_freshness_never_weaker_than_risk_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    provider = ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=Path("unused"), code_sha="a" * 40),
        market_adapter=_market_adapter(
            lambda _request: httpx.Response(200, json=_moex_payload()), max_age_seconds=300
        ),
        risk_policy=RiskPolicy(max_stale_market_age=timedelta(seconds=120)),
        configured_max_age_seconds=300,
        universe_loader=_fixture_universe,
        event_loader=_fixture_events,
        research_loader=_fixture_research,
    )

    with pytest.raises(ValueError, match="TOO_WEAK"):
        provider.load(
            operation_as_of=NOW,
            portfolio=initial_paper_portfolio(NOW),
            policy=PaperOperationPolicy(),
        )


def test_market_payload_hash_matches_canonical_payload() -> None:
    payload = _moex_payload()
    snapshot = _market_adapter(lambda _request: httpx.Response(200, json=payload)).fetch(
        universe=[_instrument("SBER")], operation_as_of=NOW
    )

    assert snapshot.source_payload_sha == sha256_payload([{"ticker": "SBER", "payload": payload}])


def test_adapter_artifact_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    base_sha = "a" * 40
    head_sha = "b" * 40

    first_manifest = build_adapter_artifact(
        output_root=first,
        work_root=tmp_path / "first-work",
        base_main_sha=base_sha,
        head_sha=head_sha,
    )
    second_manifest = build_adapter_artifact(
        output_root=second,
        work_root=tmp_path / "second-work",
        base_main_sha=base_sha,
        head_sha=head_sha,
    )

    expected = {
        "adapter-policy.json",
        "agent-request.json",
        "agent-response.json",
        "agent-validation.json",
        "failure-cases.json",
        "manifest.json",
        "market-adapter-verification.json",
        "market-fixture.json",
        "market-snapshot.json",
        "pit-verification.json",
        "production-dry-run.json",
        "report.md",
        "safety.json",
    }
    assert first_manifest == second_manifest
    assert first_manifest["PRODUCTION_DRY_RUN_READY"] == "YES"
    assert first_manifest["PIT_SAFETY"] == "PASS"
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
