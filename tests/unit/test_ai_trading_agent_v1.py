from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.ai_trading_agent_v1.application import (
    PROMPT_VERSION,
    AgentDataContext,
    AgentRunConfig,
    FakeAgentModel,
    agent_policy,
    build_read_only_tool_registry,
    existing_market_context,
    research_status,
    run_read_only_research_agent_v1,
    sample_agent_context,
)
from src.ai_trading_agent_v1.domain import (
    AgentDecisionStatus,
    AgentModelResponse,
    ProposalValidationStatus,
    ToolCallRequest,
    TradeAction,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def _config(tmp_path: Path, **overrides: object) -> AgentRunConfig:
    config = AgentRunConfig(
        output_root=tmp_path / "run",
        code_sha="a" * 40,
        run_id="unit-agent-run",
        created_at=NOW,
    )
    return replace(config, **overrides)


def _proposal(
    ticker: str = "SBER",
    action: str = "BUY",
    confidence: float = 0.7,
    weight: float = 0.1,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": confidence,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": ["Bounded read-only research thesis."],
        "risks": ["Research data can be incomplete."],
        "evidence": [{"type": "EVENT", "id": "event-1"}],
        "data_quality": "GOOD",
    }


def _output(*proposals: dict[str, object]) -> str:
    return json.dumps(
        {
            "as_of": NOW.isoformat(),
            "input_snapshot_as_of": {
                "portfolio_as_of": NOW.isoformat(),
                "market_data_as_of": NOW.isoformat(),
                "events_as_of": NOW.isoformat(),
            },
            "proposals": list(proposals),
        }
    )


def _run(
    tmp_path: Path,
    *responses: AgentModelResponse | Exception,
    context: AgentDataContext | None = None,
    **config_overrides: object,
):
    model = FakeAgentModel(list(responses))
    result = run_read_only_research_agent_v1(
        config=_config(tmp_path, **config_overrides),
        model=model,
        deterministic_context=context or sample_agent_context(NOW),
    )
    return result, model


def test_model_calls_tool_and_resumes_with_multiple_calls(tmp_path: Path) -> None:
    result, model = _run(
        tmp_path,
        AgentModelResponse(
            tool_calls=[
                ToolCallRequest(name="get_portfolio_context"),
                ToolCallRequest(name="get_market_context", arguments={"ticker": "SBER"}),
            ]
        ),
        AgentModelResponse(final_output=_output(_proposal())),
    )

    assert result.AGENT_DECISION_STATUS == AgentDecisionStatus.VALID
    assert len(model.requests) == 2
    assert [call.status for call in result.tool_calls] == ["OK", "OK"]
    assert model.requests[1].transcript[-1]["tool_name"] == "get_market_context"


def test_tool_failure_isolated_and_agent_can_finish(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        AgentModelResponse(tool_calls=[ToolCallRequest(name="arbitrary_search")]),
        AgentModelResponse(final_output=_output(_proposal(action="HOLD"))),
    )

    assert result.AGENT_DECISION_STATUS == AgentDecisionStatus.VALID
    assert result.tool_calls[0].status == "ERROR"
    assert result.tool_calls[0].error == "TOOL_NOT_REGISTERED"


def test_tool_arguments_are_typed_and_fail_independently(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        AgentModelResponse(tool_calls=[ToolCallRequest(name="get_market_context", arguments={})]),
        AgentModelResponse(final_output=_output(_proposal(action="HOLD"))),
    )

    assert result.AGENT_DECISION_STATUS == AgentDecisionStatus.VALID
    assert result.tool_calls[0].status == "ERROR"
    assert result.tool_calls[0].error == "ValidationError"


def test_step_and_tool_limits_fail_closed(tmp_path: Path) -> None:
    looping = [
        AgentModelResponse(tool_calls=[ToolCallRequest(name="get_research_status")]),
        AgentModelResponse(tool_calls=[ToolCallRequest(name="get_research_status")]),
    ]
    step_result, _ = _run(tmp_path / "step", *looping, max_agent_steps=2)
    tool_result, _ = _run(
        tmp_path / "tool",
        AgentModelResponse(
            tool_calls=[
                ToolCallRequest(name="get_research_status"),
                ToolCallRequest(name="get_portfolio_context"),
            ]
        ),
        max_tool_calls=1,
    )

    assert step_result.AGENT_DECISION_STATUS == AgentDecisionStatus.ABORTED_LIMIT
    assert step_result.validation.reasons == ["MAX_AGENT_STEPS_EXCEEDED"]
    assert tool_result.AGENT_DECISION_STATUS == AgentDecisionStatus.ABORTED_LIMIT
    assert tool_result.validation.reasons == ["MAX_TOOL_CALLS_EXCEEDED"]


def test_model_and_schema_errors_fail_closed(tmp_path: Path) -> None:
    model_error, _ = _run(tmp_path / "model", RuntimeError("offline"))
    malformed, _ = _run(tmp_path / "malformed", AgentModelResponse(final_output="not-json"))
    bad_confidence, _ = _run(
        tmp_path / "confidence",
        AgentModelResponse(final_output=_output(_proposal(confidence=1.1))),
    )

    assert model_error.AGENT_DECISION_STATUS == AgentDecisionStatus.MODEL_ERROR
    assert malformed.AGENT_DECISION_STATUS == AgentDecisionStatus.INVALID_MODEL_OUTPUT
    assert bad_confidence.AGENT_DECISION_STATUS == AgentDecisionStatus.INVALID_MODEL_OUTPUT
    assert not model_error.AGENT_RESEARCH_CAPABILITY_READY


@pytest.mark.parametrize("action", list(TradeAction))
def test_all_typed_actions_are_accepted(action: TradeAction, tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        AgentModelResponse(final_output=_output(_proposal(action=action.value))),
    )

    assert result.validation.status == ProposalValidationStatus.VALID
    assert result.final_proposals[0].action == action


@pytest.mark.parametrize(
    ("proposals", "reason"),
    [
        ([_proposal(ticker="FAKE")], "UNSUPPORTED_TICKER:FAKE"),
        ([_proposal(weight=0.21)], "TARGET_WEIGHT_ABOVE_MAX:SBER"),
        (
            [_proposal(action="BUY"), _proposal(action="SELL")],
            "CONFLICTING_DUPLICATE_PROPOSAL:SBER",
        ),
    ],
)
def test_proposal_policy_rejects_invalid_requests(
    proposals: list[dict[str, object]], reason: str, tmp_path: Path
) -> None:
    result, _ = _run(
        tmp_path,
        AgentModelResponse(final_output=_output(*proposals)),
    )

    assert result.validation.status == ProposalValidationStatus.INVALID
    assert reason in result.validation.reasons
    assert result.final_proposals == []


def test_critical_research_failure_rejects_buy(tmp_path: Path) -> None:
    context = sample_agent_context(NOW)
    context = replace(
        context,
        research_status_snapshot={
            **context.research_status_snapshot,
            "LIVE_RESEARCH_OPERATION_STATUS": "FAIL",
        },
    )
    result, _ = _run(
        tmp_path,
        AgentModelResponse(final_output=_output(_proposal())),
        context=context,
    )

    assert "BUY_WITH_CRITICAL_RESEARCH_FAIL:SBER" in result.validation.reasons


def test_stale_market_context_degrades_while_fresh_is_valid(tmp_path: Path) -> None:
    fresh = sample_agent_context(NOW)
    stale_market = {
        **fresh.market_context_snapshot,
        "by_ticker": {
            **fresh.market_context_snapshot["by_ticker"],
            "SBER": {
                **fresh.market_context_snapshot["by_ticker"]["SBER"],
                "market_data_as_of": (NOW - timedelta(days=2)).isoformat(),
                "stale": True,
            },
        },
    }
    stale = replace(fresh, market_context_snapshot=stale_market)
    fresh_result, _ = _run(
        tmp_path / "fresh",
        AgentModelResponse(final_output=_output(_proposal())),
        context=fresh,
    )
    stale_result, _ = _run(
        tmp_path / "stale",
        AgentModelResponse(final_output=_output(_proposal())),
        context=stale,
    )

    assert fresh_result.AGENT_DECISION_STATUS == AgentDecisionStatus.VALID
    assert stale_result.AGENT_DECISION_STATUS == AgentDecisionStatus.DEGRADED_STALE_DATA
    assert stale_result.validation.stale_tickers == ["SBER"]


def test_tools_and_capabilities_are_strictly_read_only(tmp_path: Path) -> None:
    context = sample_agent_context(NOW)
    config = _config(tmp_path)
    tools = build_read_only_tool_registry(context, config)
    policy = agent_policy(config, tools)

    assert set(tools) == {
        "get_portfolio_context",
        "get_recent_events",
        "get_market_context",
        "get_instrument_context",
        "get_research_status",
    }
    assert set(policy["allowed_capabilities"]) == {
        "READ_MARKET",
        "READ_EVENTS",
        "READ_PORTFOLIO",
        "READ_SYSTEM",
    }
    assert not set(policy["forbidden_tools"]).intersection(tools)


def test_audit_artifact_is_complete_hashed_and_immutable(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        AgentModelResponse(final_output=_output(_proposal())),
    )
    root = tmp_path / "run"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    required = {
        "manifest.json",
        "agent-policy.json",
        "allowed-universe.json",
        "sample-portfolio.json",
        "sample-market-context.json",
        "sample-event-context.json",
        "sample-tool-calls.jsonl",
        "sample-proposals.json",
        "validation.json",
        "safety.json",
        "report.md",
    }
    assert required.issubset({path.name for path in root.iterdir()})
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["agent_model_id"] == "fake-agent-v1"
    assert set(manifest["input_snapshot_hashes"]) == {
        "allowed_universe",
        "portfolio_snapshot",
        "market_context_snapshot",
        "event_context_snapshot",
        "research_status_snapshot",
    }
    assert manifest["BROKER_MUTATIONS"] == 0
    assert manifest["REAL_ORDERS_SENT"] == 0
    assert manifest["PAPER_ORDERS_SENT"] == 0
    assert manifest["PORTFOLIO_MUTATIONS"] == 0
    assert manifest["LIVE_OUTCOMES_READ"] == 0
    assert manifest["LIVE_TARGETS_COMPUTED"] == 0
    assert manifest["LIVE_POST_EVENT_PRICE_READS"] == 0
    assert manifest["OLD_FUTURE_HOLDOUT_OPENED"] is False
    assert result.AGENT_RESEARCH_CAPABILITY_READY

    with pytest.raises(FileExistsError, match="immutable agent output"):
        run_read_only_research_agent_v1(
            config=_config(tmp_path),
            model=FakeAgentModel([AgentModelResponse(final_output=_output(_proposal()))]),
            deterministic_context=sample_agent_context(NOW),
        )


def test_existing_market_features_are_consumed_without_target_rows(tmp_path: Path) -> None:
    path = tmp_path / "features.jsonl"
    path.write_text(
        json.dumps(
            {
                "ticker": "SBER",
                "row_id": "SBER:2026-09-08",
                "feature_as_of": "2026-09-08",
                "features": {
                    "return_1d": 0.01,
                    "volatility_20d": 0.02,
                    "relative_return_1d": 0.004,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    market = existing_market_context(path, [{"ticker": "SBER"}], NOW)

    assert market["calculation"] == "EXISTING_FEATURE_ARTIFACT_ONLY"
    assert market["by_ticker"]["SBER"]["return_1d"] == 0.01
    assert "target" not in market["by_ticker"]["SBER"]


def test_operational_status_uses_canonical_proof_and_fails_closed(tmp_path: Path) -> None:
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "OPERATIONAL_BURN_IN": "PASS",
                "SOURCE_FAILURE_ISOLATION": True,
                "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
                "BROKER_MUTATIONS": 0,
                "LIVE_OUTCOMES_READ": 0,
                "LIVE_TARGETS_COMPUTED": 0,
                "LIVE_POST_EVENT_PRICE_READS": 0,
            }
        ),
        encoding="utf-8",
    )

    proven = research_status(operation_root, proof_path, NOW)
    missing = research_status(operation_root, tmp_path / "missing.json", NOW)

    assert proven["OPERATIONAL_BURN_IN"] == "PASS"
    assert proven["SOURCE_FAILURE_ISOLATION_PROOF_LEVEL"] == "APPLICATION_PROOF"
    assert missing["OPERATIONAL_BURN_IN"] == "PARTIAL"
    assert missing["SOURCE_FAILURE_ISOLATION"] == "NOT_PROVEN"
