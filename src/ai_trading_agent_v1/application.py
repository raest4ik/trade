from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from src.ai_trading_agent_v1.domain import (
    MAX_AGENT_PROPOSED_WEIGHT,
    AgentDecisionStatus,
    AgentMode,
    AgentModelRequest,
    AgentModelResponse,
    AgentRunResult,
    NoArguments,
    ProposalValidationResult,
    ProposalValidationStatus,
    RecentEventsArguments,
    SafetyCounters,
    StructuredAgentOutput,
    TickerArguments,
    ToolCallRecord,
    ToolCallRequest,
    ToolCapability,
    TradeAction,
    TradeProposal,
)
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.free_live_issuer_accumulation.operation import (
    build_operation_status,
    verify_operation_seal,
)
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    DEFAULT_INSTRUMENT_MAPPING_PATH,
    load_instrument_mapping_rows,
)
from src.moex_target_source_discovery_v5.application import (
    DEFAULT_OPERATION_ROOT,
    canonical_registry_from_mapping,
)

ARTIFACT_VERSION = "ai-trading-agent-v1"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
DEFAULT_LIVE_ROOT = Path("artifacts/free-live-issuer-accumulation-v1")
DEFAULT_MARKET_FEATURES_PATH = Path("artifacts/tinvest-market-baseline-features-v1/features.jsonl")
DEFAULT_OPERATIONAL_PROOF_PATH = Path(
    "artifacts/moex-issuer-controlled-channel-discovery-v6/manifest.json"
)
PROMPT_VERSION = "agent-research-v1"
MAX_AGENT_STEPS = 6
MAX_TOOL_CALLS = 12
MAX_EVENTS_PER_TICKER = 5
MAX_TICKERS_PER_RUN = 10
MAX_STALENESS_MINUTES = 24 * 60
WRITE_TOOL_NAMES = {"buy", "sell", "place_order", "cancel_order"}

SYSTEM_PROMPT = """You are a read-only trading research assistant, not a broker.
You cannot execute trades, place orders, cancel orders, mutate a portfolio, or call write tools.
Use only supplied read-only tools and allowed canonical instruments.
Do not invent market facts or tickers outside the allowed universe.
If data is missing, stale, contradictory, or research health is failed, prefer HOLD or AVOID.
Return structured trade proposals with thesis, risks, evidence, target_weight, and agent_confidence.
agent_confidence is subjective research confidence, not a calibrated probability or guarantee."""


class AgentModel(Protocol):
    model_id: str

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        """Return either tool calls or final structured JSON."""
        raise NotImplementedError


class FakeAgentModel:
    def __init__(
        self, responses: Sequence[AgentModelResponse | Exception], model_id: str = "fake-agent-v1"
    ):
        self.responses = list(responses)
        self.model_id = model_id
        self.requests: list[AgentModelRequest] = []

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("fake model exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class UnconfiguredAgentModel:
    model_id = "unconfigured-agent-model"

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        raise RuntimeError("Agent model provider is not configured")


@dataclass(frozen=True, slots=True)
class ReadOnlyTool:
    name: str
    capability: ToolCapability
    arguments_model: type[BaseModel]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    output_root: Path
    code_sha: str
    run_id: str | None = None
    created_at: datetime | None = None
    instrument_mapping_path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH
    operation_root: Path = DEFAULT_OPERATION_ROOT
    live_root: Path = DEFAULT_LIVE_ROOT
    market_features_path: Path = DEFAULT_MARKET_FEATURES_PATH
    operational_proof_path: Path = DEFAULT_OPERATIONAL_PROOF_PATH
    max_agent_steps: int = MAX_AGENT_STEPS
    max_tool_calls: int = MAX_TOOL_CALLS
    max_events_per_ticker: int = MAX_EVENTS_PER_TICKER
    max_tickers_per_run: int = MAX_TICKERS_PER_RUN
    max_staleness_minutes: int = MAX_STALENESS_MINUTES
    max_agent_proposed_weight: float = MAX_AGENT_PROPOSED_WEIGHT


@dataclass(frozen=True, slots=True)
class AgentDataContext:
    as_of: datetime
    allowed_universe: list[dict[str, Any]]
    portfolio_snapshot: dict[str, Any]
    market_context_snapshot: dict[str, Any]
    event_context_snapshot: dict[str, Any]
    research_status_snapshot: dict[str, Any]


def run_read_only_research_agent_v1(
    *,
    config: AgentRunConfig,
    model: AgentModel,
    deterministic_context: AgentDataContext | None = None,
) -> AgentRunResult:
    created_at = config.created_at or datetime.now(UTC)
    run_id = config.run_id or f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    if config.output_root.exists() and any(config.output_root.iterdir()):
        raise FileExistsError("immutable agent output already exists")
    config.output_root.mkdir(parents=True, exist_ok=False)

    context = deterministic_context or build_agent_data_context(config, created_at)
    tools = build_read_only_tool_registry(context, config)
    policy = agent_policy(config, tools)
    transcript: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    tool_calls: list[ToolCallRecord] = []
    raw_model_output: str | None = None
    parsed: StructuredAgentOutput | None = None
    proposals: list[TradeProposal] = []
    validation = ProposalValidationResult(
        status=ProposalValidationStatus.INVALID,
        reasons=["MODEL_DID_NOT_RETURN_OUTPUT"],
    )
    decision_status = AgentDecisionStatus.INVALID

    try:
        for step in range(1, config.max_agent_steps + 1):
            response = model.complete(
                AgentModelRequest(
                    prompt_version=PROMPT_VERSION,
                    system_prompt=SYSTEM_PROMPT,
                    allowed_universe=context.allowed_universe,
                    transcript=transcript,
                    safety_policy=policy,
                )
            )
            if response.tool_calls:
                for call in response.tool_calls:
                    if len(tool_calls) >= config.max_tool_calls:
                        decision_status = AgentDecisionStatus.ABORTED_LIMIT
                        validation = ProposalValidationResult(
                            status=ProposalValidationStatus.INVALID,
                            reasons=["MAX_TOOL_CALLS_EXCEEDED"],
                        )
                        return finalize_agent_result(
                            run_id,
                            created_at,
                            config,
                            model,
                            context,
                            tool_calls,
                            raw_model_output,
                            parsed,
                            proposals,
                            validation,
                            decision_status,
                        )
                    record, result = call_tool(step, call.name, call.arguments, tools)
                    tool_calls.append(record)
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_name": call.name,
                            "status": record.status,
                            "result_hash": record.result_hash,
                            "result": result,
                        }
                    )
                continue

            raw_model_output = response.final_output
            if raw_model_output is None:
                decision_status = AgentDecisionStatus.INVALID_MODEL_OUTPUT
                validation = ProposalValidationResult(
                    status=ProposalValidationStatus.INVALID,
                    reasons=["FINAL_OUTPUT_MISSING"],
                )
                break
            try:
                parsed = StructuredAgentOutput.model_validate_json(raw_model_output)
            except ValidationError as exc:
                decision_status = AgentDecisionStatus.INVALID_MODEL_OUTPUT
                validation = ProposalValidationResult(
                    status=ProposalValidationStatus.INVALID,
                    reasons=[f"STRUCTURED_OUTPUT_SCHEMA_INVALID:{exc.errors()[0]['type']}"],
                )
                break
            validation = validate_proposals(parsed.proposals, context, config)
            proposals = (
                parsed.proposals if validation.status != ProposalValidationStatus.INVALID else []
            )
            decision_status = decision_status_from_validation(validation)
            break
        else:
            decision_status = AgentDecisionStatus.ABORTED_LIMIT
            validation = ProposalValidationResult(
                status=ProposalValidationStatus.INVALID,
                reasons=["MAX_AGENT_STEPS_EXCEEDED"],
            )
    except Exception as exc:
        decision_status = AgentDecisionStatus.MODEL_ERROR
        validation = ProposalValidationResult(
            status=ProposalValidationStatus.INVALID,
            reasons=[f"MODEL_ERROR:{type(exc).__name__}"],
        )

    return finalize_agent_result(
        run_id,
        created_at,
        config,
        model,
        context,
        tool_calls,
        raw_model_output,
        parsed,
        proposals,
        validation,
        decision_status,
    )


def build_agent_data_context(config: AgentRunConfig, as_of: datetime) -> AgentDataContext:
    universe = build_allowed_universe(config)
    selected = universe[: config.max_tickers_per_run]
    portfolio = synthetic_portfolio(selected, as_of)
    events = recent_event_context(
        config.live_root, [row["ticker"] for row in selected], as_of, config
    )
    market = existing_market_context(config.market_features_path, selected, as_of)
    research = research_status(config.operation_root, config.operational_proof_path, as_of)
    return AgentDataContext(as_of, selected, portfolio, market, events, research)


def build_allowed_universe(config: AgentRunConfig) -> list[dict[str, Any]]:
    mapping_rows = load_instrument_mapping_rows(config.instrument_mapping_path)
    if not mapping_rows:
        return []
    registry = canonical_registry_from_mapping(mapping_rows)
    mapping_by_ticker = {str(row.get("ticker", "")).upper(): row for row in mapping_rows}
    rows: list[dict[str, Any]] = []
    for ticker in sorted(registry):
        canonical = registry[ticker]
        mapping = mapping_by_ticker.get(ticker, {})
        if mapping_rows and not supported_mapping(mapping):
            continue
        rows.append(
            {
                "ticker": ticker,
                "legal_issuer": canonical.legal_issuer,
                "instrument_uid": mapping.get("instrument_uid"),
                "figi": mapping.get("figi"),
                "board": getattr(canonical, "primary_board", mapping.get("class_code", "TQBR")),
                "instrument_type": mapping.get("instrument_type", "INSTRUMENT_TYPE_SHARE"),
                "active": bool(mapping.get("first_1day_candle_date", "seed-registry")),
                "supported": True,
                "market_data_compatible": bool(
                    not mapping_rows
                    or (
                        mapping.get("instrument_uid")
                        and mapping.get("figi")
                        and mapping.get("first_1day_candle_date")
                    )
                ),
                "feature_compatible": bool(
                    not mapping_rows
                    or (
                        mapping.get("instrument_uid")
                        and mapping.get("figi")
                        and mapping.get("first_1day_candle_date")
                    )
                ),
            }
        )
    return rows


def supported_mapping(row: dict[str, Any]) -> bool:
    return (
        str(row.get("instrument_type", "")).upper() == "INSTRUMENT_TYPE_SHARE"
        and str(row.get("class_code", "")).upper() == "TQBR"
        and bool(str(row.get("instrument_uid", "")).strip())
        and bool(str(row.get("figi", "")).strip())
        and bool(str(row.get("first_1day_candle_date", "")).strip())
    )


def synthetic_portfolio(universe: Sequence[dict[str, Any]], as_of: datetime) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    equity = 1_000_000.0
    weights = [0.08, 0.05]
    for row, weight in zip(universe[:2], weights, strict=False):
        price = 100.0 + len(str(row["ticker"]))
        quantity = round(equity * weight / price, 4)
        positions.append(
            {
                "ticker": row["ticker"],
                "quantity": quantity,
                "average_price": round(price * 0.97, 4),
                "current_price": price,
                "current_weight": weight,
                "unrealized_pnl": round(quantity * price * 0.03, 4),
            }
        )
    return {"as_of": as_of.isoformat(), "cash": 870_000.0, "equity": equity, "positions": positions}


def synthetic_market_context(universe: Sequence[dict[str, Any]], as_of: datetime) -> dict[str, Any]:
    by_ticker: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(universe):
        ticker = str(row["ticker"])
        by_ticker[ticker] = {
            "ticker": ticker,
            "market_data_as_of": as_of.isoformat(),
            "last_price": round(100.0 + idx * 3.5, 4),
            "bid": round(99.8 + idx * 3.5, 4),
            "ask": round(100.2 + idx * 3.5, 4),
            "return_5m": round(0.0005 * (idx + 1), 6),
            "return_15m": round(0.001 * (idx + 1), 6),
            "return_1h": round(0.002 * (idx + 1), 6),
            "return_1d": round(0.004 * (idx + 1), 6),
            "volatility": round(0.012 + idx * 0.001, 6),
            "volume_context": "synthetic_read_only_sample",
            "imoex_relative_return": round(0.001 * idx, 6),
            "market_regime": "sample_neutral",
            "stale": False,
        }
    return {"as_of": as_of.isoformat(), "by_ticker": by_ticker}


def existing_market_context(
    features_path: Path,
    universe: Sequence[dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    """Read already-computed point-in-time features without opening target rows."""
    wanted = {str(row["ticker"]).upper() for row in universe}
    latest: dict[str, dict[str, Any]] = {}
    if features_path.exists():
        for row in _iter_jsonl(features_path):
            ticker = str(row.get("ticker", "")).upper()
            if ticker not in wanted:
                continue
            feature_as_of = str(row.get("feature_as_of", ""))
            previous = latest.get(ticker)
            if previous is None or feature_as_of > str(previous.get("feature_as_of", "")):
                latest[ticker] = row

    by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in sorted(wanted):
        row = latest.get(ticker)
        if row is None:
            by_ticker[ticker] = {
                "ticker": ticker,
                "market_data_as_of": None,
                "stale": True,
                "blocker": "EXISTING_FEATURE_CONTEXT_MISSING",
            }
            continue
        feature_values = cast("dict[str, Any]", row.get("features", {}))
        feature_date = _parse_datetime(str(row.get("feature_as_of", "")))
        stale = feature_date is None or as_of - feature_date > timedelta(days=1)
        by_ticker[ticker] = {
            "ticker": ticker,
            "market_data_as_of": None if feature_date is None else feature_date.isoformat(),
            "last_price": None,
            "bid": None,
            "ask": None,
            "return_5m": None,
            "return_15m": None,
            "return_1h": None,
            "return_1d": feature_values.get("return_1d"),
            "volatility": feature_values.get("volatility_20d"),
            "volume_context": {
                "volume_ratio_5d": feature_values.get("volume_ratio_5d"),
                "volume_ratio_20d": feature_values.get("volume_ratio_20d"),
            },
            "imoex_relative_return": feature_values.get("relative_return_1d"),
            "market_regime": None,
            "feature_source": str(features_path),
            "feature_row_id": row.get("row_id"),
            "stale": stale,
            "blocker": "EXISTING_FEATURE_CONTEXT_STALE" if stale else None,
        }
    return {
        "as_of": as_of.isoformat(),
        "source": str(features_path),
        "calculation": "EXISTING_FEATURE_ARTIFACT_ONLY",
        "by_ticker": by_ticker,
    }


def recent_event_context(
    live_root: Path,
    tickers: Sequence[str],
    as_of: datetime,
    config: AgentRunConfig,
) -> dict[str, Any]:
    ticker_set = {ticker.upper() for ticker in tickers}
    rows: list[dict[str, Any]] = []
    path = live_root / "live-shadow-corpus.jsonl"
    if path.exists():
        for row in _read_jsonl(path):
            ticker = str(row.get("ticker", "")).upper()
            if ticker not in ticker_set:
                continue
            published = _parse_datetime(str(row.get("published_at", "")))
            if published is None or published > as_of:
                continue
            rows.append(event_row(row, published))
    return {
        "events_as_of": as_of.isoformat(),
        "lookback_hours": 168,
        "events": sorted(rows, key=lambda item: item["published_at"], reverse=True)[
            : config.max_events_per_ticker * max(1, len(ticker_set))
        ],
    }


def event_row(row: dict[str, Any], published: datetime) -> dict[str, Any]:
    semantic = cast("dict[str, Any]", row.get("semantic_output", {}))
    return {
        "event_id": row.get("event_id"),
        "ticker": row.get("ticker"),
        "legal_issuer": row.get("issuer"),
        "published_at": published.isoformat(),
        "source": row.get("source_id"),
        "title": row.get("canonical_url"),
        "semantic_class": semantic.get("primary_event_type", "UNKNOWN"),
        "materiality": semantic.get("importance", "UNKNOWN"),
        "confidence": semantic.get("status", "UNKNOWN"),
        "exact_timestamp_evidence": row.get("timezone_contract"),
    }


def research_status(
    operation_root: Path,
    operational_proof_path: Path,
    as_of: datetime,
) -> dict[str, Any]:
    status = build_operation_status(operation_root)
    seal = verify_operation_seal(operation_root)
    last_publication = _parse_datetime(str(status.get("last_new_publication", "")))
    freshness_minutes = (
        None if last_publication is None else int((as_of - last_publication).total_seconds() // 60)
    )
    proof = _read_json(operational_proof_path)
    proof_valid = (
        proof.get("LIVE_RESEARCH_OPERATION_STATUS") == "READY"
        and proof.get("OPERATIONAL_BURN_IN") == "PASS"
        and proof.get("SOURCE_FAILURE_ISOLATION") is True
        and proof.get("SOURCE_FAILURE_ISOLATION_PROOF_LEVEL")
        in {"APPLICATION_PROOF", "REAL_BURNIN_PROOF"}
        and proof.get("BROKER_MUTATIONS") == 0
        and proof.get("LIVE_OUTCOMES_READ") == 0
        and proof.get("LIVE_TARGETS_COMPUTED") == 0
        and proof.get("LIVE_POST_EVENT_PRICE_READS") == 0
    )
    return {
        "research_status_as_of": as_of.isoformat(),
        "LIVE_RESEARCH_OPERATION_STATUS": status["LIVE_RESEARCH_OPERATION_STATUS"],
        "OPERATIONAL_BURN_IN": "PASS" if proof_valid else "PARTIAL",
        "SOURCE_FAILURE_ISOLATION": proof.get("SOURCE_FAILURE_ISOLATION")
        if proof_valid
        else "NOT_PROVEN",
        "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": proof.get("SOURCE_FAILURE_ISOLATION_PROOF_LEVEL")
        if proof_valid
        else "NOT_PROVEN",
        "operational_proof_path": str(operational_proof_path),
        "seal": seal,
        "event_freshness_minutes": freshness_minutes,
        "source_health": status.get("source_health", []),
        "data_freshness": {
            "last_successful_poll": status.get("last_successful_poll"),
            "last_new_publication": status.get("last_new_publication"),
        },
        "ML_V2_DATASET_STATUS": status.get("ML_V2_DATASET_STATUS")
        or "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
    }


def build_read_only_tool_registry(
    context: AgentDataContext, config: AgentRunConfig
) -> dict[str, ReadOnlyTool]:
    tools = {
        "get_portfolio_context": ReadOnlyTool(
            "get_portfolio_context",
            ToolCapability.READ_PORTFOLIO,
            NoArguments,
            lambda args: context.portfolio_snapshot,
        ),
        "get_recent_events": ReadOnlyTool(
            "get_recent_events",
            ToolCapability.READ_EVENTS,
            RecentEventsArguments,
            lambda args: tool_recent_events(args, context, config),
        ),
        "get_market_context": ReadOnlyTool(
            "get_market_context",
            ToolCapability.READ_MARKET,
            TickerArguments,
            lambda args: tool_market_context(args, context),
        ),
        "get_instrument_context": ReadOnlyTool(
            "get_instrument_context",
            ToolCapability.READ_MARKET,
            TickerArguments,
            lambda args: tool_instrument_context(args, context),
        ),
        "get_research_status": ReadOnlyTool(
            "get_research_status",
            ToolCapability.READ_SYSTEM,
            NoArguments,
            lambda args: context.research_status_snapshot,
        ),
    }
    forbidden = WRITE_TOOL_NAMES.intersection(tools)
    if forbidden:
        raise ValueError(f"WRITE_TOOL_REGISTERED:{','.join(sorted(forbidden))}")
    return tools


def tool_recent_events(
    args: dict[str, Any], context: AgentDataContext, config: AgentRunConfig
) -> dict[str, Any]:
    ticker = str(args.get("ticker", "")).upper()
    limit = min(int(args.get("limit", config.max_events_per_ticker)), config.max_events_per_ticker)
    lookback_hours = min(int(args.get("lookback_hours", 168)), 168)
    lower_bound = context.as_of - timedelta(hours=lookback_hours)
    events: list[dict[str, Any]] = []
    for row in cast("list[dict[str, Any]]", context.event_context_snapshot.get("events", [])):
        published = _parse_datetime(str(row.get("published_at", "")))
        if (
            row.get("ticker") == ticker
            and published is not None
            and lower_bound <= published <= context.as_of
        ):
            events.append(row)
    return {
        "ticker": ticker,
        "lookback_hours": lookback_hours,
        "limit": limit,
        "events": events[:limit],
    }


def tool_market_context(args: dict[str, Any], context: AgentDataContext) -> dict[str, Any]:
    ticker = str(args.get("ticker", "")).upper()
    by_ticker = cast(
        "dict[str, dict[str, Any]]", context.market_context_snapshot.get("by_ticker", {})
    )
    return by_ticker.get(
        ticker, {"ticker": ticker, "stale": True, "blocker": "MARKET_CONTEXT_MISSING"}
    )


def tool_instrument_context(args: dict[str, Any], context: AgentDataContext) -> dict[str, Any]:
    ticker = str(args.get("ticker", "")).upper()
    by_ticker = {str(row["ticker"]).upper(): row for row in context.allowed_universe}
    return by_ticker.get(
        ticker, {"ticker": ticker, "supported": False, "blocker": "TICKER_NOT_IN_UNIVERSE"}
    )


def call_tool(
    step: int, name: str, arguments: dict[str, Any], tools: dict[str, ReadOnlyTool]
) -> tuple[ToolCallRecord, dict[str, Any]]:
    if name not in tools:
        result = {"error": "TOOL_NOT_REGISTERED", "tool": name}
        return (
            ToolCallRecord(
                step=step,
                tool_name=name,
                capability=ToolCapability.READ_SYSTEM,
                arguments_hash=sha256_payload(arguments),
                result_hash=sha256_payload(result),
                status="ERROR",
                error="TOOL_NOT_REGISTERED",
            ),
            result,
        )
    tool = tools[name]
    try:
        validated_arguments = tool.arguments_model.model_validate(arguments).model_dump()
        result = tool.handler(validated_arguments)
        status = "OK"
        error = None
    except Exception as exc:
        result = {"error": type(exc).__name__, "message": str(exc)}
        status = "ERROR"
        error = type(exc).__name__
    return (
        ToolCallRecord(
            step=step,
            tool_name=name,
            capability=tool.capability,
            arguments_hash=sha256_payload(arguments),
            result_hash=sha256_payload(result),
            status=status,
            error=error,
        ),
        result,
    )


def validate_proposals(
    proposals: Sequence[TradeProposal], context: AgentDataContext, config: AgentRunConfig
) -> ProposalValidationResult:
    reasons: list[str] = []
    stale: list[str] = []
    allowed = {str(row["ticker"]).upper() for row in context.allowed_universe}
    supported = {
        str(row["ticker"]).upper()
        for row in context.allowed_universe
        if row.get("supported") is True and row.get("market_data_compatible") is True
    }
    seen_actions: dict[str, TradeAction] = {}
    for proposal in proposals:
        ticker = proposal.ticker.upper()
        if ticker not in allowed:
            reasons.append(f"UNSUPPORTED_TICKER:{ticker}")
        if ticker not in supported:
            reasons.append(f"UNSUPPORTED_INSTRUMENT:{ticker}")
        if proposal.target_weight > config.max_agent_proposed_weight:
            reasons.append(f"TARGET_WEIGHT_ABOVE_MAX:{ticker}")
        previous = seen_actions.get(ticker)
        if previous is not None and previous != proposal.action:
            reasons.append(f"CONFLICTING_DUPLICATE_PROPOSAL:{ticker}")
        seen_actions[ticker] = proposal.action
        market_row = tool_market_context({"ticker": ticker}, context)
        market_as_of = _parse_datetime(str(market_row.get("market_data_as_of", "")))
        if (
            market_row.get("stale") is True
            or market_as_of is None
            or context.as_of - market_as_of > timedelta(minutes=config.max_staleness_minutes)
        ):
            stale.append(ticker)
    if context.research_status_snapshot.get("LIVE_RESEARCH_OPERATION_STATUS") == "FAIL":
        for proposal in proposals:
            if proposal.action == TradeAction.BUY:
                reasons.append(f"BUY_WITH_CRITICAL_RESEARCH_FAIL:{proposal.ticker.upper()}")
    if reasons:
        return ProposalValidationResult(
            status=ProposalValidationStatus.INVALID, reasons=reasons, stale_tickers=stale
        )
    if stale:
        return ProposalValidationResult(
            status=ProposalValidationStatus.DEGRADED,
            reasons=["STALE_MARKET_CONTEXT"],
            stale_tickers=stale,
        )
    return ProposalValidationResult(status=ProposalValidationStatus.VALID, reasons=[])


def decision_status_from_validation(validation: ProposalValidationResult) -> AgentDecisionStatus:
    if validation.status == ProposalValidationStatus.VALID:
        return AgentDecisionStatus.VALID
    if validation.status == ProposalValidationStatus.DEGRADED:
        return AgentDecisionStatus.DEGRADED_STALE_DATA
    return AgentDecisionStatus.INVALID


def finalize_agent_result(
    run_id: str,
    created_at: datetime,
    config: AgentRunConfig,
    model: AgentModel,
    context: AgentDataContext,
    tool_calls: list[ToolCallRecord],
    raw_model_output: str | None,
    parsed: StructuredAgentOutput | None,
    proposals: list[TradeProposal],
    validation: ProposalValidationResult,
    decision_status: AgentDecisionStatus,
) -> AgentRunResult:
    result = AgentRunResult(
        run_id=run_id,
        created_at=created_at,
        code_sha=config.code_sha,
        agent_model_id=model.model_id,
        prompt_version=PROMPT_VERSION,
        AGENT_MODE=AgentMode.READ_ONLY_RESEARCH,
        AGENT_DECISION_STATUS=decision_status,
        AGENT_RESEARCH_CAPABILITY_READY=decision_status == AgentDecisionStatus.VALID,
        ML_V2_DATASET_STATUS=str(
            context.research_status_snapshot.get(
                "ML_V2_DATASET_STATUS", "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY"
            )
        ),
        universe=context.allowed_universe,
        portfolio_snapshot=context.portfolio_snapshot,
        market_context_snapshot=context.market_context_snapshot,
        event_context_snapshot=context.event_context_snapshot,
        research_status_snapshot=context.research_status_snapshot,
        tool_calls=tool_calls,
        raw_model_output=raw_model_output,
        parsed_structured_output=parsed,
        validation=validation,
        final_proposals=proposals,
        safety=SafetyCounters(),
    )
    write_agent_artifact(
        config.output_root,
        result,
        agent_policy(config, build_read_only_tool_registry(context, config)),
    )
    return result


def agent_policy(config: AgentRunConfig, tools: dict[str, ReadOnlyTool]) -> dict[str, Any]:
    return {
        "AGENT_MODE": AgentMode.READ_ONLY_RESEARCH.value,
        "PROMPT_VERSION": PROMPT_VERSION,
        "MAX_AGENT_STEPS": config.max_agent_steps,
        "MAX_TOOL_CALLS": config.max_tool_calls,
        "MAX_EVENTS_PER_TICKER": config.max_events_per_ticker,
        "MAX_TICKERS_PER_RUN": config.max_tickers_per_run,
        "MAX_AGENT_PROPOSED_WEIGHT": config.max_agent_proposed_weight,
        "EXTERNAL_WEB_SEARCH_ENABLED": False,
        "registered_tools": [
            {
                "name": tool.name,
                "capability": tool.capability.value,
                "arguments_schema": tool.arguments_model.model_json_schema(),
            }
            for tool in tools.values()
        ],
        "forbidden_tools": sorted(WRITE_TOOL_NAMES),
        "allowed_capabilities": sorted({tool.capability.value for tool in tools.values()}),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha": sha256_payload({"system_prompt": SYSTEM_PROMPT}),
    }


def write_agent_artifact(output_root: Path, result: AgentRunResult, policy: dict[str, Any]) -> None:
    payload = result.model_dump(mode="json")
    safety = result.safety.model_dump(mode="json")
    tool_hashes = [row.result_hash for row in result.tool_calls if row.result_hash]
    input_snapshot_hashes = {
        "allowed_universe": sha256_payload(result.universe),
        "portfolio_snapshot": sha256_payload(result.portfolio_snapshot),
        "market_context_snapshot": sha256_payload(result.market_context_snapshot),
        "event_context_snapshot": sha256_payload(result.event_context_snapshot),
        "research_status_snapshot": sha256_payload(result.research_status_snapshot),
    }
    manifest = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "run_id": result.run_id,
        "created_at": result.created_at.isoformat(),
        "code_sha": result.code_sha,
        "agent_model_id": result.agent_model_id,
        "prompt_version": result.prompt_version,
        "AGENT_MODE": result.AGENT_MODE.value,
        "AGENT_DECISION_STATUS": result.AGENT_DECISION_STATUS.value,
        "AGENT_RESEARCH_CAPABILITY_READY": result.AGENT_RESEARCH_CAPABILITY_READY,
        "ML_V2_DATASET_STATUS": result.ML_V2_DATASET_STATUS,
        "allowed_universe_size": len(result.universe),
        "tool_call_count": len(result.tool_calls),
        "tool_result_hashes": tool_hashes,
        "input_snapshot_hashes": input_snapshot_hashes,
        "proposal_count": len(result.final_proposals),
        "validation_status": result.validation.status.value,
        "input_snapshot_as_of": (
            None
            if result.parsed_structured_output is None
            else {
                key: value.isoformat()
                for key, value in result.parsed_structured_output.input_snapshot_as_of.items()
            }
        ),
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {key: value for key, value in manifest.items() if key != "ARTIFACT_SHA"}
    )
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "agent-policy.json", policy)
    _write_json(output_root / "allowed-universe.json", {"instruments": result.universe})
    _write_json(output_root / "sample-portfolio.json", result.portfolio_snapshot)
    _write_json(output_root / "sample-market-context.json", result.market_context_snapshot)
    _write_json(output_root / "sample-event-context.json", result.event_context_snapshot)
    _write_jsonl(
        output_root / "sample-tool-calls.jsonl",
        [row.model_dump(mode="json") for row in result.tool_calls],
    )
    _write_json(
        output_root / "sample-proposals.json",
        {
            "raw_model_output": result.raw_model_output,
            "parsed_structured_output": None
            if result.parsed_structured_output is None
            else result.parsed_structured_output.model_dump(mode="json"),
            "final_proposals": [row.model_dump(mode="json") for row in result.final_proposals],
        },
    )
    _write_json(output_root / "validation.json", result.validation.model_dump(mode="json"))
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest, result)
    _write_json(output_root / "run.json", payload)


def sample_agent_context(as_of: datetime | None = None) -> AgentDataContext:
    now = as_of or datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    universe = [
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
        }
        for ticker, issuer in (
            ("SBER", "ПАО Сбербанк"),
            ("GAZP", "ПАО Газпром"),
            ("ROSN", "Rosneft Oil Company"),
            ("YDEX", "Yandex"),
        )
    ]
    return AgentDataContext(
        as_of=now,
        allowed_universe=universe,
        portfolio_snapshot=synthetic_portfolio(universe, now),
        market_context_snapshot=synthetic_market_context(universe, now),
        event_context_snapshot={
            "events_as_of": now.isoformat(),
            "lookback_hours": 168,
            "events": [
                {
                    "event_id": "sample-sber-event",
                    "ticker": "SBER",
                    "legal_issuer": "ПАО Сбербанк",
                    "published_at": (now - timedelta(hours=2)).isoformat(),
                    "source": "sample-read-only",
                    "title": "Sample issuer publication",
                    "semantic_class": "FINANCIAL_RESULTS",
                    "materiality": "MEDIUM",
                    "confidence": "COMPLETE",
                    "exact_timestamp_evidence": "fixture exact timestamp",
                }
            ],
        },
        research_status_snapshot={
            "research_status_as_of": now.isoformat(),
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
            "seal": {"sealed_epoch_verified": True, "violations": 0},
            "event_freshness_minutes": 120,
            "source_health": [],
            "data_freshness": {
                "last_successful_poll": now.isoformat(),
                "last_new_publication": now.isoformat(),
            },
            "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        },
    )


def sample_fake_agent_model(as_of: datetime | None = None) -> FakeAgentModel:
    now = as_of or datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    final = {
        "as_of": now.isoformat(),
        "input_snapshot_as_of": {
            "portfolio_as_of": now.isoformat(),
            "market_data_as_of": now.isoformat(),
            "events_as_of": now.isoformat(),
        },
        "proposals": [
            _proposal("SBER", "BUY", 0.68, 0.10, "sample-sber-event"),
            _proposal("GAZP", "HOLD", 0.55, 0.05, "sample-market-context"),
            _proposal("ROSN", "AVOID", 0.61, 0.0, "sample-risk"),
        ],
    }
    return FakeAgentModel(
        [
            AgentModelResponse(
                tool_calls=[
                    ToolCallRequest(name="get_portfolio_context"),
                    ToolCallRequest(name="get_research_status"),
                    ToolCallRequest(name="get_market_context", arguments={"ticker": "SBER"}),
                    ToolCallRequest(
                        name="get_recent_events",
                        arguments={"ticker": "SBER", "lookback_hours": 168, "limit": 3},
                    ),
                ]
            ),
            AgentModelResponse(final_output=json.dumps(final, ensure_ascii=False)),
        ],
        model_id="fake-agent-research-v1",
    )


def _proposal(
    ticker: str, action: str, confidence: float, weight: float, evidence_id: str
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": confidence,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": [f"{ticker} read-only research proposal from deterministic fake model."],
        "risks": ["Synthetic sample; no performance claim and no execution semantics."],
        "evidence": [{"type": "EVENT", "id": evidence_id}],
        "data_quality": "GOOD",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield cast("dict[str, Any]", value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any], result: AgentRunResult) -> None:
    proposal_lines = [
        f"- {proposal.ticker}: {proposal.action.value} target_weight={proposal.target_weight}"
        for proposal in result.final_proposals
    ]
    lines = [
        "# AI trading agent v1",
        "",
        f"- run_id: {manifest['run_id']}",
        f"- code_sha: {manifest['code_sha']}",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- AGENT_MODE: {manifest['AGENT_MODE']}",
        f"- AGENT_RESEARCH_CAPABILITY_READY: {manifest['AGENT_RESEARCH_CAPABILITY_READY']}",
        f"- AGENT_DECISION_STATUS: {manifest['AGENT_DECISION_STATUS']}",
        f"- ML_V2_DATASET_STATUS: {manifest['ML_V2_DATASET_STATUS']}",
        f"- registered_tool_calls: {manifest['tool_call_count']}",
        f"- validation_status: {manifest['validation_status']}",
        "",
        "## Sample proposals",
        *proposal_lines,
        "",
        (
            "No broker, order, portfolio, outcome, target, holdout, training, "
            "or backtest mutation occurred."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
