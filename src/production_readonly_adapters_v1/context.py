from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from src.ai_trading_agent_v1.application import (
    AgentRunConfig,
    build_allowed_universe,
    recent_event_context,
    research_status,
)
from src.paper_trading_operation_v1.application import PaperOperationContext
from src.paper_trading_operation_v1.domain import PaperOperationPolicy
from src.production_readonly_adapters_v1.domain import FreshMarketSnapshot
from src.risk_engine_paper_v1.domain import MarketQuote, PaperPortfolio, RiskPolicy


class FreshMarketAdapter(Protocol):
    adapter_id: str

    def fetch(
        self,
        *,
        universe: list[dict[str, Any]],
        operation_as_of: datetime,
    ) -> FreshMarketSnapshot: ...


@dataclass(frozen=True, slots=True)
class ProductionPaperOperationContextProvider:
    agent_config: AgentRunConfig
    market_adapter: FreshMarketAdapter
    risk_policy: RiskPolicy
    configured_max_age_seconds: float
    universe_loader: Callable[[AgentRunConfig], list[dict[str, Any]]] = build_allowed_universe
    event_loader: Callable[..., dict[str, Any]] = recent_event_context
    research_loader: Callable[..., dict[str, Any]] = research_status

    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext:
        canonical = self.universe_loader(self.agent_config)
        by_ticker = {str(row["ticker"]).upper(): row for row in canonical}
        held = [position.ticker.upper() for position in portfolio.positions]
        missing_held = [ticker for ticker in held if ticker not in by_ticker]
        if missing_held:
            raise ValueError(f"HELD_POSITION_OUTSIDE_CANONICAL_UNIVERSE:{missing_held[0]}")
        candidates = [ticker for ticker in sorted(by_ticker) if ticker not in held]
        remaining = max(policy.max_operation_universe - len(held), 0)
        selected = [*held, *candidates[:remaining]]
        universe = [by_ticker[ticker] for ticker in selected]
        effective_age = min(
            self.configured_max_age_seconds,
            self.risk_policy.max_stale_market_age.total_seconds(),
        )
        if effective_age <= 0:
            raise ValueError("MARKET_CONTEXT_MAX_AGE_INVALID")
        snapshot = self.market_adapter.fetch(
            universe=universe,
            operation_as_of=operation_as_of,
        )
        if snapshot.effective_max_age_seconds > effective_age:
            raise ValueError("MARKET_ADAPTER_FRESHNESS_POLICY_TOO_WEAK")
        market_context = {
            "as_of": operation_as_of.isoformat(),
            **snapshot.audit_payload(),
            "by_ticker": {str(row["ticker"]): row for row in snapshot.quotes},
        }
        quotes = [
            MarketQuote(
                ticker=str(row["ticker"]),
                as_of=datetime.fromisoformat(str(row["market_data_as_of"])),
                last_price=float(row["last_price"]),
                bid=None if row["bid"] is None else float(row["bid"]),
                ask=None if row["ask"] is None else float(row["ask"]),
                lot_size=int(row["lot_size"]),
                supported=True,
            )
            for row in snapshot.quotes
        ]
        return PaperOperationContext(
            universe=universe,
            market_quotes=quotes,
            market_context=market_context,
            event_context=self.event_loader(
                self.agent_config.live_root,
                selected,
                operation_as_of,
                self.agent_config,
            ),
            research_status=self.research_loader(
                self.agent_config.operation_root,
                self.agent_config.operational_proof_path,
                operation_as_of,
            ),
        )


def effective_market_max_age(
    configured_seconds: float,
    risk_policy: RiskPolicy,
) -> timedelta:
    return min(timedelta(seconds=configured_seconds), risk_policy.max_stale_market_age)
