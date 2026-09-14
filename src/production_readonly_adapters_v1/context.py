from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from src.ai_trading_agent_v1.application import (
    AgentRunConfig,
    build_allowed_universe,
    recent_event_context,
    research_status,
)
from src.current_moex_tradability_v1.domain import (
    CandidateClassification,
    CurrentUniverseResolution,
)
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import (
    PaperOperationContext,
    PreparedPaperOperationContext,
)
from src.paper_trading_operation_v1.domain import PaperOperationPolicy
from src.production_readonly_adapters_v1.domain import FreshMarketSnapshot, RawMarketSnapshot
from src.risk_engine_paper_v1.domain import MarketQuote, PaperPortfolio, RiskPolicy


class FreshMarketAdapter(Protocol):
    adapter_id: str

    def fetch(
        self,
        *,
        universe: list[dict[str, Any]],
        operation_as_of: datetime,
    ) -> FreshMarketSnapshot: ...

    def fetch_raw(self, *, universe: list[dict[str, Any]]) -> RawMarketSnapshot: ...

    def validate_snapshot(
        self,
        snapshot: RawMarketSnapshot,
        *,
        decision_as_of: datetime,
    ) -> FreshMarketSnapshot: ...


class CurrentTradabilityResolver(Protocol):
    def resolve(self, canonical: Sequence[dict[str, Any]]) -> CurrentUniverseResolution: ...


@dataclass(frozen=True, slots=True)
class ProductionPaperOperationContextProvider:
    agent_config: AgentRunConfig
    market_adapter: FreshMarketAdapter
    risk_policy: RiskPolicy
    configured_max_age_seconds: float
    universe_loader: Callable[[AgentRunConfig], list[dict[str, Any]]] = build_allowed_universe
    event_loader: Callable[..., dict[str, Any]] = recent_event_context
    research_loader: Callable[..., dict[str, Any]] = research_status
    tradability_resolver: CurrentTradabilityResolver | None = None

    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext:
        universe, selected, eligibility = self._selected_universe(portfolio, policy)
        snapshot = self.market_adapter.fetch(
            universe=universe,
            operation_as_of=operation_as_of,
        )
        return self._context(
            cycle_started_at=operation_as_of,
            decision_as_of=operation_as_of,
            universe=universe,
            selected=selected,
            snapshot=snapshot,
            eligibility=eligibility,
        )

    def load_after_market_fetch(
        self,
        *,
        cycle_started_at: datetime,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> PreparedPaperOperationContext:
        universe, selected, eligibility = self._selected_universe(portfolio, policy)
        raw_snapshot = self.market_adapter.fetch_raw(universe=universe)
        decision_as_of = raw_snapshot.market_fetch_completed_at
        if decision_as_of < cycle_started_at:
            raise ValueError("DECISION_CUTOFF_BEFORE_CYCLE_START")
        snapshot = self.market_adapter.validate_snapshot(
            raw_snapshot,
            decision_as_of=decision_as_of,
        )
        context = self._context(
            cycle_started_at=cycle_started_at,
            decision_as_of=decision_as_of,
            universe=universe,
            selected=selected,
            snapshot=snapshot,
            eligibility=eligibility,
        )
        return PreparedPaperOperationContext(
            cycle_started_at=cycle_started_at,
            decision_as_of=decision_as_of,
            context=context,
        )

    def _selected_universe(
        self,
        portfolio: PaperPortfolio,
        policy: PaperOperationPolicy,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, object]]:
        canonical = self.universe_loader(self.agent_config)
        by_ticker = {str(row["ticker"]).upper(): row for row in canonical}
        held = [position.ticker.upper() for position in portfolio.positions]
        for ticker in held:
            by_ticker.setdefault(
                ticker,
                {
                    "ticker": ticker,
                    "board": "TQBR",
                    "canonical_status": "HELD_OUTSIDE_CANONICAL",
                    "supported": True,
                    "market_data_compatible": True,
                    "feature_compatible": False,
                },
            )
        canonical_sha = sha256_payload(canonical)
        if self.tradability_resolver is None:
            eligible = set(by_ticker) - set(held)
            classifications: dict[str, CandidateClassification] = {}
            eligibility: dict[str, object] = {
                "current_eligibility_policy_version": "TEST_OR_LEGACY_BYPASS",
                "canonical_universe_sha": canonical_sha,
                "eligible_count": len(eligible),
                "ineligible_count": 0,
                "unknown_count": 0,
                "rejected_candidates": [],
            }
        else:
            resolution = self.tradability_resolver.resolve(canonical)
            classifications = resolution.by_ticker()
            eligible = {
                ticker
                for ticker, classification in classifications.items()
                if classification.candidate_eligible
            }
            eligibility = {
                **resolution.audit_payload(),
                "canonical_universe_sha": canonical_sha,
                "eligible_universe_sha": sha256_payload(sorted(eligible)),
            }
        candidates = [
            ticker for ticker in sorted(by_ticker) if ticker not in held and ticker in eligible
        ]
        remaining = max(policy.max_operation_universe - len(held), 0)
        selected = [*held, *candidates[:remaining]]
        selected_rows: list[dict[str, Any]] = []
        for ticker in selected:
            row = dict(by_ticker[ticker])
            classification = classifications.get(ticker)
            if classification is not None:
                row["current_tradability_status"] = classification.current_moex_status.value
                row["current_tradability_reason"] = classification.reason
            elif ticker in held:
                row["current_tradability_status"] = "UNKNOWN"
                row["current_tradability_reason"] = "HELD_POSITION_VISIBILITY_OVERRIDE"
            selected_rows.append(row)
        eligibility["selected_tickers"] = selected
        eligibility["selected_universe_sha"] = sha256_payload(selected_rows)
        return selected_rows, selected, eligibility

    def _context(
        self,
        *,
        cycle_started_at: datetime,
        decision_as_of: datetime,
        universe: list[dict[str, Any]],
        selected: list[str],
        snapshot: FreshMarketSnapshot,
        eligibility: dict[str, object],
    ) -> PaperOperationContext:
        effective_age = min(
            self.configured_max_age_seconds,
            self.risk_policy.max_stale_market_age.total_seconds(),
        )
        if effective_age <= 0:
            raise ValueError("MARKET_CONTEXT_MAX_AGE_INVALID")
        if snapshot.effective_max_age_seconds > effective_age:
            raise ValueError("MARKET_ADAPTER_FRESHNESS_POLICY_TOO_WEAK")
        for field in (
            "current_eligibility_fetched_at",
            "current_eligibility_source_time",
        ):
            eligibility_time = eligibility.get(field)
            if (
                isinstance(eligibility_time, str)
                and datetime.fromisoformat(eligibility_time) > decision_as_of
            ):
                raise ValueError("CURRENT_UNIVERSE_SNAPSHOT_AFTER_DECISION_CUTOFF")
        market_context = {
            "as_of": decision_as_of.isoformat(),
            "cycle_started_at": cycle_started_at.isoformat(),
            **snapshot.audit_payload(),
            **eligibility,
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
                decision_as_of,
                self.agent_config,
            ),
            research_status=self.research_loader(
                self.agent_config.operation_root,
                self.agent_config.operational_proof_path,
                decision_as_of,
            ),
        )


def effective_market_max_age(
    configured_seconds: float,
    risk_policy: RiskPolicy,
) -> timedelta:
    return min(timedelta(seconds=configured_seconds), risk_policy.max_stale_market_age)
