from __future__ import annotations

from src.ai_trading_agent_v1.application import AgentRunConfig, UnconfiguredAgentModel
from src.production_readonly_adapters_v1.context import ProductionPaperOperationContextProvider
from src.production_readonly_adapters_v1.moex import MoexIssFreshMarketAdapter
from src.production_readonly_adapters_v1.ollama import OllamaAgentModel
from src.risk_engine_paper_v1.domain import RiskPolicy
from src.shared.config.settings import Settings


def create_production_agent_model(settings: Settings) -> OllamaAgentModel | UnconfiguredAgentModel:
    if settings.ai_provider.strip().lower() != "ollama":
        return UnconfiguredAgentModel()
    return OllamaAgentModel(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        think=settings.ollama_think,
        timeout_seconds=settings.ai_request_timeout_seconds,
        max_retries=settings.ai_max_retries,
        max_output_tokens=settings.ai_max_output_tokens,
        random_seed=settings.ai_random_seed,
        context_length=settings.ollama_context_length,
    )


def create_fresh_market_adapter(
    settings: Settings,
    risk_policy: RiskPolicy,
) -> MoexIssFreshMarketAdapter:
    effective_age = min(
        settings.market_context_max_age_seconds,
        risk_policy.max_stale_market_age.total_seconds(),
    )
    return MoexIssFreshMarketAdapter(
        base_url=settings.moex_iss_base_url,
        timeout_seconds=settings.moex_http_timeout_seconds,
        max_retries=settings.moex_http_max_retries,
        user_agent=settings.moex_http_user_agent,
        max_age_seconds=effective_age,
    )


def create_production_context_provider(
    settings: Settings,
    agent_config: AgentRunConfig,
    risk_policy: RiskPolicy,
) -> ProductionPaperOperationContextProvider:
    return ProductionPaperOperationContextProvider(
        agent_config=agent_config,
        market_adapter=create_fresh_market_adapter(settings, risk_policy),
        risk_policy=risk_policy,
        configured_max_age_seconds=settings.market_context_max_age_seconds,
    )
