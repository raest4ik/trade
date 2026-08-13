from __future__ import annotations

from typing import Final

SOURCE: Final = "TINVEST_API"
SOURCE_COST: Final = "ZERO_RUB"
SOURCE_USAGE_SCOPE: Final = "PRIVATE_CLIENT_INTERNAL_USE"
SOURCE_USAGE_READINESS: Final = "PRIVATE_INTERNAL_USE_CONFIRMED"
PUBLIC_REDISTRIBUTION_ALLOWED: Final = False
PRIVATE_DATASET_BUILD_ALLOWED: Final = True
PRIVATE_MODEL_TRAINING_ALLOWED: Final = True
PRIVATE_RESEARCH_BACKTEST_ALLOWED: Final = True
PRICE_ADJUSTMENT_STATUS: Final = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"

REAL_TRADING_ALLOWED: Final = False
REAL_ORDER_SUBMISSION_ALLOWED: Final = False
REAL_STOP_ORDER_ALLOWED: Final = False
REAL_MONEY_MOVEMENT_ALLOWED: Final = False
BROKER_ACCOUNT_MUTATION_ALLOWED: Final = False
MARGIN_TRADING_ALLOWED: Final = False
LIVE_EXECUTION_ALLOWED: Final = False
SANDBOX_ORDER_SUBMISSION_ALLOWED: Final = False

MODEL_TRAINED: Final = False
STRATEGY_BACKTEST_EXECUTED: Final = False
PAPER_TRADING_EXECUTED: Final = False
BUY_SELL_SIGNALS_GENERATED: Final = False
PAID_SERVICES_USED: Final = False

POLICY_EVIDENCE: Final = (
    {
        "url": "https://developer.tbank.ru/invest/intro/intro/",
        "accessed_at": "2026-08-13",
        "summary": (
            "T-Invest states API data is free and supports market-data access and client-built "
            "checks of algorithms on historical data."
        ),
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/intro/token",
        "accessed_at": "2026-08-13",
        "summary": (
            "Read-only tokens can access schedules, quotes, and historical data and cannot "
            "submit trading instructions; sandbox tokens are isolated from ordinary methods."
        ),
    },
    {
        "url": "https://www.tbank.ru/invest/disclaimers/basic-information/",
        "accessed_at": "2026-08-13",
        "summary": (
            "Clients may use, store, and process exchange information, while onward/public "
            "distribution requires exchange consent."
        ),
    },
    {
        "url": "https://developer.tbank.ru/invest/api/market-data-service-get-candles",
        "accessed_at": "2026-08-13",
        "summary": "GetCandles documents daily history and bounded request intervals.",
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/intro/limits",
        "accessed_at": "2026-08-13",
        "summary": "Official unary request-rate limits apply to instruments and market data.",
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/useful-info/faq_corp_action",
        "accessed_at": "2026-08-13",
        "summary": (
            "Splits and reverse splits affect historical quotations; conversions can change "
            "ISIN, FIGI, ticker, nominal, currency, and price."
        ),
    },
    {
        "url": "https://developer.tbank.ru/invest/services/instruments/head-instruments",
        "accessed_at": "2026-08-13",
        "summary": (
            "Indicatives exposes indices including IMOEX and its UID can be used with "
            "GetCandles; instrument identity changes must not be guessed."
        ),
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/intro/load_history",
        "accessed_at": "2026-08-13",
        "summary": (
            "Historical depth varies by instrument; first_1day_candle_date provides the "
            "available start and daily requests are bounded to approximately six years."
        ),
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/developer/sandbox",
        "accessed_at": "2026-08-13",
        "summary": "The sandbox is an isolated endpoint; this project performs connectivity only.",
    },
    {
        "url": "https://developer.tbank.ru/invest/intro/developer/network",
        "accessed_at": "2026-08-13",
        "summary": (
            "Python SDK 1.49.2 or newer supports SSL_TBANK_VERIFY=True and validates "
            "T-Invest TLS with its bundled Russian trusted root certificate."
        ),
    },
)


def source_policy() -> dict[str, object]:
    return {
        "source": SOURCE,
        "source_cost": SOURCE_COST,
        "source_usage_scope": SOURCE_USAGE_SCOPE,
        "source_usage_readiness": SOURCE_USAGE_READINESS,
        "public_redistribution_allowed": PUBLIC_REDISTRIBUTION_ALLOWED,
        "private_dataset_build_allowed": PRIVATE_DATASET_BUILD_ALLOWED,
        "private_model_training_allowed": PRIVATE_MODEL_TRAINING_ALLOWED,
        "private_research_backtest_allowed": PRIVATE_RESEARCH_BACKTEST_ALLOWED,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "policy_evidence": list(POLICY_EVIDENCE),
    }


def execution_safety() -> dict[str, bool]:
    return {
        "real_trading_allowed": REAL_TRADING_ALLOWED,
        "real_order_submission_allowed": REAL_ORDER_SUBMISSION_ALLOWED,
        "real_stop_order_allowed": REAL_STOP_ORDER_ALLOWED,
        "real_money_movement_allowed": REAL_MONEY_MOVEMENT_ALLOWED,
        "broker_account_mutation_allowed": BROKER_ACCOUNT_MUTATION_ALLOWED,
        "margin_trading_allowed": MARGIN_TRADING_ALLOWED,
        "live_execution_allowed": LIVE_EXECUTION_ALLOWED,
        "sandbox_order_submission_allowed": SANDBOX_ORDER_SUBMISSION_ALLOWED,
    }
