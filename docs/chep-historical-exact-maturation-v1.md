# chep-historical-exact-maturation-v1

This data-maturation path consumes only the CHEP rows from
`artifacts/exact-event-live-official-collection-v1/`, verifies the collector
artifact hashes, splits the rows at `FUTURE_EVENT_HOLDOUT_START=2026-08-11`,
and persists cohort hashes before market-history access.

Historical CHEP rows are passed through the existing frozen `EXACT_INTRADAY`
alignment flow using T-Invest read-only minute candles and existing IMOEX
benchmark cache. Future CHEP rows stay metadata-only and are blocked before
price lookup, reaction construction, target construction, or feature
maturation.

The one-shot command is:

```powershell
uv run python -m apps.cli.mature_chep_historical_exact --base-main-sha <BASE_MAIN_SHA>
```

Add `--live-readonly` only when the environment has the existing
`TINVEST_READONLY_TOKEN` and `SSL_TBANK_VERIFY=true`; the client has no order,
account mutation, sandbox order, money movement, or broker write surface.

Required safety invariants:

- DATA_MATURATION_ONLY=true
- MODEL_TRAINING_PERFORMED=false
- TEST_OUTCOME_USED=false
- TEST_EVALUATION_PERFORMED=false
- FUTURE_EVENT_HOLDOUT_USED=false
- FUTURE_EVENT_HOLDOUT_OBSERVED=false
- REAL_TRADING_ALLOWED=false
- REAL_ORDER_SUBMISSION_ALLOWED=false
- SANDBOX_ORDER_SUBMISSION_ALLOWED=false
- RULES_V3_CHANGED=false
- QWEN_CHANGED=false
- NLP_TUNING_PERFORMED=false
- STRICT_EXACT_METHODOLOGY_CHANGED=false
