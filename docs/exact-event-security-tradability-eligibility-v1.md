# exact-event-security-tradability-eligibility-v1

Reusable pre-maturation data-quality gate for strict-EXACT events.

Pipeline position:

`OFFICIAL_EVENT -> EXACT_TIMESTAMP -> CANONICAL_INSTRUMENT_MAPPING -> SECURITY_TRADABILITY_ELIGIBILITY -> MARKET_HISTORY_ONLY_IF_ELIGIBLE`

The gate keeps event validity separate from market-reaction eligibility. CHEP 2026 official
events remain `VALID_EXACT_EVENT`, but their market eligibility is
`SECURITY_NOT_TRADING_AT_EVENT_TIME` based on positive diagnostic evidence from the CHEP security
history diagnostics artifact.

Safety invariants:

- MODEL_TRAINING_PERFORMED=false
- TEST_OUTCOME_USED=false
- TEST_EVALUATION_PERFORMED=false
- FUTURE_EVENT_HOLDOUT_USED=false
- FUTURE_EVENT_HOLDOUT_OBSERVED=false
- FUTURE_CHEP_PRICE_LOOKUPS=0
- FUTURE_CHEP_REACTION_ATTEMPTS=0
- FUTURE_CHEP_TARGET_ATTEMPTS=0
- REAL_TRADING_ALLOWED=false
- REAL_ORDER_SUBMISSION_ALLOWED=false
- BROKER_ACCOUNT_MUTATION_ALLOWED=false
- SANDBOX_ORDER_SUBMISSION_ALLOWED=false
- DATA_COST_RUB=0

Empty candle responses alone never prove non-trading. They map to
`SECURITY_HISTORY_UNAVAILABLE` until independent exchange/security-history evidence establishes
`SECURITY_NOT_TRADING_AT_EVENT_TIME`.
