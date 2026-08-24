# EXACT Event Official Source Discovery v5

This PR is data-acquisition-only. It searches for new official zero-cost public source
mechanisms for underrepresented Russian issuers, while keeping `EXACT_INTRADAY` strict and
unchanged.

The discovery priority is metadata-only:

- `A_ZERO_FEATURE_READY`: tickers already in the exact corpus with zero feature-ready rows
- `B_EXACT_1_5`: tickers with 1 to 5 exact rows
- `C_EXACT_6_20`: tickers with 6 to 20 exact rows
- `D_CANONICAL_TQBR_NOT_IN_EXACT`: canonical TQBR RUB tickers absent from the exact corpus
- `DEPRIORITIZED`: dominant or already well-covered cohorts, including MGNT, T, and X5

Discovery is bounded by `MAX_TICKERS`, `MAX_URLS_PER_TICKER`, `MAX_REQUESTS_PER_DOMAIN`,
`MAX_PAGES_PER_SOURCE`, and `MAX_ITEMS_PER_SOURCE`. CI uses self-contained source discovery cache
fixtures; production runs without an available source snapshot fail closed instead of guessing
official URLs.

Accepted exact source mechanisms require official confirmation, public zero-cost access, HTTPS
source/domain consistency, and timestamp evidence with date, time, and timezone. Date-only values
are rejected, fetch time is never used as publication time, and ambiguous source identity fails
closed.

Safety invariants:

- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `TEST_EVALUATION_PERFORMED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `DATE_ONLY_COERCIONS=0`
- `FETCH_TIME_USED_AS_PUBLICATION_TIME=false`
- `STRICT_EXACT_METHODOLOGY_CHANGED=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- T-Invest read-only is the only allowed market-history maturation provider for new historical
  events
- no MOEX substitution
- no forward-fill
- no sparse label family
- no backtest, paper trading, orders, BUY/SELL, predictions, or signals

