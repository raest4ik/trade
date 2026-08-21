# EXACT Event Source Depth Expansion v4

This PR is data-acquisition-only. It keeps `EXACT_INTRADAY` strict, does not create a sparse label
family, and does not run any model or TEST evaluation.

The source-depth priority is pre-registered and outcome-free:

- `TIER_1`: current exact events <= 5
- `TIER_2`: current exact events 6..20
- `TIER_3`: current exact events 21..50
- `DEPRIORITIZED`: current exact events > 50
- tie-break: ticker ascending

Archive expansion is bounded by `MAX_SOURCES_PER_RUN`, `MAX_PAGES_PER_SOURCE`, and
`MAX_ITEMS_PER_SOURCE`. Official archives that are missing, date-only, policy-blocked, empty, or
duplicate-only are recorded with normalized blockers instead of being forced into success.

Accepted new EXACT events require official source timestamps with date, time, and explicit timezone
or offset. Date-only items are never coerced, and fetch time is never used as publication time.

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
- `DATA_COST_RUB=0`
- T-Invest read-only is the only allowed market-history maturation provider for new historical
  events
- no MOEX substitution
- no forward-fill
- no backtest, paper trading, orders, BUY/SELL, predictions, or signals
