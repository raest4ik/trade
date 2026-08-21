# EXACT event security history diagnostics v1

This PR adds a diagnostics-only layer for the historical PR36 EXACT events blocked by
`MARKET_HISTORY_MISSING`: GEMC, BTBR, and INCB.

The diagnostic cohort is derived from the PR36 artifact, then filtered to historical
`MARKET_HISTORY_MISSING` rows. Future metadata-only events are written only to the future exclusion
audit and never enter market history probes.

The artifact is written to `artifacts/exact-event-security-history-diagnostics-v1` and contains
`manifest.json`, `per-event-diagnostics.jsonl`, `future-holdout-exclusions.jsonl`, and `report.md`.
It does not mutate the PR36 production dataset, so `OUTPUT_DATASET_SHA` is intentionally unchanged.

Safety boundaries:

- `RESEARCH_ONLY=true`
- `DIAGNOSTICS_ONLY=true`
- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `MOEX_SUBSTITUTION_USED=false`
- `FORWARD_FILL_USED=false`
- `SYNTHETIC_MARKET_DATA_USED=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`
- no broker write APIs
- no BUY/SELL, orders, portfolio simulation, or backtest
