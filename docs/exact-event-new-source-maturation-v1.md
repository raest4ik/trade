# EXACT event new source maturation v1

This PR adds a deterministic, data-maturation-only audit for the five EXACT events introduced by
`exact-event-source-diversity-v3` relative to
`exact-event-market-history-warmup-recovery-v1`.

The cohort is derived from the dataset diff, not from a hardcoded ticker list. Future holdout
events with publication dates on or after `2026-08-11` remain metadata-only and never enter the
market alignment path. Historical events may mature only when local T-Invest minute cache contains
both security and IMOEX benchmark history, strict pre-event feature timestamps are before the
publication timestamp, and the frozen EXACT alignment can build all existing horizons.

The artifact is written to `artifacts/exact-event-new-source-maturation-v1` and includes
`events.jsonl`, `features.jsonl`, `targets.jsonl`, `per-event-status.jsonl`, `manifest.json`, and
`report.md`. Existing event rows outside the PR35 cohort and all existing feature rows are checked
for exact preservation.

Safety boundaries:

- `RESEARCH_ONLY=true`
- `DATA_MATURATION_ONLY=true`
- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `CONFIRMED_SIGNAL=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`
- no MOEX substitution
- no forward-fill
- no source expansion
- no BUY/SELL, orders, portfolio simulation, or backtest
