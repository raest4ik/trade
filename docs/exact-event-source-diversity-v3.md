# exact-event-source-diversity-v3

`exact-event-source-diversity-v3` is a data-acquisition-only artifact that extends the
merged warmup-recovered EXACT corpus with additional official-source metadata records.

The v3 builder starts from `artifacts/exact-event-market-history-warmup-recovery-v1`, verifies
the recovered dataset SHA, preserves every existing event and feature row as a prefix, and then
adds deterministic metadata-only records discovered from the official MOEX issuer RSS cache.

## Scope

- source universe: current T-Invest TQBR RUB share universe, excluding IMOEX benchmark
- new transport: official MOEX RSS with explicit `pubDate` timezone offsets
- source policy: zero-cost official public sources only
- acquisition bounds: single cached RSS feed, deterministic ticker extraction, fixed max rows
- future holdout: publication dates on or after 2026-08-11 remain metadata-only

## Safety

- MODEL_TRAINING_PERFORMED=false
- TEST_OUTCOME_USED=false
- FUTURE_EVENT_HOLDOUT_USED=false
- FUTURE_EVENT_HOLDOUT_OBSERVED=false
- RULES_V3_CHANGED=false
- QWEN_CHANGED=false
- NLP_TUNING_PERFORMED=false
- CONFIRMED_SIGNAL=false
- BACKTEST_APPROVED=false
- PAPER_TRADING_APPROVED=false
- REAL_TRADING_APPROVED=false

No model training, TEST outcome use, future holdout outcome observation, backtest, paper trading,
orders, or BUY/SELL output is performed.

## Preservation

The artifact reports:

- EXACT_V2_PRESERVED=YES
- EXISTING_EVENT_ROWS_PRESERVED=PASS
- EXISTING_FEATURE_ROWS_PRESERVED=PASS
- LEAKAGE_CHECK=PASS
- DUPLICATE_RECONCILIATION=PASS

Historical v2/warmup rows are not rewritten. New rows do not make date-only timestamps exact by
coercion; only source-native exact RSS timestamps with explicit offsets are accepted.
