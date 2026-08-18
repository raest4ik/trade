# exact-event-data-diagnostics-v1

This PR adds a diagnostic-only artifact builder for the frozen EXACT event corpus after
`exact-event-predictive-baseline-v1`.

The builder reads `artifacts/exact-event-market-dataset-v2` and the locked baseline split manifest,
then writes immutable diagnostics to `artifacts/exact-event-data-diagnostics-v1`.

Required scope and safety labels:

- DIAGNOSTIC_ONLY=true
- MODEL_TRAINING_PERFORMED=false
- TEST_OUTCOME_USED=false
- FUTURE_EVENT_HOLDOUT_USED=false
- FUTURE_EVENT_HOLDOUT_OBSERVED=false
- RULES_V3_CHANGED=false
- QWEN_CHANGED=false
- NLP_TUNING_PERFORMED=false
- BACKTEST_APPROVED=false
- PAPER_TRADING_APPROVED=false
- REAL_TRADING_APPROVED=false
- CONFIRMED_SIGNAL=false
- PRICE_ADJUSTMENT_STATUS=UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES

Outcome/target diagnostics are restricted to TRAIN+VALIDATION event ids from the locked
`15m-split-manifest.json`. TEST rows are counted as metadata-only. Future holdout events with
`publication_date >= 2026-08-11` remain metadata-only and must have no outcome fields exported.

The diagnostics include:

- eligibility funnel with reconciliation
- warmup loss analysis
- source, issuer, and ticker concentration
- event type coverage, UNKNOWN share, and entropy
- duplicate and cluster diagnostics
- timestamp metadata quality
- TRAIN+VALIDATION target geometry for 1m, 5m, 15m, 30m, and 60m horizons
- fail-closed EXACT-vs-DATE pairing status
- target-free feature quality checks
- temporal coverage
- prioritized data improvement report

Run:

```powershell
uv run python -m apps.cli.build_exact_event_data_diagnostics
```

The expected dataset SHA is:

```text
20ab67ff4d94c59d6cf714f8b2f7c048031bda120bbd92ceb6e6185a838e14c3
```

No new model, A/B/C evaluation, TEST evaluation, future holdout outcome read, backtest, paper
trading, order submission, or BUY/SELL signal is part of this artifact.
