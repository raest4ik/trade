# exact-event-market-history-warmup-recovery-v1

This PR adds a data-infrastructure recovery artifact for EXACT event market-history warmup
losses identified by `exact-event-data-diagnostics-v1`.

The recovery preserves the frozen EXACT feature methodology:

- pre-event market context feature names are unchanged
- formulas and lookbacks are unchanged
- no shorter-window substitution is used
- no forward-fill is used
- no MOEX substitution is used
- reaction target methodology is unchanged

The implementation uses only existing T-Invest read-only minute candle cache with provenance
`TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES`. It does not read broker tokens, does not use the
sandbox token, and does not call account or order endpoints.

Run:

```powershell
uv run python -m apps.cli.recover_exact_event_market_history
```

Required safety labels:

- DATA_RECOVERY_ONLY=true
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
- REAL_TRADING_ALLOWED=false
- REAL_ORDER_SUBMISSION_ALLOWED=false
- SANDBOX_ORDER_SUBMISSION_ALLOWED=false

The artifact is written to:

```text
artifacts/exact-event-market-history-warmup-recovery-v1
```

It contains root-cause rows, recovered feature rows, remaining fail-closed rows, preservation and
leakage checks, before/after concentration summaries, and deterministic hashes. It does not train a
model, does not evaluate TEST, does not open future holdout outcomes, and does not run any
backtest/trading workflow.
