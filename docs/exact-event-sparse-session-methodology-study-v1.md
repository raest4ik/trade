# EXACT Event Sparse Session Methodology Study v1

This PR is a diagnostics/study-only PR. It does not change `EXACT_INTRADAY`, does not recover
GEMC/INCB, does not create a new target family, and does not write production dataset rows.

The study uses the frozen development boundary from
`artifacts/exact-event-predictive-baseline-v1/15m-split-manifest.json`:

- protocol: `DETERMINISTIC_CHRONOLOGICAL_60_20_20_GROUPED_V1`
- development splits: `TRAIN` + `VALIDATION`
- TEST split rows are excluded before analysis
- unknown split membership fails closed
- future holdout starts at `2026-08-11` and is excluded

Decision rules are pre-registered in code before the real artifact run. A separate sparse-family
study is justified only when metadata-only evidence shows repeated sparse events across multiple
independent tickers, material coverage gain beyond the strict 60 second rule, and no single-ticker
dominance. Otherwise the recommendation is either keep strict-only or gather more data first.
Rows with uncertain local cache coverage remain in the per-event audit but are excluded from the
methodology recommendation denominator.

Allowed inputs are timestamp/availability metadata only:

- exact publication timestamp
- ticker, issuer, source family, and instrument identity
- candle `begin_at`, `end_at`, `is_complete`, and `instrument_uid`
- pre-event candle-density from timestamp counts only

Forbidden inputs and outputs:

- OHLC, VWAP, volume
- returns, abnormal returns, log returns
- target classes, labels, predictions, signals, PnL
- TEST outcomes, future holdout outcomes
- backtests, paper trading, orders, BUY/SELL

The code parses only event `metadata` objects and timestamp candle fields for study rows. It does
not call `align_exact_event()`, does not train models, and does not inspect target files.

Safety flags:

- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `TEST_EVALUATION_PERFORMED=false`
- `OBSERVED_TEST_ROWS_USED=0`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `STRICT_EXACT_METHODOLOGY_CHANGED=false`
- `MARKET_DATA_METHOD_CHANGED=false`
- `PRODUCTION_DATASET_CHANGED=false`
- `CONFIRMED_SIGNAL=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`
