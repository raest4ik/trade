# Exact event predictive baseline v1

This is a research-only event-driven baseline for exact publication timestamps.
It compares A market context, B frozen Rules v3 event features, and C their union.

## Locked Design

- Dataset SHA: `20ab67ff4d94c59d6cf714f8b2f7c048031bda120bbd92ceb6e6185a838e14c3`
- Primary horizon: `15m`
- Primary cohort rows: 408
- Primary cohort SHA: `47eaa4f88c99ca0f04fc987514e03f80671922018157007f8b81ba3b87205d6b`
- Split SHA: `574d080a572cf7a4f6904f04e7594243afe95fb7bd2eb484ac0fd6328fab38d2`
- TEST status: `OBSERVED_AFTER_EXACT_BASELINE_V1`
- TEST evaluation count primary: 1

## Primary Result

- Incremental value: `NO_EXACT_EVENT_INCREMENTAL_SIGNAL`
- Timestamp hypothesis: `TIMESTAMP_HYPOTHESIS_NOT_SUPPORTED`
- Confirmed signal: `False`
- MGNT/top1 share: 0.243902
- HHI: 0.170434
- Effective issuer count: 5.867365

No future holdout outcomes were used or observed. No PnL, Sharpe, backtest, paper trading, BUY/SELL signal, order, position sizing, portfolio simulation, or broker mutation is part of this artifact.
