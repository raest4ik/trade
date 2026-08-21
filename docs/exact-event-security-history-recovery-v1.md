# EXACT Event Security History Recovery v1

This PR is data-recovery-only. It uses the PR37 diagnostics artifact as the source of truth for
selecting historical events whose current T-Invest TQBR identity has minute history available, then
acquires only the bounded security-side minute cache needed by the frozen EXACT market alignment
methodology.

The recovery cohort is selected from `artifacts/exact-event-security-history-diagnostics-v1` where
`ROOT_CAUSE=CURRENT_IDENTITY_HAS_HISTORY`, `RECOVERY_POSSIBLE=true`, and
`RECOVERY_PERFORMED=false`. Future holdout events with publication date on or after `2026-08-11`
are excluded before any market or target path can run.

The new artifact is `artifacts/exact-event-security-history-recovery-v1`. It records UID-bound
cache provenance using ticker, FIGI, instrument UID, class code, and interval. Cache acquisition is
bounded to the event day plus the frozen seven-day warmup window. Existing raw minute cache roots are
not deleted or rewritten; the recovery cache is artifact-local and idempotently deduped by instrument
UID and candle timestamp.

Frozen methodology boundaries:

- T-Invest read-only production exchange candles only
- benchmark methodology unchanged: existing IMOEX cache
- frozen EXACT horizons: 1m, 5m, 15m, 30m, 60m
- no MOEX substitution
- no forward-fill
- no synthetic market data
- `MAX_FEATURE_TIMESTAMP < PUBLICATION_TIMESTAMP` for every recovered feature row
- existing non-cohort event rows preserved
- existing feature rows preserved
- Rules v3 unchanged
- Qwen prompt/schema unchanged

Safety flags:

- `RESEARCH_ONLY=true`
- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `CONFIRMED_SIGNAL=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`
- `REAL_ORDER_SUBMISSION_ALLOWED=false`
- `SANDBOX_ORDER_SUBMISSION_ALLOWED=false`

This PR does not train a model, inspect TEST, open future outcomes, run a backtest, paper trade,
submit orders, or emit BUY/SELL output.
