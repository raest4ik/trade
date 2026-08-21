# EXACT Event Session Alignment Diagnostics v1

This PR is diagnostics-only. It investigates PR38 events that remained blocked with
`FINAL_BLOCKER=SESSION_ALIGNMENT_FAILED` after bounded T-Invest security minute cache acquisition.

The diagnostic cohort is selected from `artifacts/exact-event-security-history-recovery-v1`, not
from a hardcoded ticker list. Only historical rows with `RECOVERY_STATUS=BLOCKED` and
`FINAL_BLOCKER=SESSION_ALIGNMENT_FAILED` are included. Future holdout events on or after
`2026-08-11` are excluded before any market or target path can run.

The artifact is `artifacts/exact-event-session-alignment-diagnostics-v1`. It contains timestamp-only
evidence:

- publication timestamp
- complete and incomplete candle timestamp counts
- previous and next candle timestamps
- nearest complete timestamps before and after publication
- common security/benchmark begin timestamps
- baseline and effective timestamp candidates
- window equality flags
- cache-window sufficiency
- frozen `align_exact_event(..., expose_outcomes=False)` internal missing reason
- normalized root cause and recovery recommendation type

The manifest preserves the PR38 source-of-truth lineage with `PR38_ARTIFACT_SHA` and
`PR38_RECOVERY_COHORT_SHA`.

The artifact must not contain OHLC, return, abnormal return, log return, target class, prediction,
or trading signal values.

Frozen boundaries:

- `align_exact_event()` unchanged
- `classify_session()` unchanged
- `HORIZONS_MINUTES` unchanged
- one-minute common-candle tolerance unchanged
- baseline/effective definitions unchanged
- exact timestamp equality checks unchanged
- benchmark methodology unchanged
- feature/reaction formulas unchanged

Safety flags:

- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `ALIGNMENT_METHODOLOGY_CHANGED=false`
- `MARKET_DATA_METHOD_CHANGED=false`
- `CONFIRMED_SIGNAL=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`

No model training, TEST outcome use, future holdout observation, methodology recovery, backtest,
paper trading, orders, or BUY/SELL output is performed.
