# Exact event predictive baseline v1

This PR builds the first leakage-safe predictive baseline for exact timestamp events.
The predictive unit is an event:

NEWS/EVENT -> frozen event features -> pre-event market context -> exact post-event target.

## Frozen Inputs

The runner accepts `exact-event-market-dataset-v2` and fails closed if any frozen dataset,
source registry, provenance, timestamp, reaction, or cluster SHA changes. The event features
remain frozen Rules v3 output. Qwen is not run, prompts are not tuned, UNKNOWN remains a
legitimate frozen category, and no event taxonomy is changed.

Future events with `publication_date >= 2026-08-11` are a blind holdout. Research code may
report metadata counts for that period, but it must not read or aggregate outcomes, target
classes, target distributions, correlations, predictions, or metrics. The artifact records:

- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `CONFIRMED_SIGNAL=false`

## Horizons and Targets

`PRIMARY_EXACT_HORIZON=15m` is fixed before metrics. Secondary horizons are `1m`, `5m`, `30m`,
and `60m`. Each horizon has a separate eligible cohort, cohort SHA, split SHA, target, and
one-shot TEST evaluation record.

The regression target is:

`abnormal_return_h = security_return_h - IMOEX_return_h`

Security and IMOEX use the same effective market window. The target window starts strictly after
the publication timestamp using the exact corpus alignment policy. Missing target candles fail
closed; target prices are not forward-filled.

Classification uses the existing deterministic project-wide flat threshold of +/-0.002 abnormal
return. Thresholds are not derived from cohort outcomes.

## A/B/C Comparison

A/B/C always use the same event IDs, target, split, and temporal boundaries for a horizon:

- A = pre-event market context only
- B = frozen event features only
- C = exact union of A and B

Models are intentionally simple: majority class and Logistic Regression for classification,
zero/train-mean and Ridge Regression for regression. Preprocessing is inside sklearn pipelines and
is fit only on TRAIN for validation, then TRAIN+VALIDATION once for the locked TEST evaluation.
There is no hyperparameter search, target encoding, target-derived feature selection, model zoo,
ensemble, tree boosting, or neural network.

## Temporal Protocol

The preferred calendar split is TRAIN <= 2024-12-31, VALIDATION = 2025, and TEST =
2026-01-01 through 2026-08-10. If that is unusable for the actual exact cohort, the runner falls
back before metric inspection to a deterministic chronological 60/20/20 grouped split. Publication
dates and `event_cluster_id` are atomic and cannot cross partitions.

The locked config is written before TEST evaluation. The primary 15m TEST is observed exactly once:

- `TEST_CONFIG_LOCKED=YES`
- `TEST_EVALUATION_COUNT_PRIMARY=1`
- `TEST_STATUS=OBSERVED_AFTER_EXACT_BASELINE_V1`

## Concentration and Interpretation

MGNT concentration is not downsampled or issuer-weighted in training. The artifact reports row
weighted metrics, issuer macro metrics, per-ticker diagnostics, ticker shares, top1/top3 share,
HHI, and effective issuer count. Per-ticker detailed metrics are diagnostic and require explicit N
context.

The central comparison is C vs A on the primary 15m target. `EXACT_EVENT_INCREMENTAL_SIGNAL_CANDIDATE`
requires coherent support across multiple primary metrics, issuer-macro support, and evidence not
explained only by MGNT. Otherwise the status is `NO_EXACT_EVENT_INCREMENTAL_SIGNAL`.

The timestamp hypothesis is reported separately as either `TIMESTAMP_HYPOTHESIS_NOT_SUPPORTED` or
`TIMESTAMP_HYPOTHESIS_SUPPORTED_AS_CANDIDATE`. It is never a causal proof claim.

## Safety

This artifact is research only. No PnL, Sharpe, Sortino, drawdown, turnover, position sizing,
portfolio construction, backtest, paper trading, BUY/SELL output, sandbox order, production order,
broker mutation, or money movement is implemented or approved.

Required flags:

- `RESEARCH_ONLY=true`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`
- `PRICE_ADJUSTMENT_STATUS=UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES`
