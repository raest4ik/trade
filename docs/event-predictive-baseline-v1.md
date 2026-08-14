# Event predictive baseline v1

This experiment returns the project to its original event-driven question: do frozen event
features add predictive value beyond market context that was already available before a real
publication? The predictive unit is an event, never an arbitrary market day.

## Frozen inputs

The runner accepts only `event-market-predictive-dataset-v2` with the four SHA-256 fingerprints
frozen by PR #28. A mismatch in the dataset, source registry, provenance manifest, or feature
schema fails closed. The primary family is `DATE_SAFE_DAILY`; `EXACT_INTRADAY` stays separate and
is marked `INSUFFICIENT_DATA_FOR_BASELINE`.

`COMPARISON_COHORT_V1` is the intersection of REAL, DATE_ONLY, uniquely matched, reaction-ready,
feature-ready, point-in-time-safe rows. A, B, and C always use the same event IDs:

- A uses only pre-event market context.
- B uses only frozen Rules v3 event features.
- C uses the exact union of A and B.

Rules v3, financial facts, ontology, gold sets, Qwen prompt/schema/model, and target thresholds are
unchanged. Qwen is not run. The historical arbitrary-day market-only experiment remains a frozen
negative baseline and is not compared directly because its predictive unit differs.

## Blind temporal protocol

Publication dates define a deterministic chronological split: 2022-2024 TRAIN, 2025 VALIDATION,
and 2026 TEST. All events on one date, all issuer-date groups, and available same-story groups stay
in one partition. Split boundaries and row/ticker counts are inspected before target outcomes.

Logistic Regression and Ridge use fixed a-priori parameters. `DictVectorizer` safely handles unseen
event categories, and all vectorization/scaling is fit only on TRAIN for validation. The final
configuration is locked before any TEST target is loaded; final models are then fit on
TRAIN+VALIDATION and TEST is evaluated once. Its permanent status is
`OBSERVED_AFTER_EVENT_BASELINE_V1`.

TEST must not be reused for feature, model, threshold, source, issuer, split, or hyperparameter
tuning. After any result, confirmation requires a new forward holdout. Development-only
leave-one-issuer-out results and TEST per-ticker, per-year, and source-family views are diagnostics,
not tuning inputs or claims of causal/generalized performance.

The frozen run used comparison cohort SHA
`72d7969a61bd1ea43a9acfdfe5a088b8a3abfb23570bd056d431b61105073290` and split SHA
`39811927521e938889468c6b3fc4fec92b3d537fbf4f984e9ae37bab1090cec7`. It contained 1,233 events:
406 TRAIN, 479 VALIDATION, and 348 TEST. TEST was evaluated exactly once and is now
`OBSERVED_AFTER_EVENT_BASELINE_V1`.

## Frozen result

| TEST model | Balanced accuracy | Macro F1 | Log loss | Abnormal MAE | Abnormal RMSE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A market only | 0.384487 | 0.369508 | 1.181178 | 0.019514 | 0.033327 | -0.221442 |
| B event only | 0.327491 | 0.217927 | 1.036457 | 0.018003 | 0.030675 | -0.034775 |
| C event plus market | 0.362260 | 0.341432 | 1.255697 | 0.020235 | 0.033886 | -0.262743 |

C minus A was -0.022227 balanced accuracy, -0.028076 macro F1, +0.074519 log loss,
+0.000722 MAE, and +0.000559 RMSE. Issuer-macro results also did not support C over A. The frozen
interpretation is therefore `NO_EVENT_INCREMENTAL_SIGNAL`, not a reason to tune against this TEST.
The immutable research artifact SHA is
`71dbb103956b81098bc8ff1a479f6d9cea4cc1ec68dbc329b9410f76511585fa`.

## Interpretation and safety

`EVENT_INCREMENTAL_SIGNAL_CANDIDATE` requires C to improve over A across multiple classification
and regression measures, issuer-macro support, and support outside YDEX. Otherwise the status is
`NO_EVENT_INCREMENTAL_SIGNAL`. In either case `CONFIRMED_SIGNAL=false` because issuer concentration
is material and T-Invest daily candle price-adjustment status is unverified.

No PnL, Sharpe, Sortino, drawdown, turnover, position sizing, portfolio construction, backtest,
paper trading, BUY/SELL output, sandbox order, production order, broker mutation, or money movement
is implemented or approved. The existing live collector remains collection-only and is not wired
to training or prediction.
