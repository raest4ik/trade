# T-Invest Market Predictive Baseline v1

This experiment is the first real predictive baseline on the frozen private T-Invest daily market
dataset. It asks only whether fixed linear models beat naive out-of-sample baselines. It is an
associational research model, not a strategy, recommendation, signal service, or trading system.

## Frozen inputs and targets

The command accepts only `tinvest-market-baseline-features-v1` with dataset SHA
`92e8d813b755f715da8a3323fd540c8400943773202ffe5cc1c7ac6033c35425`, split SHA
`6dad767f62e69e5e55e25bc48998707d3aeafbec76f65334bbdd0ad4bec85929`, and feature-schema SHA
`83aee83bb403d7035e1e3daabe5c05b680e8263ea6a36e3c7812d11c20f838e0`. Any mismatch fails closed.
The primary regression target is next-session abnormal return against T-Invest IMOEX. Raw
next-session security return is secondary. UP/FLAT/DOWN uses the existing frozen 0.2% policy.

Extreme observations are retained without clipping, winsorization, synthetic replacement, or
target-based deletion. The source warning `UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES` remains on the
model artifact.

## Evaluation protocol

Stage one fits fixed Logistic Regression and Ridge pipelines on TRAIN and evaluates VALIDATION.
There is no parameter search. Each pipeline fits its `StandardScaler` only on the stage-authorized
fit rows. The final model configuration, targets, threshold, seed, preprocessing, features, and all
three input fingerprints are written with exclusive-create semantics before TEST is loaded for
evaluation.

The final pipelines then refit preprocessing and models on TRAIN plus VALIDATION. The state file is
advanced to evaluation count one before TEST target materialization, making a failed attempt
non-repeatable. TEST is evaluated once and becomes `OBSERVED_AFTER_BASELINE_V1`. It must not be
used for iterative changes; future blind confirmation requires a new forward holdout.

Metrics include naive and learnable classification and regression views, per-ticker and per-year
diagnostics, ticker-macro summaries, and date-equal-weighted diagnostics. Coefficients are reported
on standardized features and carry `ASSOCIATIONAL_MODEL_ONLY`; they do not establish causality.

`BASELINE_SIGNAL_PRESENT` requires simultaneous classification and abnormal-return improvements,
plus broad ticker and year support. Otherwise the deterministic result is `NO_PREDICTIVE_SIGNAL`.
Neither status means trade-ready, backtest-ready, or production-ready.

## Safety

The module has no broker client, live collector integration, order surface, portfolio accounting,
or strategy backtest. It never computes PnL, Sharpe, Sortino, drawdown, turnover, commissions,
slippage, position sizes, or portfolio returns. BUY/SELL generation, paper trading, sandbox orders,
production orders, account mutation, and money movement remain disabled.

Run the one-time evaluation only against a fresh output directory:

```powershell
uv run python -m apps.cli.train_tinvest_market_baseline
```

Generated private data, predictions, and model binaries remain under the gitignored `artifacts/`
namespace and must not be redistributed.

## Frozen baseline v1 result

The one permitted TEST evaluation completed with `TEST_CONFIG_LOCKED=YES`,
`TEST_EVALUATION_COUNT=1`, and `TEST_STATUS=OBSERVED_AFTER_BASELINE_V1`. The immutable local
artifact SHA is `98c4aeff24815bcc946344a7a14d81f399023e395889fa5400394bc1c455064d`; its
configuration SHA is `c53b31df2a192a7cd23b00b57582d457f8bbadf6e85f1dc85b3f1d14df5177a0`.

| TEST classification | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.416667 | 0.333333 | 0.196078 | 0.245098 | 1.001427 | 0.610649 |
| Logistic Regression | 0.400020 | 0.333069 | 0.301226 | 0.357270 | 1.025859 | 0.625050 |

| TEST abnormal-return regression | MAE | RMSE | R2 | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero baseline | 0.010435 | 0.015609 | -0.000115 | n/a | n/a |
| TRAIN+VALIDATION mean baseline | 0.010465 | 0.015610 | -0.000205 | n/a | n/a |
| Ridge Regression | 0.010504 | 0.015576 | 0.004162 | 0.072928 | 0.029867 |

The TEST confusion matrix (actual rows, predicted columns in DOWN/FLAT/UP order) is
`[[877, 392, 3139], [259, 107, 973], [820, 328, 2957]]`. Logistic predicts UP for 7,069 of
9,852 rows. Ridge's small aggregate RMSE improvement does not survive the stricter joint criteria:
MAE is worse than both naive regressors, classification probability metrics are worse than the
majority baseline, and abnormal-return correlations are weak and temporally inconsistent. The
deterministic status is therefore `NO_PREDICTIVE_SIGNAL`.

The result is research-only. It does not authorize a backtest, paper trade, sandbox order,
production order, BUY/SELL output, live integration, or any other execution activity.
