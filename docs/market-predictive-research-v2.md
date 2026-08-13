# Market Predictive Research v2

This development cycle investigates market-only predictive structure without touching baseline
v1's observed TEST. The permitted development interval is `2000-02-04..2022-09-15`. The loader
fails closed with `OBSERVED_TEST_READ_ATTEMPT` for any requested interval reaching
`2022-09-20`. It scans JSONL date metadata before materializing a row, so excluded feature and
target payloads are not parsed by the research command.

Baseline v1 remains frozen with dataset, split, feature-schema, and artifact fingerprints recorded
in every v2 manifest. Its `2022-09-20..2026-08-11` TEST is observed and is prohibited for feature,
target, model, hyperparameter, threshold, or candidate selection. V2 performs no comparison or
evaluation on it.

## Development protocol

For each fixed 1, 3, and 5-session target horizon, five deterministic expanding folds group all
tickers for one date, keep validation strictly later than training, purge the full target horizon,
and apply a one-session embargo. There is no random split. Compounded security and T-Invest IMOEX
returns define the forward target; abnormal return is their difference. Classification thresholds
are fixed before evaluation as `0.002 * sqrt(horizon)`.

The schema retains 29 frozen t-1 features and adds deterministic momentum acceleration,
volatility term structure, volume trend, month-end status, and rank/z-score cross-sectional views
of five frozen features. Every cross-sectional input was already available by t-1. No target-day
OHLCV, forward fill, synthetic session, target encoding, clipping, winsorization, or
target-dependent deletion is used.

Fixed fold-local pipelines compare Ridge and Logistic Regression with one nonlinear family,
HistGradientBoosting. Preprocessing is fit only on each fold's TRAIN rows. Diagnostics include
target/class distributions, annual and ticker summaries, autocorrelation, cross-sectional
dispersion, missingness, feature stability, univariate Pearson/Spearman associations, per-fold
metrics, per-ticker/year stability, and same-date rank IC. These are associational diagnostics,
not causal or confirmed performance claims.

## Holdout and execution safety

Rows after `2026-08-11` are reserved as `FUTURE_BLIND_HOLDOUT_V1`. The status command reports only
session/date/ticker coverage from feature metadata. It never loads targets or computes model
performance. Future status is `ACCUMULATING` and `FUTURE_HOLDOUT_OBSERVED=false`.

This PR does not calculate PnL, Sharpe, Sortino, strategy drawdown, turnover, commissions,
slippage, positions, or portfolio returns. It cannot generate BUY/SELL instructions, submit real
or sandbox orders, mutate accounts, move money, paper trade, or connect models to live automation.
The price warning remains `UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES`.

```powershell
uv run python -m apps.cli.run_market_predictive_research_v2
uv run python -m apps.cli.market_future_holdout_status
```

Generated private research data remains under gitignored `artifacts/market-predictive-research-v2`.

## Frozen development result

The completed artifact SHA is
`e552307fc89b3f4e749591af671f1587b7f300cf51210adc9f037932873425ee`.
It contains 39,757 development rows, 43 features, 5 expanding folds per horizon (15 fold
definitions total), fold-manifest SHA
`ae99de61eb5e178abf46879a2d20294395c4b24628c809a8ccabd8d796a4f817`, and feature-schema SHA
`f7a60ecf55d7d0f7d455035810312224a30ee637a3bab2dfede231ca9dc0bb45`.

The lowest aggregate abnormal-return RMSE belongs to one-session
HistGradientBoostingRegressor: `0.018770`, versus `0.018858` for zero and `0.018859` for the
TRAIN-fold mean. Its MAE, however, is worse (`0.010998` versus zero's `0.010812`), and it beats
the naive RMSE in only 3 of 5 folds. The first three one-session fold rank IC means are positive
(`0.0459`, `0.0133`, `0.0562`), then weaken (`0.0084`, `-0.0146`). Across all out-of-fold dates
for this candidate, mean daily rank IC is `0.02186`, median is `0.0`, standard deviation is
`0.38063`, and the positive fraction is `0.49215`.

Classification diagnostics improve over majority-class macro F1, but they do not rescue the
unstable regression and rank evidence. Ticker and year diagnostics are mixed, including negative
or near-zero correlations for several issuers and later regimes. The deterministic conclusion is
therefore `DEVELOPMENT_STATUS=NO_DEV_SIGNAL` and `CONFIRMED_SIGNAL=false`. No observed or future
holdout was used.
