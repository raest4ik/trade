# Market Baseline Dataset v1

`market-baseline-features-v1` is a market-only, point-in-time research dataset built from
the official MOEX ISS daily-candle endpoint. It remains independent from the event corpus,
event readiness gates, Rules, Qwen, and the Windows live corpus collector.

## Data rights gate

Free delayed HTTP access does not by itself confirm permission for predictive trading use.
The official [MOEX market-data policy](https://www.moex.com/ru/datapolicy/) identifies
automated processing in algorithmic-trading and risk-management systems as Non-display
usage and describes the applicable information agreements. The official
[MOEX offers page](https://www.moex.com/a6400) likewise defines Non-display use, while the
[IMOEX page](https://www.moex.com/en/index/imoex) states separate terms for index
information and trademarks.

The project makes no legal conclusion beyond that official wording. Until usage rights are
verified or an applicable agreement is identified, the machine-readable policy is:

```text
source_usage_status = RESEARCH_ONLY_PENDING_USAGE_RIGHTS
market_usage_readiness = BLOCKED_BY_SOURCE_USAGE_RIGHTS
overall_production_readiness = TRAINING_BLOCKED_FOR_TRADING_USE
production_training_allowed = false
backtest_for_trading_allowed = false
live_signal_use_allowed = false
```

The project data budget remains zero rubles. No paid contract or data service will be
purchased as part of this work. Research dataset construction and quality diagnostics are
allowed, but row-count readiness can never override the source-usage gate.

## Row and label semantics

Each row identifies one `ticker` and target `trade_date`. Every feature is computed from
completed sessions through the immediately preceding common ticker/IMOEX session (`t-1`).
The current target day's OHLCV never enters X. Targets are physically stored in a separate
file and contain the ticker next-session return, the same-window IMOEX return, their
difference, and a fixed v1 `UP`/`FLAT`/`DOWN` label. The flat band is fixed at +/-0.2% and is
not selected using validation or test data.

Features contain only lagged market data: returns, volatility, volume statistics, moving
average distances, IMOEX context, relative returns, trailing beta/correlation, and calendar
values known before the target session. Event and news outputs are excluded.

## Price integrity

Ticker sessions align exactly with the IMOEX calendar across each feature and target
window. Missing prices are never forward-filled and no synthetic rows are created. Listing
boundaries remove only unavailable warm-up windows.

The exact candle response provides OHLC, value, volume, begin, and end, but no official
primary-source contract located for this endpoint establishes corporate-action adjustment
semantics. The status therefore remains `UNVERIFIED_MOEX_ISS_CANDLE_PRICES`.

`price-integrity-audit.json` reports raw consecutive-session return percentiles, threshold
counts, dates and tickers for discontinuities, and abnormal-return extremes. The audit is
diagnostic only. It does not clip, winsorize, filter, or remove an observation because its
future return is extreme.

## Temporal split

The split is chronological and grouped by `trade_date`, so every ticker on a date belongs
to one partition. One session is purged from the end of TRAIN and VALIDATION, and one is
embargoed at the beginning of VALIDATION and TEST. Random splitting is unavailable. Any
future preprocessing must fit scalers, imputers, and encoders on TRAIN only.

## Commands

```powershell
uv run python -m apps.cli.market_baseline_build
uv run python -m apps.cli.market_baseline_status
```

The lower request bound is `2000-01-01`; manifests report only dates actually returned by
MOEX. The default upper bound is yesterday to avoid an incomplete current session.
Generated data and reports remain under the gitignored `artifacts/market-baseline-v1/`.

## Readiness axes

Data readiness is based on real feature-ready rows: below 1,000 is `NOT_READY`, 1,000-4,999
is `MARKET_PILOT_READY`, 5,000-9,999 is `MARKET_BASELINE_EXPERIMENT_READY`, and at least
10,000 is `MARKET_BASELINE_TRAINING_READY`. Fewer than five tickers adds
`LOW_TICKER_DIVERSITY`.

This technical axis is separate from usage readiness. The current dataset is technically
large enough, but production training, trading backtests, and live signals remain blocked
by source-usage rights. No model is trained here. The event collector continues to grow its
separate corpus independently.
