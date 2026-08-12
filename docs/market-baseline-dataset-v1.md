# Market Baseline Dataset v1

`market-baseline-features-v1` is a market-only, point-in-time dataset built from the
official zero-cost MOEX ISS daily-candle endpoint. It is independent from the event corpus,
event readiness gates, Rules, Qwen, and the Windows live corpus collector.

## Row and label semantics

Each row identifies one `ticker` and target `trade_date`. Every feature is computed from
completed sessions through the immediately preceding common ticker/IMOEX session (`t-1`).
The current target day's OHLCV never enters X. Targets are physically stored in a separate
file and contain the ticker next-session return, the same-window IMOEX return, their
difference, and a fixed v1 `UP`/`FLAT`/`DOWN` label. The flat band is fixed at +/-0.2% and is
not selected using validation or test data.

The feature schema contains lagged returns, rolling volatility and volume statistics,
distance from moving averages, lagged IMOEX features, ticker-relative returns, trailing
20-session beta/correlation, and calendar values known before the target session.

## Missing data and corporate actions

Ticker sessions must align exactly with the IMOEX calendar across the full 20-session
feature window and target transition. Missing prices are never forward-filled and no
synthetic market rows are created. Listing boundaries therefore remove only unavailable
warm-up windows. Extreme future returns are retained; target-based cleaning is forbidden.

The MOEX candle schema exposes OHLC, value, volume, begin, and end. The official ISS
materials describe these as chart candles but do not provide a reliable adjusted-price
contract in this response. The dataset therefore records
`UNVERIFIED_MOEX_ISS_CANDLE_PRICES` and must not be described as corporate-action adjusted.

## Temporal split

The split is chronological and grouped by `trade_date`, so every ticker on a date belongs
to one partition. One session is purged from the end of TRAIN and VALIDATION, and one
session is embargoed at the beginning of VALIDATION and TEST. Random splitting is not
available. Future preprocessing must fit scalers, imputers, and encoders on TRAIN only.

## Commands

```powershell
uv run python -m apps.cli.market_baseline_build
uv run python -m apps.cli.market_baseline_status
```

The default acquisition lower bound is `2000-01-01`, while reported coverage always uses
the dates actually returned by MOEX. The default upper bound is yesterday, avoiding an
incomplete current trading session. Generated data and reports live under the gitignored
`artifacts/market-baseline-v1/` directory.

## Readiness

Readiness is based on real feature-ready rows: below 1,000 is `NOT_READY`, 1,000-4,999 is
`MARKET_PILOT_READY`, 5,000-9,999 is `MARKET_BASELINE_EXPERIMENT_READY`, and 10,000 or more
is `MARKET_BASELINE_TRAINING_READY`. Fewer than five represented tickers adds
`LOW_TICKER_DIVERSITY`. These statuses authorize no training in this change; model training,
tuning, backtesting, and trading signals remain out of scope.
