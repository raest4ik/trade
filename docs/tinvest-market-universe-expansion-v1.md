# T-Invest Market Universe Expansion v1

This corpus expands the accepted ten-security market dataset through the official T-Invest
read-only API. It does not train, select, or evaluate a model and does not create a backtest or
trading surface.

## Membership and source

Discovery calls `InstrumentsService/Shares` with `INSTRUMENT_STATUS_ALL`. The deterministic
candidate rule is `instrument_type=share`, `currency=rub`, and `class_code=TQBR`. Other official
class codes remain in the discovery diagnostics and are not admitted automatically. No ticker or
name heuristic is used.

The catalog is a current T-Invest snapshot, not a point-in-time historical membership dataset:

- `UNIVERSE_MEMBERSHIP_MODE=CURRENT_TINVEST_CATALOG_SNAPSHOT`
- `HISTORICAL_MEMBERSHIP_POINT_IN_TIME_VERIFIED=false`
- `SURVIVORSHIP_BIAS_RISK=PRESENT`

Only the official production read endpoint is used, with a read-only token and production TLS
verification. IMOEX is acquired independently from T-Invest. MOEX ISS is not used to fill gaps.

## Acquisition and quality

The build uses bounded daily-candle chunks, bounded retry and 429 backoff, UID-bound identities,
per-series checkpoints, and an exclusive run lock. A repeated run reuses completed ranges; a later
end date requests only the missing tail. Raw and generated files stay under gitignored `artifacts/`.

Candles must have unique ticker/date identities, valid OHLC ranges, nonnegative volume, and complete
status. Missing sessions are reported and never forward-filled, interpolated, or fabricated. Extreme
returns above 10%, 20%, and 50% are retained and reported. Prices remain
`UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES`; corporate-action jumps are not heuristically changed.

History depth is diagnostic: `HISTORY_LT_252`, `HISTORY_252_PLUS`, `HISTORY_756_PLUS`, or
`HISTORY_1260_PLUS`. A short-history instrument remains in the raw corpus.

## Features and protected partitions

The expanded artifact uses the accepted 43 Market Predictive Research v2 feature definitions.
Every feature window ends at `t-1`. No target or prediction is persisted in the expanded feature
artifact.

- `DEVELOPMENT`: through 2022-09-15
- `PURGE_EMBARGO_GAP`: 2022-09-16 through 2022-09-19
- `OBSERVED_V1_TEST`: 2022-09-20 through 2026-08-11
- `FUTURE_BLIND_HOLDOUT`: from 2026-08-12

Observed and future partitions may accumulate market rows, but this command never passes them to
research/model code and never computes predictions or predictive metrics. The future holdout status
is metadata-only: `ACCUMULATING`, `FUTURE_HOLDOUT_OBSERVED=false`.

## Incremental command

From PowerShell, with `TINVEST_READONLY_TOKEN` in the environment and
`SSL_TBANK_VERIFY=True`:

```powershell
uv run python -m apps.cli.tinvest_market_universe_build
```

The command is safe for future manual daily execution and is idempotent. This PR does not install a
Windows scheduled task. It exposes no account, order, stop-order, money-movement, sandbox-order,
BUY/SELL, model-training, or backtest behavior.
