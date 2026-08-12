# T-Invest Read-Only Market Data v1

This pipeline creates a private, zero-cost historical daily market dataset from the official
T-Invest API. It is data infrastructure only. It does not train a model, run a strategy backtest,
generate BUY/SELL decisions, perform paper trading, or expose any order or account-mutation API.

## Official Evidence

Evidence was checked on 2026-08-13 using only official T-Bank/T-Invest pages:

- [API introduction](https://developer.tbank.ru/invest/intro/intro/) states that T-Invest API data
  is free and provides market data for client-built historical algorithm checks.
- [Token documentation](https://developer.tbank.ru/invest/intro/intro/token) says a read-only token
  can read schedules, quotes, and historical data but cannot submit trading instructions. A
  sandbox token is isolated and fails against the ordinary contour.
- [Client exchange-information terms](https://www.tbank.ru/invest/disclaimers/basic-information/)
  permit a client to use, store, and process received exchange information. Onward or public
  distribution requires written exchange consent.
- [GetCandles](https://developer.tbank.ru/invest/api/market-data-service-get-candles) documents the
  official daily candle method, a roughly six-year maximum request interval, and a 2400-row limit.
- [Historical quotations](https://developer.tbank.ru/invest/intro/intro/load_history) states that
  depth varies by instrument and exposes `first_1day_candle_date` as the available-history start.
- [Instrument lookup](https://developer.tbank.ru/invest/api/instruments-service-find-instrument)
  documents exact REST lookup and UID/FIGI metadata.
- [Instrument service guidance](https://developer.tbank.ru/invest/services/instruments/head-instruments)
  documents `Indicatives`, including IMOEX, and says its UID can be passed to `GetCandles`.
- [Trading schedules](https://developer.tbank.ru/invest/services/instruments/head-instruments)
  documents a maximum one-week current schedule window; this pipeline does not infer historical
  sessions from current schedules.
- [API limits](https://developer.tbank.ru/invest/intro/intro/limits) defines unary rate limits. The
  client uses bounded retries and exponential backoff, and does not evade those limits.
- [Sandbox](https://developer.tbank.ru/invest/intro/developer/sandbox) documents the isolated
  sandbox endpoint. This version performs harmless read connectivity only and never creates,
  funds, or trades a sandbox account.
- [Corporate actions](https://developer.tbank.ru/invest/intro/useful-info/faq_corp_action) documents
  split, reverse-split, spinoff, stock-dividend, delisting, and conversion effects.

The resulting policy is `PRIVATE_CLIENT_INTERNAL_USE`: private dataset construction, private model
training, and private research backtesting are allowed. Public redistribution is not allowed. This
does not change the MOEX ISS dataset policy, which remains blocked for production/trading use.

## Secret And Execution Safety

Only `TINVEST_READONLY_TOKEN` is accepted by the production market-data contour. Only
`TINVEST_SANDBOX_TOKEN` is accepted by the separate sandbox connectivity contour. Values are read
from environment variables, hidden from dataclass representations, and never stored in logs,
exceptions, manifests, fixtures, or artifacts. Missing variables fail closed and report only the
missing environment-variable name.

`TInvestReadOnlyClient` exposes only instrument lookup, indicative lookup, historical candles, and
trading schedules. It has no generic arbitrary-endpoint method and no Orders, StopOrders,
Operations, funding, transfer, withdrawal, or account-mutation surface. Production and sandbox
base URLs are fixed and there is no cross-contour fallback.

All execution flags remain false, including real trading, real and sandbox order submission, stop
orders, money movement, broker-account mutation, margin, and live execution. Dataset readiness or
future model results cannot change these flags. Enabling execution requires a separate explicit
future PR and user decision.

## Instrument Identity And History

The security universe is `SBER`, `SBERP`, `GAZP`, `LKOH`, `ROSN`, `NVTK`, `YDEX`, `T`, `VTBR`, and
`GMKN`. The resolver accepts exactly one exact-ticker UID. Missing or ambiguous matches fail closed.
The persisted mapping includes ticker, class code, UID, optional FIGI, instrument type,
`first_1day_candle_date`, name, and resolution time. Different historical UIDs are never joined.

IMOEX is resolved only through T-Invest `Indicatives`. If it is available, its T-Invest candles
provide benchmark features and the abnormal-return target. If it is unavailable, the dataset is
built without benchmark features and abnormal targets. MOEX ISS never fills a T-Invest row.

Daily history starts at the later of the requested date and the provider's
`first_1day_candle_date`. Requests are split into 1800-day chunks, respect provider limits, and use
bounded retry/backoff for transient failures and 429 responses. Per-UID checkpoints make reruns
idempotent and resumable. No forward fill, synthetic session, or synthetic candle is created.

## Dataset And Leakage Policy

Raw data is stored under the gitignored `artifacts/tinvest-market-raw-v1` namespace. Features and
targets are stored separately under the gitignored
`artifacts/tinvest-market-baseline-features-v1` namespace. Manifests contain fingerprints and
provenance but no credentials.

Rolling features use only observations through `t-1`. The next-session return and optional IMOEX
abnormal return belong only to the target row for `t`. The chronological split groups all tickers
for the same date and applies purge and embargo dates between train, validation, and test. There is
no random split, target-based filtering, clipping, or winsorization. Extreme returns are retained
and separately audited.

The official corporate-action page says historical quotations are adjusted for split and reverse
split, while conversions may replace identifiers and change price-related fields. It does not
establish a universal fully-adjusted guarantee for every instrument and action. Therefore the
honest status is `UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES`; the extreme-return audit is diagnostic,
and multiple UIDs are never silently stitched.

An optional MOEX overlap report compares next-session returns only where row IDs intersect. It is
diagnostic: providers are never averaged, one never fills the other, and event daily rows remain a
separate count.

## Commands

```powershell
uv run python -m apps.cli.tinvest_market_build
uv run python -m apps.cli.tinvest_market_status
```

The build performs read-only authentication, a harmless metadata/candle sample, bounded history
acquisition, dataset generation, split creation, and reporting. `--no-check-sandbox` skips the
separate sandbox connectivity check. Neither command prints token values.
