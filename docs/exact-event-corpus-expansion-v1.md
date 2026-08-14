# Exact Event Corpus Expansion v1

## Purpose

This stage expands the real event corpus using official publication times before any further
event-model work. The frozen event baseline v1 remains observed and is not reused for iterative
tuning. Rules v3, financial-fact extraction, the Qwen prompt/schema/model, and all model settings
remain unchanged.

The immutable generated dataset family is `exact-event-market-dataset-v1`. It is physically
separate from DATE_ONLY / `DATE_SAFE_DAILY` artifacts.

## Source discovery and acceptance

`exact-event-source-registry-v1` covers all 315 mapped TQBR RUB instruments. A registry row records
issuer identity, official HTTPS domain, source family, parser, timestamp capability, timestamp
field, timezone semantics, historical range, incremental support, access/cost requirements,
policy status, collector status, and a fail-closed reason.

An official source is EXACT only when it provides a real publication time and the timezone is
explicit or documented. Calendar dates, HTML download time, and imputed market/open/close times
never become EXACT.

The implemented exact-capable families are:

| Ticker | Official source | Timestamp evidence | Capability |
| --- | --- | --- | --- |
| ROSN | Rosneft press-release RSS | RFC 822 `pubDate` with numeric offset | MIXED |
| YDEX | Yandex IR press-release RSS | RSS `pubDate` with timezone | MIXED |
| GMKN | Nornickel official application state | `activeFrom` Unix epoch seconds | MIXED |
| MGNT | Magnit official `/ru/api/news` feed | `date` Unix epoch seconds | EXACT |

Nornickel values resolving to source-local midnight are treated as date placeholders and rejected
from the exact corpus. Magnit pagination is bounded to 50 pages and 400 items per run. Selection is
source/date ordered and never uses event labels, market returns, target availability, or model
outputs. Responses are cached by URL and SHA-256 for deterministic resume.

The Bank of Russia lists accredited issuer-disclosure information agencies, but the official
e-disclosure REST gateway requires a paid authenticated subscription. It is excluded by the
project's zero-cost policy. No blocked public page, CAPTCHA, robots rule, authentication boundary,
or rate limit is bypassed.

References:

- <https://www.cbr.ru/admissionfinmarket/navigator/inf_ag/>
- <https://e-disclosure.ru/poluchenie-informacii/shlyuz-api>
- <https://developer.tbank.ru/invest/api/market-data-service-get-candles>
- <https://developer.tbank.ru/invest/services/history-md/head-history-md>

## Timestamp and clustering contracts

Each exact event stores the raw source value, UTC instant, publication date/time/timezone,
timestamp source field, stable source identity, canonical URL, title hash, provenance, and storage
policy. Unix values are parsed as UTC instants. No default clock time is injected.

Clustering is deterministic. It groups only same-issuer records with the same canonical
story/title identity in a 15-minute update window. Distinct events remain present. Exact source ID
or canonical-URL duplicates are diagnosed separately.

## T-Invest minute alignment

Intraday market data comes only from the production T-Invest read-only `GetCandles` method with
`CANDLE_INTERVAL_1_MIN` and `CANDLE_SOURCE_EXCHANGE`. Requests cover at most one UTC day. No MOEX
ISS substitution, source mixing, synthetic bars, interpolation, or forward fill is permitted.

T-Invest documents candle `time` as the UTC start of the candle interval. Therefore:

- the baseline is the last complete candle ending at or before publication;
- an event on a minute boundary starts at that minute's candle;
- an event inside a minute starts at the next full minute candle;
- security and IMOEX must use identical actual baseline/effective/target timestamps;
- missing candles produce explicit missing reasons;
- pre-open, after-close, non-trading-day, and ambiguous session gaps fail closed.

Session state is inferred from actual common T-Invest exchange candles, not weekday heuristics.
Unsupported pre-open or after-close events keep their exact timestamp but receive no fabricated
intraday reaction.

## Features and reactions

The feature cutoff is the publication timestamp. Pre-event market features end no later than that
cutoff. Frozen Rules v3 analyzes the information available at publication. Qwen is neither run nor
used. Targets contain 1/5/15/30/60 minute security, IMOEX, and abnormal returns only when both
series provide the same exact window.

The first bounded build produced:

- 449 REAL EXACT events, up from 42;
- 342 complete exact reactions;
- 239 exact event-market feature-ready rows;
- 4 tickers / 4 issuers;
- volume status `EXACT_BASELINE_EXPERIMENT_READY`;
- diversity status `EXACT_LOW_TICKER_DIVERSITY`.

The volume milestone does not override the diversity warning. No downsampling or artificial
balancing is performed.

## Forward holdout

`FUTURE_EVENT_HOLDOUT_START=2026-08-11`. Events on or after that date may be collected, matched,
timestamped, clustered, processed by frozen Rules v3, and given pre-event context. Research/model
outcome reads raise `FUTURE_EVENT_HOLDOUT_READ_ATTEMPT`.

Only count/date/ticker/issuer/missingness metadata is exported. Target distributions, abnormal
return summaries, predictions, correlations, and metrics are absent. The machine-readable status
is:

```text
FUTURE_EVENT_HOLDOUT_STATUS=ACCUMULATING
FUTURE_EVENT_HOLDOUT_OBSERVED=false
```

## Generated files

The gitignored `artifacts/exact-event-market-dataset-v1/` directory contains events, features,
pre-holdout targets, exclusions, source registry, clusters, timestamp/reaction/provenance manifests,
future holdout status, and a descriptive exact-vs-date-only diagnostic. Raw source and minute data
remain private local caches and are not committed or redistributed.

Build command:

```powershell
uv run python -m apps.cli.build_exact_event_corpus
```

The command requires `TINVEST_READONLY_TOKEN` and exposes no account, order, sandbox-order, money
movement, or execution service.

## Explicit non-goals

This stage performs no model training, A/B/C reevaluation, feature selection, threshold tuning,
backtest, paper trading, order submission, BUY/SELL generation, hybrid NLP, Rules changes, or Qwen
changes. All external data and APIs used here cost 0 RUB.
