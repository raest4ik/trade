# Market Reactions v2

Market Reactions v2 produces auditable intraday research labels adjusted for broad market
movement. A raw stock return can reflect a market-wide move rather than company-specific news.
The first supported benchmark is the MOEX Russia Index (`IMOEX`).

## Reaction semantics

The existing security-candle rules remain unchanged:

- baseline: the last fully completed one-minute candle with `end_at <= published_at`;
- effective event: the first saved candle with `begin_at >= published_at`;
- target time: effective event plus 1, 5, 15, 30, or 60 calendar minutes;
- target close: the first saved candle with `end_at >= target time`.

The calculation follows saved candles and does not assume every calendar minute is a trading
minute. A gap can therefore move the observed target beyond the nominal target time.

## Benchmark alignment

Benchmark observations are bounded by the actual security interval, not by an independently
chosen nominal window:

- benchmark baseline: last completed IMOEX candle ending at or before the security baseline
  observation;
- benchmark target: first completed IMOEX candle ending at or after the security target
  observation.

The stored security and benchmark timestamps make any gap or alignment difference visible.
The v2 implementation does not interpolate prices and does not include a trading-calendar
subsystem.

## Returns

All prices and return calculations use `Decimal`. Intermediate values are not rounded.

```text
security_simple = security_target / security_baseline - 1
benchmark_simple = benchmark_target / benchmark_baseline - 1
abnormal_simple = security_simple - benchmark_simple

security_log = ln(security_target / security_baseline)
benchmark_log = ln(benchmark_target / benchmark_baseline)
abnormal_log = security_log - benchmark_log
```

For example, if SBER rises by 1.50% while IMOEX rises by 0.90%, the abnormal simple return is
approximately +0.60 percentage points.

## Timestamp safety

`NewsItem.publication_timestamp_quality` is structured data with three values:

- `EXACT`: a trusted publication timestamp; reaction calculation is allowed;
- `DATE_ONLY`: only the publication date is known; reaction calculation is blocked;
- `UNKNOWN`: timestamp provenance is not trusted; reaction calculation is blocked.

The migration assigns `UNKNOWN` to existing rows. The Batch 001 seed importer explicitly writes
`DATE_ONLY`; its technical midnight timestamp is only an ingestion aid. `DATE_ONLY /
DO_NOT_USE_FOR_REACTION` rows cannot create security, benchmark, abnormal-return, or training
labels.

## Missing data

When a security reaction exists but one or both benchmark observations are absent, the benchmark
adjustment is stored with `status=MISSING`, a diagnostic `missing_reason`, and null abnormal
returns. A missing benchmark is never treated as a zero return and no price is fabricated.

When the security point itself is unavailable, the adjustment is `NOT_APPLICABLE` and remains
null.

## Storage and audit

`market_benchmarks` stores benchmark identity and MOEX ISS routing metadata. IMOEX one-minute
candles are stored idempotently in `benchmark_candles`. The shared `market_data_imports` audit
table distinguishes `SECURITY` and `BENCHMARK` datasets and records code, date range, provider,
counts, timestamps, status, and error code.

Each `reaction_benchmark_adjustments` row is linked to exactly one reaction point and benchmark.
It stores benchmark values, observation timestamps, simple/log returns, abnormal simple/log
returns, status, and missing-data reason.

## Commands

```bash
just backfill-benchmark IMOEX 2026-07-01 2026-07-07
just compute-abnormal-reactions <news-id>
just compute-abnormal-reactions-all 100
just export-market-reaction-dataset artifacts/market-reaction-dataset-v2.jsonl
```

The Python CLIs expose `--dry-run` and `--limit` for bounded bulk computation where applicable.

## Dataset contract

`market-reaction-dataset-v2` is raw, auditable JSONL. Every row identifies the news, publication
timestamp quality, instrument, ticker, analysis/reaction versions, horizon, security interval and
returns, benchmark interval and returns, and abnormal returns.

The `labels` object is deliberately separate from event-side data. Abnormal returns are labels,
not input features available at publication time. The export performs no ML feature engineering.

## Limitations

This version supports only one-minute MOEX ISS candles, IMOEX, and 1/5/15/30/60-minute horizons.
It does not interpolate missing candles, model exchange calendars, acquire historical corporate
news, predict returns, generate trading signals, or execute trades. The result is a research label,
not a trading recommendation.
