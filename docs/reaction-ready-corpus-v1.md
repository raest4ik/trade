# Reaction-Ready Corpus v1

## Scope

`reaction-ready-corpus-v1` is the first controlled pipeline for real corporate releases with
trusted publication timestamps and benchmark-adjusted MOEX reactions. It orchestrates existing
historical ingestion, deterministic matching, `event-rules-v2`, `financial-facts-v2`,
`reaction-v2-benchmark-adjusted`, and `ml-features-v1`. It does not change any of those semantics.

The acquisition filter is limited to source, publication date, and the configured instrument
universe. Future return, abnormal return, future volume, reaction magnitude, polarity, and event
type are never acquisition criteria.

## Provenance

Every source is classified as `REAL`, `SYNTHETIC`, `SEED`, or `OTHER`. A source is `REAL` only when
its source code is explicitly approved after source audit. Unknown source codes never default to
`REAL`. Synthetic smoke data, Batch 001, and manually curated seed data are excluded from
`real_reaction_ready_rows` even if they can pass the technical pipeline.

The current approved source is:

| Source code | Owner | Kind | Policy | Timestamp |
| --- | --- | --- | --- | --- |
| `ROSNEFT_PRESS_RELEASES_RSS` | Rosneft Oil Company | issuer-owned RSS | `EXCERPT_ALLOWED` | `EXACT`, RFC 822 numeric offset |

The generated source audit is written to
`artifacts/historical-news-v1/source-audit.json`. The audit covers SBER, SBERP, GAZP, LKOH, ROSN,
NVTK, YDEX, T, VTBR, and GMKN. It records ownership, HTTPS, observed depth, timestamp precision,
timezone semantics, text availability, storage policy, archive capability, access constraints,
and blockers.

No CAPTCHA, authentication, private endpoint, rate limit, or robots/access control may be bypassed.
Third-party disclosure data that requires a licensed feed must use the existing licensed local
archive path and remain outside Git. Official pages without an exact timestamp or stable compliant
machine feed remain excluded and are not assigned guessed publication times.

## Timestamp And Storage Rules

Only `PublicationTimestampQuality.EXACT` can enter the reaction-ready corpus. `DATE_ONLY` and
`UNKNOWN` remain visible in coverage and exclusions but are never converted to midnight, market
open, or another guessed time. Offset-bearing timestamps are normalized to UTC; a source timezone
may be used only when its semantics are explicit.

Full text is retained only for `FULL_TEXT_ALLOWED`. The Rosneft RSS is handled as
`EXCERPT_ALLOWED`, so only the issuer-provided excerpt is stored. `METADATA_ONLY` and `UNKNOWN`
policies do not become reaction-ready text records.

## Controlled Workflow

Generate the source audit:

```bash
uv run python -m apps.cli.audit_historical_sources
```

Run the existing bounded feed adapter. The first live run must use at most 24 months and a limit no
greater than 100; smoke runs use at most 10. The canonical Rosneft URL includes its trailing slash.

```bash
uv run python -m apps.cli.backfill_historical_news \
  --feed-url https://www.rosneft.com/press/releases/rss/ \
  --source-kind ISSUER_RSS \
  --source-code ROSNEFT_PRESS_RELEASES_RSS \
  --storage-policy EXCERPT_ALLOWED \
  --source-timezone Europe/Moscow \
  --from 2024-08-11 --to 2026-08-11 \
  --limit 10 --max-pages 1 --dry-run --match-instruments
```

Repeat without `--dry-run` after reviewing the counts. Rerunning an unchanged range must produce
zero newly imported rows. Import runs retain separate failure/retry audit records, and corrections
are linked without deleting the original candidate.

Run deterministic matching and analysis, then inspect the bounded security and IMOEX windows:

```bash
uv run python -m apps.cli.prepare_reaction_ready_corpus \
  --from 2026-06-01 --to 2026-06-30 \
  --source-codes ROSNEFT_PRESS_RELEASES_RSS \
  --tickers ROSN --limit 100
```

Each planned request uses one-minute candles, three calendar days of pre-event safety, and seven
calendar days of post-event safety. This covers the 5/15/30/60 minute pre-event features and
1/5/15/30/60 minute labels across weekends and exchange closures without downloading unbounded
history. Use the existing `backfill_candles` and `backfill_benchmark` commands for each plan, then
the existing `compute_abnormal_reactions` command. Missing security data, missing IMOEX data,
non-trading events, and other failures remain explicit classifications; candles are never invented.

Build the canonical corpus with the import run that represents the controlled acquisition:

```bash
uv run python -m apps.cli.build_reaction_ready_corpus \
  --from 2026-06-01 --to 2026-06-30 \
  --source-codes ROSNEFT_PRESS_RELEASES_RSS \
  --tickers ROSN --limit 100 \
  --ingestion-run-id <UUID>
```

The command reruns deterministic matching and analysis, invokes the existing
`ml-feature-dataset-v1` builder, and writes:

- `artifacts/reaction-ready-corpus-v1/manifest.json`
- `artifacts/reaction-ready-corpus-v1/coverage.json`
- `artifacts/reaction-ready-corpus-v1/corpus.jsonl`
- `artifacts/reaction-ready-corpus-v1/exclusions.jsonl`
- `artifacts/reaction-ready-corpus-v1/ml-feature-dataset-v1/`

The canonical JSONL reuses the ML feature row: `features_available_at_publication` is the existing
feature payload and `labels` is the existing label payload. The event object is a compact versioned
summary, not a second feature schema.

## Funnel And Exclusions

The manifest reports a cumulative funnel:

```text
discovered -> validated -> imported -> EXACT -> matched
-> market-data-ready -> reaction-ready -> feature-ready
```

Every stage contains its count and loss percentage from the previous stage. Coverage is grouped by
source, ticker, primary event type, month, year, timestamp quality, matching status, and label
horizon. Exclusions include `DATE_ONLY`, `UNKNOWN_TIMESTAMP`, `UNMATCHED`, `AMBIGUOUS`,
`NO_EVENT_ANALYSIS`, `SECURITY_MARKET_DATA_MISSING`, `IMOEX_DATA_MISSING`, `NO_VALID_REACTION`,
`STORAGE_POLICY`, and `SOURCE_ERROR` where applicable. SBER/SBERP shared issuer aliases remain
ambiguous unless the text contains instrument-specific evidence.

## Research Readiness

The operational guideline is:

| Real feature rows | Status |
| ---: | --- |
| under 100 | `NOT_READY` |
| 100-499 | `PILOT_ONLY` |
| 500-999 | `BASELINE_EXPERIMENT_READY` |
| 1000 or more | `BASELINE_TRAINING_READY` |

This is a project policy, not a statistical guarantee. The report emits `LOW_DIVERSITY` when the
corpus is concentrated in a single ticker, event type, or month. Row count alone is not sufficient.

No model training, scaler fitting, feature selection, backtest, trading signal, rules/AI hybrid, or
mass Qwen analysis belongs in this pipeline. Live smoke is separate from CI, and all generated or
licensed data stays under the gitignored `artifacts/` tree.
