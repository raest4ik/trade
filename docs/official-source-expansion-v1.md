# Official Source Expansion v1

## Outcome

This phase expands the real issuer-owned corpus without changing event rules, financial-fact
rules, reaction semantics, ML feature semantics, or AI configuration. Source selection remains
independent of deterministic events, AI output, and all future market data.

The official Yandex IR RSS is accepted as a second reaction-ready source. The cumulative generated
snapshot is `reaction-ready-corpus-v3`; raw/live rows and generated artifacts remain gitignored.

## Acceptance gate

An official source is `REACTION_READY` only when all of these properties are evidenced:

- issuer ownership and public/legal access;
- stable item identity;
- `EXACT` publication timestamp;
- explicit timezone or UTC-offset semantics;
- usable content storage policy;
- bounded acquisition with explicit date range and limit.

Other audit statuses are `NLP_ONLY_DATE_ONLY`, `NLP_ONLY_UNKNOWN_TIME`, `ACCESS_BLOCKED`,
`LICENSED_SOURCE_REQUIRED`, `UNSTABLE_SOURCE`, and `REJECTED`. Time is never inferred from a page
date, an update timestamp, crawl time, or market data.

## LUKOIL

The official "Subscription to press releases" page exposes a public `Get RSS link` flow. RSS mode
does not require email or CAPTCHA. A normal form submission returned an issuer-owned RSS 2.0
endpoint, and its HTTPS form responded successfully.

The observed channel was empty. Therefore no item `guid`, fallback link identity, `pubDate`, UTC
offset, description contract, or historical depth could be verified. The archive page contains a
calendar date, an internal `LastUpdatedTime`, and `PublicationTime=null`; these fields do not prove
an exact publication timestamp or timezone.

LUKOIL status is `UNSTABLE_SOURCE`, not reaction-ready. The generated channel payload is not
committed as source configuration, and no time is guessed.

## NOVATEK

The public NOVATEK press archive and one official release page were inspected. The release exposes
a calendar date such as "Moscow, 24 July 2026", but no exact publication time or offset was found in
HTML metadata, OpenGraph, JSON-LD, RSS, or Atom.

NOVATEK status is `NLP_ONLY_DATE_ONLY`. It is excluded from reaction-ready acquisition and no time
imputation is performed.

## Yandex

The official IR page publishes:

```text
https://ir.yandex.ru/press-releases/news.rss
```

The bounded audit found 20 items with:

- issuer-owned HTTPS release links;
- unique issuer-owned GUIDs and links;
- RFC 822 `pubDate` values with a numeric `+0300` offset and second precision;
- title and issuer-provided HTML excerpt;
- a link to the full issuer release;
- one page and no pagination.

The locale-specific `.ru` host is deliberate. The neutral host content-negotiates to a different
English channel for non-browser clients; the explicit public `.ru` endpoint makes acquisition
deterministic without browser impersonation. The source is configured as
`YANDEX_IR_PRESS_RELEASES_RSS`, `ISSUER_RSS`, `EXACT`, and `EXCERPT_ALLOWED`. It reuses
`IssuerFeedNewsSource`; there is no Yandex-specific parser.

## Other official sources

The bounded audit covers SBER/SBERP, GAZP, LKOH, ROSN, NVTK, YDEX, T, VTBR, and GMKN. SBERP shares
the SBER issuer source and remains explicitly subject to matching ambiguity.

Only Rosneft RSS and Yandex IR RSS currently pass every reaction-ready gate. Other sources remain
date-only, inaccessible to the controlled client, or without a stable exact-time feed contract.
No third-party aggregator, private undocumented API, licensed dump, CAPTCHA bypass, authentication
bypass, or robots bypass is used.

## Controlled acquisition

The Yandex import used the generic feed client and the existing staging/import architecture:

```text
historical acquisition -> historical_news_candidates -> NewsItem
-> deterministic instrument matcher -> event-rules-v2 -> financial-facts-v2
-> YDEX/IMOEX minute candles -> reaction-v2-benchmark-adjusted
-> ml-feature-dataset-v1
```

The first live run was limited to ten items. After it succeeded, the limit was raised to 20 over an
explicit `2026-07-16..2026-08-11` range. The second run retained the first ten as duplicates and
imported only the remaining ten. A subsequent same-range rerun must be duplicate-only.

News selection uses source, date range, issuer, source order, and limit. It does not use event-rule
results, Qwen predictions, returns, abnormal returns, volume, or subjective "interestingness".

## Market windows

Each exact matched publication contributes a 60-minute pre-event context, a 60-minute post-event
label requirement, and a bounded safety interval. Overlapping intervals are merged, then split into
non-overlapping windows no longer than 14 days. Gaps are not bridged unless their safety intervals
overlap.

Market closures and unavailable candles are retained as exclusions. The pipeline does not invent
reaction values. Existing reaction and feature semantics are unchanged.

## Corpus v3

Generated files are written under `artifacts/reaction-ready-corpus-v3/`:

- `manifest.json`;
- `coverage.json`;
- `funnel.json` with overall, ticker, and source funnels;
- `source-audit.json`;
- `corpus.jsonl`;
- `ml-feature-dataset-v1/` artifacts.

The generated manifest reports source, ticker, event, and month distributions; label availability;
UNKNOWN rate; Batch 001 reaction count; frozen Batch 002 checksum; and separate readiness gates.
Synthetic and seed rows are never classified as REAL.

## Annotation Batch 003

When at least 20 eligible real rows exist, the generator writes
`artifacts/corpus-quality-v1/annotation-batch-003.jsonl`. Selection is deterministic across
ticker/source strata and publication order. It does not inspect future returns.

Every row is `DRAFT`, `UNASSIGNED`, and `is_gold=false`. Batch 003 is not imported as final gold.
Batch 002 remains byte-for-byte unchanged, and Qwen `OTHER` predictions are not reused as labels.

## Diagnostics and readiness

Warnings remain research diagnostics:

- `LOW_TICKER_DIVERSITY` when one ticker exceeds 70%;
- `LOW_EVENT_DIVERSITY` when one event type exceeds 70%;
- `HIGH_UNKNOWN_EVENT_RATE` when UNKNOWN exceeds 50%.

Fewer than 100 real feature-ready rows is `NOT_READY`. Even above row-count thresholds, dominant
UNKNOWN, low ticker/event diversity, or poor temporal coverage remains a blocker. Supported-looking
UNKNOWN rows may be recorded as `RULE_MISS_REVIEW_CANDIDATE`, but deterministic rules are not tuned
before human gold review.

No Qwen bulk run, hybrid inference, ML training, backtest, signal generation, or broker integration
is part of this phase.
