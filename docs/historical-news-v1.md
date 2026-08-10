# Historical Corporate News Acquisition v1

## Scope

This subsystem imports legally obtained historical corporate news through bounded source
adapters. Adapters return source items and never write to the database. The ingestion service
records an import run, stages every candidate, validates provenance and storage rights, and only
then promotes eligible candidates to `NewsItem`.

It does not crawl websites, scrape e-disclosure.ru, bypass access controls, discover private
APIs, or provide credentials for commercial archives.

## Sources

Supported source kinds are `LOCAL_ARCHIVE`, `MANUAL_RESEARCH`, `ISSUER_JSON`,
`DISCLOSURE_ARCHIVE`, `INTERFAX_DISCLOSURE`, `ISSUER_RSS`, and `ISSUER_ATOM`.

`LocalArchiveNewsSource` consumes strict `historical-news-source-v1` JSONL. The Interfax-named
adapter is provider-neutral and only parses a locally supplied, licensed JSONL archive. It has no
network endpoint. `IssuerFeedNewsSource` accepts issuer-owned HTTPS RSS/Atom URLs and implements
timeouts, bounded retries, conditional ETag/Last-Modified requests, a User-Agent, rate limiting,
and a hard item bound.

No live licensed source is configured in the repository. Licensed dumps and credentials must
stay outside Git, for example under the ignored `artifacts/private-sources/` directory.

## Provenance And Staging

`historical_news_sources` stores immutable source identity, kind, timezone, URL, and storage
policy. `historical_news_import_runs` records the requested range, status, counts, timestamps, and
error. `historical_news_candidates` stores source item identity, URL, title, original timestamp,
parsed UTC timestamp, timestamp quality, fetch time, import run, permitted content/hash,
correction relation, validation status, and promoted news ID.

The unique `(source_id, source_item_id)` key makes reruns idempotent. Equal content hashes from
different sources are flagged as exact duplicates but neither record is deleted. Corrections are
linked only when the source explicitly supplies `corrects_source_item_id`; no fuzzy or NLP guess
is made and the original remains intact.

## Timestamp Trust

Aware timestamps are converted to UTC and marked `EXACT`. Naive timestamps become `EXACT` only
when that specific source has an explicit IANA timezone. Date-only values remain `DATE_ONLY`; UTC
midnight is a technical storage value, not an imputed publication time. Naive timestamps without
a trustworthy source timezone remain `UNKNOWN` and are not promoted.

**EXACT timestamps are mandatory for market-reaction labels. DATE_ONLY/UNKNOWN can be used for
NLP research but not reaction training.** Existing reaction code enforces this independently;
this subsystem does not calculate reactions during import.

## Content Storage Policy

- `FULL_TEXT_ALLOWED`: full content may be staged and promoted.
- `EXCERPT_ALLOWED`: content is stored only when the item explicitly marks it as an excerpt.
- `METADATA_ONLY`: third-party text is never stored or copied into `NewsItem.raw_content`.
- `UNKNOWN`: no content is stored until rights are clarified.

The most restrictive source/item policy wins. Metadata-only candidates remain auditable but are
not promoted because `NewsItem` requires content.

## Matching And Reaction Readiness

Import can optionally call the existing deterministic instrument matcher. Exported records are
`reaction_ready` only when they have a promoted `NewsItem`, an `EXACT` timestamp, at least one
instrument match, and no ambiguous match. This flag does not claim that candles or benchmark data
exist and does not create a market-reaction label.

## Commands

```bash
uv run python -m apps.cli.import_historical_news --help
uv run python -m apps.cli.backfill_historical_news --help
uv run python -m apps.cli.export_historical_corpus --help
uv run python -m apps.cli.historical_news_stats --help
```

Use `--dry-run` before imports. Both import commands require an explicit `--from`, `--to`, and a
bounded `--limit`. The corpus exporter emits `historical-news-corpus-v1`; content is omitted by
default. `--reaction-ready` filters to the strict subset. Stats include source, quality, status,
ticker, year, month, matching, ambiguity, and reaction-readiness counts.

## Limitations

The generic feed parser intentionally supports ordinary RSS 2.0 and Atom feeds, not arbitrary
vendor XML. Feed cache validators are process-local in v1. `ISSUER_JSON` uses the local JSONL
contract; a live JSON HTTP adapter is not included. A licensed disclosure client can be connected
later through `HistoricalNewsSourceClient` once its official contract and storage rights are
available.
