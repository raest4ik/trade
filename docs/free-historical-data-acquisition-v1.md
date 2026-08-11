# Free Historical Data Acquisition v1

## Decision

`DATA_BUDGET=ZERO` is a permanent project constraint. The acquisition system may use a
source only when access, data use, provenance, timestamp semantics, identity, and storage
policy are all verified without payment. A paid source is not a fallback.

The current honest answer is:

- 40 free REAL records are verified and all 40 have source-provided EXACT timestamps;
- 35 are unambiguously matched (19 ROSN and 16 YDEX);
- 26 have existing reaction labels and 21 have existing `ml-features-v1` rows;
- no new compliant source was found during this audit, so the PR pilot imports zero rows;
- large official archives exist, but the observed additional volume is either date-only or
  blocked by unverified automation policy/access stability;
- the project is `NOT_READY` for predictive ML.

Targets such as 100, 500, 1000, and 5000 are planning thresholds, not observed facts.

## Hard Safety Policy

The project does not use paid APIs, datasets, archives, feeds, subscriptions, licenses,
card-backed trials, or commercial-only endpoints. It does not bypass authentication,
paywalls, CAPTCHA, robots directives, rate limits, access controls, or terms. It does not
reverse engineer private APIs or perform prohibited mass scraping.

Interfax, Cbonds, and the MOEX/NSD corporate-information product are recorded only as
`REJECTED_PAID`. No integration or purchase path exists for them.

Internet Archive capture time, crawler time, and `first_seen_at` are never publication time.
`EXACT` requires a date, clock time, and verified timezone directly from source metadata.

## Source Audit

The machine-readable audit is generated at
`artifacts/free-historical-data-v1/source-audits.json`. The current status summary is:

| Source | Ticker | Status | Evidence and blocker |
|---|---:|---|---|
| Rosneft official RSS | ROSN | `COMPLIANT_EXACT` | 20 issuer items, RFC 822 offset, excerpt storage |
| Yandex IR RSS | YDEX | `COMPLIANT_EXACT` | 20 issuer items, RFC 822 offset, excerpt storage |
| Sberbank IR | SBER/SBERP | `UNSTABLE` | controlled access timeout; no stable exact machine contract |
| Gazprom press archive | GAZP | `UNSTABLE` | exact clocks visible back to 2002; direct access and automation policy unverified |
| LUKOIL press archive | LKOH | `DISCOVERY_ONLY` | 248 pages observed; date-only and automation policy unverified |
| NOVATEK press archive | NVTK | `DISCOVERY_ONLY` | deep archive, date-only |
| Yandex yearly archive | YDEX | `DISCOVERY_ONLY` | reaches 2005, date-only |
| T-Bank news | T | `UNSTABLE` | JS archive; exact machine contract unverified |
| VTB IR | VTBR | `UNSTABLE` | controlled access timeout; timestamp contract unverified |
| Nornickel IR archive | GMKN | `DISCOVERY_ONLY` | 576 items previously observed; date-only and policy unverified |
| MOEX ISS research | universe | `DISCOVERY_ONLY` | delayed market data is not an issuer-news corpus |
| MOEX corporate information | universe | `REJECTED_PAID` | subscription product |
| Interfax disclosure | universe | `REJECTED_PAID` | no accepted free documented machine endpoint |
| Cbonds | universe | `REJECTED_PAID` | commercial API/subscription |
| Open dataset research | universe | `DISCOVERY_ONLY` | no candidate passed license, provenance, mapping, and time checks together |

SBERP shares the Sberbank issuer source. Matching ambiguity remains explicit.

MOEX ISS remains the existing market-data source. The official ISS documentation says
delayed market data is available without subscription, while reuse beyond familiarization
may require a contract. No MOEX endpoint was accepted here as a free corporate-news archive.
Reaction and feature semantics are unchanged.

## Verified And Estimated Volume

`source-volume.json` separates evidence classes:

- `VERIFIED`: 40 available, 40 EXACT, 0 DATE_ONLY. These are the 20 ROSN and 20 YDEX
  records already present locally; rerunning their feeds creates duplicates, not new rows.
- `ESTIMATED`: 8,056 possible archive records. The estimate consists of roughly 5,000
  Gazprom exact-time pages, 2,480 LUKOIL date-only pages, and 576 Nornickel date-only pages.
- `eligible`: 40 available and 40 EXACT. Estimated blocked records are not counted here.

The 5,000 Gazprom figure is explicitly an estimate from visible archive depth, not a verified
count or an authorization to crawl. The archive becomes eligible only after direct access,
robots/terms, stable identity, storage policy, and timezone semantics are all verified.

## Provider Architecture

The existing `HistoricalNewsSourceClient` port remains the ingestion boundary. The new
providers implement that contract:

- `IssuerFeedNewsSource` for RSS/Atom;
- `SitemapArchiveNewsSource` for bounded sitemap discovery and metadata sampling;
- `PaginatedIssuerArchiveNewsSource` for bounded issuer pagination;
- `PublicJsonNewsSource` for documented public JSON APIs.

All deep providers use `BoundedHttpClient`: credential-free HTTPS, domain allowlist,
concurrency at most two, a minimum request interval, bounded retries, response-size limits,
and explicit page/item caps. Parsers are injected so unit tests use fixtures and
`httpx.MockTransport`; CI performs no live HTTP.

Stable identity is `source_code + source_item_id`; content hashes remain separate. Existing
database uniqueness and ingestion transactions provide idempotency and duplicate prevention.
`fetched_at` is the separately stored first-seen observation and never changes
`source_published_at`.

## Commands

Generate all local gitignored audit and readiness artifacts:

```bash
uv run python -m apps.cli.historical_news_stats
uv run python -m apps.cli.build_free_historical_data_report
```

Run a bounded dry-run against an accepted source:

```bash
uv run python -m apps.cli.collect_live_news \
  --source-code ROSNEFT_PRESS_RELEASES_RSS \
  --from 2026-08-01 \
  --to 2026-08-12 \
  --limit 10 \
  --dry-run
```

Remove `--dry-run` only for an operator-approved import. `--match-instruments` invokes the
existing deterministic matcher. The command does not invoke Rules, Qwen, reactions, features,
or predictive ML. The accepted source registry prevents arbitrary URLs.

## Pilot And Selection

The PR pilot is intentionally empty because no new accepted source was discovered and all 40
accepted feed items already exist. A future pilot is capped at 200 new REAL rows and ordered by
source, publication time, and stable source item ID. Selection never reads Rules/Qwen output,
event class, returns, prices, or volumes. Existing IDs are excluded before the limit.

Only REAL + EXACT + unambiguous deterministic matches may enter the existing reaction pipeline.
DATE_ONLY, unknown timezone, ambiguous, unmatched, weekends/closures without candles, and
metadata-only records remain excluded. No fuzzy ticker guessing is introduced.

## Readiness And Growth

Current cumulative state:

- REAL: 40
- REAL EXACT: 40
- matched / ambiguous / unmatched: 35 / 0 / 5
- reaction-ready: 26
- feature-ready: 21
- ticker distribution: ROSN 19, YDEX 16 among matched rows
- source distribution: 20 Rosneft RSS, 20 Yandex RSS
- observed publication range: 2025-06-20 through 2026-08-11

The count gate is `NOT_READY` below 100 feature rows, `PILOT_ONLY` at 100-499,
`BASELINE_EXPERIMENT_READY` at 500-999, and `BASELINE_TRAINING_READY` at 1000 or more.
Ticker, source, time, and market-regime diversity can lower readiness. Current ticker diversity
is also insufficient.

The zero-cost path is:

1. keep bounded incremental polling of accepted issuer RSS feeds;
2. periodically repeat policy and timestamp audits for the blocked official archives;
3. add issuers only after every acceptance field is proven;
4. retain the permitted free history and build an own archive over time;
5. periodically regenerate readiness reports.

There are not enough repeated collection observations to estimate monthly growth or time to
100, 500, or 1000. These values remain `UNKNOWN`; no paid fallback is proposed.

## NLP And ML Freeze

This work does not change `event-rules-v3`, financial facts, ontology, the Qwen prompt/schema/
model/config, reaction labels, or `ml-features-v1`. It creates no rules v4, hybrid, ensemble,
backtest, or predictive model. The observed Batch 004 holdout is not used for tuning.
