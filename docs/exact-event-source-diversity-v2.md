# Exact Event Source Diversity v2

## Purpose

`exact-event-market-dataset-v2` expands issuer and source diversity while preserving the
immutable v1 corpus. It is a data-acquisition and audit release. It does not train, evaluate,
or tune a predictive model and does not change the frozen event rules, financial facts, or
Qwen configuration.

The primary targets are at least ten exact-timestamp tickers and at least 250 feature-ready
rows. Total event count is secondary. Existing MGNT rows are retained without downsampling.

## Frozen Parent

The builder verifies all six v1 manifest hashes and the frozen counts before processing a new
event. Existing v1 event and feature rows are copied byte-equivalently as the v2 prefix. A
duplicate whose source identity or canonical URL already exists in v1 is excluded and reported.
The v1 artifact is never overwritten.

## Official Zero-Cost Sources

The v2 adapters use only bounded, unauthenticated resources loaded by official public pages:

- X5 official WordPress REST news route (`date_gmt`);
- VK official public Next.js page state (`publications.pub_date`);
- T-Bank official public newsroom API (`publishedAt`);
- Novabev official embedded application state (`activeFrom`), excluding source-local midnight
  placeholders;
- Moscow Exchange official RSS shareholder notices (`pubDate`);
- Moscow Exchange official RSS issuer notices for SMLT (`pubDate`).
- Moscow Exchange official RSS issuer notices for VTBR (`pubDate`).

Every adapter uses HTTPS, an exact host allowlist, bounded item limits, bounded retries, a
payload-size limit, and a content-addressed local cache. Redirects to another host fail closed.
No paid feed, authentication, private endpoint, CAPTCHA bypass, or rate-limit bypass is used.
An unavailable source is recorded as `FAILED_CLOSED`; it does not weaken TLS verification or
abort acquisition from independent official sources.

An event is `EXACT` only when the source provides a publication time and deterministic timezone.
Retrieval time, HTTP headers, and file timestamps are never publication timestamps. Unknown
timezone or timestamp semantics are rejected rather than guessed.

## Frozen Reaction Semantics

New rows reuse the existing exact-event pipeline and T-Invest read-only production candles.
Security and IMOEX use the same effective window. During-session targets start at the next
complete minute after publication. There is no interpolation, forward fill, synthetic candle,
or MOEX ISS substitution. Pre-open, after-close, non-trading-day, and incomplete-data cases
remain fail closed.

Events on or after `2026-08-11` remain future holdout metadata. Their targets, abnormal returns,
correlations, and performance are not read or exported. The hard holdout guard runs before any
v2 artifact is written.

## Diagnostics

The manifest reports event, reaction, and feature counts; ticker, issuer, and source-family
concentration; session and horizon coverage; deterministic clusters and duplicate exclusions;
UNKNOWN event coverage; and event-type diversity. Feature-ready loss is reconciled into explicit
reason buckets, including market-history warmup and missing pre-event context.
The registry remains a 315-instrument audit and reports exact-capable source-family counts and
sanitized acquisition status for each attempted collector.

UNKNOWN diagnostics use publication-time metadata and frozen event output only. Market outcomes
are not used to propose NLP changes. Rules v3 and Qwen remain frozen, and no hybrid logic is
introduced.

`READY_FOR_EXACT_BASELINE_EXPERIMENT` requires at least 100 feature-ready rows, at least ten
feature-ready tickers, and at least three tickers with ten or more feature-ready rows. This is a
data-readiness status, not a model result or permission to trade.

## Safety

This package has no training, A/B/C evaluation, backtest, paper-trading, order, BUY/SELL, account
mutation, or money-movement capability. All market access is read-only, and external data cost is
zero rubles.
