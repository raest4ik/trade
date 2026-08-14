# Event corpus and event-market dataset v1

## Purpose

This work restores the event-driven path: a real official issuer event is the
predictive unit, frozen event semantics describe it, market context contains only
information available before publication, and reaction targets are stored separately.
It does not create a row for an arbitrary ticker-day.

The previous market-only research remains a frozen negative control:
`NO_PREDICTIVE_SIGNAL / NO_DEV_SIGNAL`. A future controlled experiment may compare
market context only, event features only, and their combination. This change does not
train or select a model and does not inspect observed or future market holdouts.

## Sources and rights

`event-source-registry-v1` is generated for the complete 315-share TQBR/RUB universe.
It registers only verified URLs; unknown issuers are marked
`NO_OFFICIAL_SOURCE_FOUND` and no URL is guessed.

The implemented source families are:

- Rosneft issuer RSS already present in the real exact-timestamp corpus.
- Yandex issuer RSS plus bounded, explicit official archive year pages.
- NOVATEK issuer press-release archive with bounded numbered pages.

The Yandex archive is linked to `YDEX` only from 24 July 2024, when the new IPJSC
shares began trading under that ticker. Older archive rows belong to the former
`YNDX` issuer/security context and are explicitly excluded rather than guessed. The
official mapping basis is the issuer's 30 July 2024 financial release and FAQ.

All acquisition is public, HTTPS, zero-cost, bounded, and unauthenticated. It does not
bypass paywalls, CAPTCHA, robots controls, authentication, or rate limits. Only source
metadata, title-derived features, content hashes, and provenance are emitted. Raw full
text is not redistributed. Local response cache pages and generated datasets live
under `artifacts/` and remain gitignored for private internal research.

The archive collector uses bounded retries, payload limits, deterministic limits,
canonical URLs, SHA-256 cache metadata, and resumable page caching. A cached page is
accepted only when its URL and digest match. A structurally empty page fails closed.
Deduplication distinguishes duplicate source records, duplicate canonical URLs,
same-title issuer/day repetitions, and updated publications.

## Time and leakage policy

Publication quality is not inferred:

- confirmed source timestamps remain `EXACT` and use `EXACT_INTRADAY` reactions;
- archive dates without a confirmed time remain `DATE_ONLY` and use
  `DATE_SAFE_DAILY` reactions;
- unverified times are excluded.

For an exact event, the context cutoff must be strictly before its timestamp. For a
date-only event on date `D`, the market feature `feature_as_of` must be strictly before
`D`. Its baseline is the last common security/IMOEX session before `D`; its target is
the first common session after `D`. The close on `D` is never used. Security and IMOEX
returns use the same dates and abnormal return is their difference.

`features.jsonl` contains only `X_event` and pre-event `X_market`.
`targets.jsonl` contains post-event `Y_reaction`. Event IDs reconcile one-to-one and
the leakage audit rejects reaction or future-value fields in features. Group metadata
records issuer/date clusters for a future grouped temporal split.

## Frozen contracts

The builder verifies the frozen Rules v3 fingerprint and the frozen Qwen prompt and
schema hashes before running. Rules, facts, ontology, Qwen configuration, gold labels,
and live collector behavior are unchanged. Qwen is not invoked by this workflow.

The T-Invest 315-share dataset supplies identity, pre-event market context, and reaction
prices under `PRIVATE_INTERNAL_USE_CONFIRMED`. MOEX ISS data is not used. No token is
read or printed by the event builder. There is no order, account mutation, paper
trading, sandbox trading, backtest, or BUY/SELL capability.

## Rebuild

The existing PostgreSQL database and gitignored T-Invest artifacts must be available.
The range and limits are always explicit:

```powershell
uv run python -m apps.cli.build_event_market_dataset `
  --date-from 2022-01-01 `
  --date-to 2026-08-12 `
  --per-source-limit 2000
```

Generated output is written to
`artifacts/event-market-predictive-dataset-v1/`, including the source registry,
features, physically separate targets, exclusions, provenance, schemas, and manifest.
Reruns reuse verified cache pages and produce the same dataset SHA for the same inputs.

## Readiness

Volume thresholds are diagnostic: below 100 is `EVENT_DATA_NOT_READY`, 100-249 is
`EVENT_PILOT_READY`, 250-499 is `EVENT_BASELINE_EXPERIMENT_READY`, and at least 500 is
`EVENT_BASELINE_TRAINING_READY`. Independently, fewer than ten tickers emits
`LOW_EVENT_TICKER_DIVERSITY`. A volume status is not permission to train and never
means trading-ready.
