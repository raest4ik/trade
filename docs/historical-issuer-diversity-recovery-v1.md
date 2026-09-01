# Historical Issuer Diversity Recovery v1

This audit answers whether the ML v2 issuer diversity blocker can be closed with at least three
new pre-2026-08-11 issuer tickers without weakening the canonical ML v2 readiness gate.

## Scope

The canonical cohort remains `ISSUER_ORIGINATED_STRICT_EXACT_HISTORICAL_FEATURE_READY`.
The ML v2 gate from `docs/ml-v2-readiness-gate-v1.md` is not changed.

This PR is audit-only. It does not train models, run backtests, tune Rules v3/Qwen/NLP, read future
holdout outcomes, or select sources using market returns/model performance.

## Current Gap

The current issuer cohort has 547 feature-ready rows across 7 issuer tickers. `MGNT` dominates with
319 rows, a 0.583181 share. If MGNT does not grow, at least 91 additional non-MGNT issuer rows are
needed to bring top-1 share to 0.50 or below. The cohort has 332 UNKNOWN rows, so at least 117
additional non-UNKNOWN issuer rows are needed to bring aggregate UNKNOWN rate to 0.50 or below.

These are independent blockers. Adding three tickers is necessary but not sufficient unless the new
rows also improve concentration and have factual semantic classifications.

## Source Classes

Prior zero-cost public issuer HTML/RSS discovery is treated as exhausted based on immutable evidence
from `timezone-verified-issuer-exact-source-discovery-v2` and
`issuer-exact-historical-diversity-expansion-v1`; those candidates are not re-mined unless a URL or
mechanism changes.

New acquisition classes reviewed by this audit:

- Interfax CRKI e-disclosure Gateway REST API:
  `https://e-disclosure.ru/poluchenie-informacii/shlyuz-api`
- Interfax CRKI e-disclosure FTP export:
  `https://www.e-disclosure.ru/poluchenie-informacii/vygruzka-na-ftp`
- MOEX Corporate Information Center:
  `https://www.moex.com/tsentr-korporativnoj-informatsii`
- Existing project credentials/connectors, capability-only scan with secret values never emitted.
- Existing local caches/artifacts from prior maturation/recovery passes.

The Interfax CRKI Gateway is the only reviewed public/provider-documented path that plausibly closes
the issuer disclosure gap, because it is an official/authenticated disclosure API with structured
events and archive access. It still requires licensed/test access, response-field verification for
publication timestamp and timezone semantics, and license review for storage/internal ML research
before any ingestion.

## Decision

The artifact decision is `PAID_OR_AUTHENTICATED_SOURCE_REQUIRED`.

The strict answer to "can we add at least three new issuer tickers right now without weakening the
gate and without touching future holdout?" is `NO`: the missing resource is licensed/authenticated
access to historical issuer disclosure data with publication-specific timestamp/timezone provenance.

The exact next action is to request test/licensed access to the Interfax CRKI e-disclosure Gateway,
verify publication timestamp/timezone fields and storage/internal-ML rights, then run a bounded
pre-2026-08-11 ingestion only if at least three new issuer tickers pass the source acceptance
contract.
