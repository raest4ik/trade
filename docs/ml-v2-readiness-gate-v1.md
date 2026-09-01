# ML v2 readiness gate v1

This document fixes the canonical readiness gate for the next event predictive ML
experiment. It is a readiness audit only: no model is trained, no model config is
changed, no backtest is run, and no future holdout outcomes are read.

## Existing Contract Differences

Earlier readiness checks used related but different scopes:

- `exact-dataset-readiness-audit-v1` accepted an issuer baseline when issuer
  feature-ready rows were at least 500, issuer tickers were at least 5, and
  issuer UNKNOWN rate was at most 50%.
- `event-market-dataset-v2` reports model data readiness only when exact
  feature-ready rows are at least 500 and unique tickers are at least 10.
- `exact-event-predictive-baseline-v1` already observed its TEST metrics and
  predictions, so that TEST can no longer serve as a fresh final test or tuning
  surface for ML v2.

The ML v2 gate keeps the stricter ticker-diversity requirement and applies it to
the issuer-originated strict-EXACT cohort, because the next experiment is meant
to test issuer/event information rather than exchange-originated notice effects.
This is an explicit methodological canonicalization for ML v2, not a hidden
refactor of the earlier gates.

## Canonical Cohort

`ISSUER_ORIGINATED_STRICT_EXACT_HISTORICAL_FEATURE_READY`

Rows must satisfy all of the following:

- issuer-originated official source;
- exact publication timestamp confirmed;
- unique deterministic instrument/ticker attribution;
- publication date before the future holdout start, `2026-08-11`;
- point-in-time pre-event market features available at or before publication;
- fixed target/reaction coverage available for the audited horizon.

Exchange-originated and MOEX risk events remain separate control families and
are not mixed into the issuer cohort.

## Gate Criteria

The fixed gate for `READY_FOR_CONTROLLED_ML_V2` is:

- issuer feature-ready rows >= 500;
- unique issuer tickers >= 10;
- issuer semantic UNKNOWN rate <= 50%;
- top-1 issuer ticker share <= 50%;
- top issuer source-family share <= 50% and source-family HHI <= 50%;
- primary 15m target coverage >= 95%;
- deterministic replay PASS;
- leakage audit PASS;
- future holdout untouched;
- old baseline TEST marked `OBSERVED_DO_NOT_TUNE_ON`.

If any criterion fails, the manifest must emit exactly one final decision from
the machine-readable enum and name one main blocker plus secondary blockers.

## Old TEST And Future Holdout

The `exact-event-predictive-baseline-v1` TEST is observed historical evidence.
It may be cited as the v1 result, but ML v2 must not select architecture,
features, thresholds, source inclusion, or model families from its TEST metrics
or predictions.

Future events with publication date `>= 2026-08-11` remain unobserved. The
readiness audit may count future metadata only when already present in manifests,
but it must not read target values, target distributions, predictions, returns,
or model metrics for future events.
