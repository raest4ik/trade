# Fresh Real Annotation Corpus v1

## Purpose

`ru-corporate-events-real-batch-003-gold-v1` is now an
`OBSERVED_EVALUATION_SET`. Its Rules and Qwen metrics have been inspected, so it must not be used
for keyword tuning, UNKNOWN-to-OTHER fallback design, Qwen prompt tuning, threshold selection,
hybrid design, or model selection. It remains useful only as a historical benchmark.

This phase prepares `annotation-batch-004` from new issuer records. It does not change or evaluate
`event-rules-v2`, `financial-facts-v2`, the frozen Qwen prompt/schema/model, reaction semantics, or
ML feature semantics. No event label is generated automatically.

## Source and selection contract

The approved live sources remain:

- `ROSNEFT_PRESS_RELEASES_RSS`;
- `YANDEX_IR_PRESS_RELEASES_RSS`.

Both are issuer-owned HTTPS RSS feeds with stable item identities, RFC 822 exact timestamps carrying
a numeric UTC offset, and an `EXCERPT_ALLOWED` storage policy. The bounded audit still covers nine
issuer configurations; the other sources remain blocked, date-only, unstable, or without a usable
storage policy. A calendar date is never promoted to `EXACT`.

Selection depends only on source, an explicit date range, stable publication/source-item ordering,
ticker linkage, and a limit of at most 100. Previous Batch 001/002/003 news IDs, source identities,
and content hashes are excluded. Rules output, Qwen output, disagreement, returns, abnormal returns,
future volume, and market movement are not selection inputs.

The approved one-page feeds expose only 20 items each. When fewer than 50 non-overlapping records
remain, the manifest reports `SOURCE_DEPTH_BLOCKER`; records are never invented to satisfy the
target.

## Annotation and storage

Generated data is written to `artifacts/fresh-real-corpus-v1/`, with the human export mirrored at
`artifacts/corpus-quality-v1/annotation-batch-004.jsonl`. These paths are gitignored. Every record is
REAL, `EXACT`, `DRAFT`, `UNASSIGNED`, and `is_gold=false`. Only the issuer-permitted excerpt is
included. Original timestamp text and timezone provenance are retained while `published_at` is
stored in UTC.

The human JSONL contains no Rules/Qwen prediction, human label, future return, reaction label, or
future volume. Matching state is retained as provenance. Only unambiguous matched records can later
be reaction-ready; this PR does not run market backfill.

## Frozen temporal split

Before any future extractor or prompt work, Batch 004 is sorted by publication time and frozen into:

- older approximately 70%: `DEVELOPMENT`;
- newer approximately 30%: `FRESH_HOLDOUT`.

The canonical assignment list has a reproducible SHA-256 in `split-manifest.json`. Repeating the
same bounded build is byte-stable for annotation, coverage, split, and manifest artifacts.
Development may
be human-reviewed and used in a later PR for Rules v3, ontology work, Qwen prompt v1, or a hybrid
prototype. None of that occurs here.

The fresh holdout remains blind. This PR does not run Rules, Qwen, hybrid evaluation, or model-error
analysis on it and emits no holdout predictions. Schema, provenance, counts, and split integrity are
the only holdout checks.

## Limitations

The available source depth is small, the issuer universe remains concentrated in ROSN and YDEX,
and event diversity is unknown until independent human review. No event distribution may be
claimed from this unlabeled batch. Batch 004 is not gold, no ML model is trained, and no backtest,
signal, paper-trading, voting, fallback, or ensemble path is introduced.
