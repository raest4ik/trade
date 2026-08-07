# ADR 0004: Event Extraction Evaluation Datasets

## Status

Accepted

## Context

The deterministic event and financial-fact extractor needs regression checks
that are independent from live market data and external AI services. A plain
export of predictions is useful for inspection, but it cannot tell whether new
rule changes improve or degrade extraction quality.

## Decision

We store reviewed gold labels in JSONL records with schema version
`event-gold-v1`. Each line is one news item and includes the original `news_id`,
`published_at`, `raw_content_hash`, split, review status, optional raw content,
predicted events/facts for reviewer context, and reviewed gold events/facts with
evidence spans.

Imported datasets are persisted in first-class tables:
`evaluation_datasets`, `evaluation_examples`, `gold_events`,
`gold_financial_facts`, and `evaluation_runs`. Dataset imports are idempotent by
source file hash. Temporal train/validation/test split assignment is explicit
and stored on each example.

Evaluation runs re-run the current deterministic extractor against stored raw
news, match predicted facts to gold facts with deterministic one-to-one dynamic
programming, and persist metrics plus report artifacts. Thresholds live in
`config/evaluation_thresholds.toml`.

## Consequences

The project now has a reviewable contract for extraction quality before rules
change. The gold JSONL format can be versioned without rewriting historical
evaluation runs. CI can validate code paths without network access, while full
local smoke tests can run through Docker and PostgreSQL.
