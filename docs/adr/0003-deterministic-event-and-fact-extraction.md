# ADR 0003: Deterministic Event And Fact Extraction

## Status

Accepted

## Context

The project needs explainable labels for corporate news before adding any
predictive or AI-assisted analysis. The first version must be reproducible,
auditable, and safe to join with saved instrument matches and market reaction
labels.

This phase still excludes LLM calls, ML models, embeddings, sentiment analysis,
impact scoring, fuzzy matching, real-time processing, and trading automation.

## Decision

Add an `events` module with deterministic rule version `event-rules-v1`.

The analyzer classifies supported corporate event types and extracts financial
facts using explicit regex rules. Saved results are versioned by `news_id` and
`analysis_version`; rerunning the same version replaces previous rows. Future
rule versions can coexist with old analyses.

## Stored Shape

`news_event_analyses` stores the parent analysis row, status, primary event type,
created time, and analyzed time.

`detected_events` stores event type, confidence metadata, matched rule id,
evidence text, and character offsets.

`extracted_financial_facts` stores metric, raw value, normalized value, unit,
currency, scale, period fields, fact role, comparison type, change direction,
change value, confidence metadata, evidence text, character offsets, extractor
version, and matched rule id.

## Statuses And Unknowns

Unknown or incomplete extraction is explicit:

- `NO_EVENT_FOUND` means no event and no numeric financial facts were found.
- `AMBIGUOUS` means more than one event type matched.
- `PARTIAL` means extraction found usable data but at least one numeric fact has
  an unknown metric.
- `UNKNOWN` enum values preserve incomplete period, comparison, role, or event
  information without inventing labels.

## API

`POST /api/v1/news/{news_id}/analyze-event` runs the deterministic analyzer for a
stored news item and persists the result.

`GET /api/v1/news/{news_id}/event-analysis` reads the saved result. The optional
`debug=true` response includes rule ids and counts, but not stack traces,
secrets, external prompts, or raw execution internals.

## Dataset Export

The CLI command
`uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl`
exports JSONL records linking stored news metadata, event analysis, saved
instrument matches, and saved market reactions.

Raw content is excluded by default and can be included explicitly with
`--include-raw-content`.

## Consequences

The first rule set is conservative and explainable, but it will miss or partially
extract some real-world wording. That is preferable to silent model-dependent
behavior at this stage. Later versions can add more deterministic patterns,
entity-aware rules, or separate model outputs while keeping rule versions and
dataset exports reproducible.
