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

Add an `events` module with deterministic event rule version `event-rules-v1`
and financial fact extractor version `financial-facts-v1`.

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

Raw value, scale, and normalized value are all retained. For example,
`12,5 млрд рублей` is stored as raw value `12.5`, scale `BILLION`, normalized
value `12500000000`, and currency `RUB`. This prevents later evaluation code
from losing whether a value was reported in millions, billions, percent, or
percentage points.

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

## Why Evidence Spans

Every event and financial fact keeps `evidence_text`, `start_position`, and
`end_position` such that slicing the original raw content recreates the evidence.
This makes deterministic labels auditable and lets future dataset review tools
show exactly why a label exists without mutating the source news text.

## Ambiguity

If several event types match one news item, all detected events are stored and
the analysis can be marked `AMBIGUOUS`. Unknown metric, missing period, and weak
local proximity are preserved explicitly instead of being hidden behind a forced
classification.

## Dataset Export

The CLI command
`uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl`
exports JSONL records linking stored news metadata, event analysis, saved
instrument matches, and saved market reactions.

Raw content is excluded by default and can be included explicitly with
`--include-raw-content`.

The export is intended to join event/fact labels with saved instrument matches
and saved market reaction labels. It must not calculate future market data while
exporting; this avoids look-ahead leakage into future training or evaluation
datasets.

## Future ML And LLM Use

A later compact ML model can be added as a separately versioned extractor whose
outputs coexist with `event-rules-v1`. LLM usage should be a slower second
review contour for low-confidence or ambiguous rows, not a silent replacement
for deterministic labels. Any model or LLM output should store its own version,
evidence, confidence metadata, and provenance so it can be compared against the
rule-based baseline.

## Consequences

The first rule set is conservative and explainable, but it will miss or partially
extract some real-world wording. That is preferable to silent model-dependent
behavior at this stage. Later versions can add more deterministic patterns,
entity-aware rules, or separate model outputs while keeping rule versions and
dataset exports reproducible.
