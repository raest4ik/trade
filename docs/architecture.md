# Architecture

## Layers

The application is a modular monolith split by feature and layer.

- `news.domain` owns the `NewsItem` entity, timestamp normalization, content hash
  calculation, and validation rules that are independent of frameworks.
- `news.application` contains news use cases and repository protocols.
- `news.infrastructure` contains SQLAlchemy table mappings and repository code.
- `news.presentation` contains FastAPI routes and Pydantic request/response
  schemas.
- `instruments.domain` owns `Instrument`, `IssuerAlias`, text normalization, and
  deterministic matching rules.
- `instruments.application` coordinates instrument creation, alias creation, and
  news-to-instrument matching.
- `instruments.infrastructure` stores instruments, aliases, seed data, and saved
  `NewsInstrumentMatch` rows.
- `instruments.presentation` exposes MVP endpoints for registry maintenance and
  matching.
- `market_data.domain` owns immutable OHLCV candle entities and import audit
  records.
- `market_data.infrastructure` owns the MOEX ISS HTTP adapter and SQLAlchemy
  candle storage.
- `market_data.presentation` exposes backfill, candle listing, and import audit
  endpoints.
- `reactions.domain` owns historical news market reaction labels.
- `reactions.application` calculates labels from saved news, saved instrument
  matches, and saved candles without rerunning the matcher.
- `events.domain` owns deterministic corporate-event classification and
  financial fact extraction rules.
- `events.application` coordinates stored news loading, analysis, and versioned
  event analysis persistence.
- `events.infrastructure` stores event analyses, detected event rows, and
  extracted financial fact rows.
- `events.presentation` exposes event analysis endpoints and explicit warnings.
- `evaluation.domain` owns gold annotation validation and deterministic metrics.
- `evaluation.application` coordinates dataset import, evaluation runs, and
  report writing.
- `evaluation.infrastructure` stores evaluation datasets, examples, gold labels,
  and run audit rows.
- `evaluation.presentation` exposes dataset and run endpoints.
- `shared` contains reusable configuration, database, and logging infrastructure.

Domain code does not import FastAPI, SQLAlchemy, Alembic, or any concrete
database driver.

## Create News Flow

1. `POST /api/v1/news` receives a Pydantic request model.
2. Boundary validation rejects blank text, invalid URLs, missing publication time,
   and timezone-naive timestamps.
3. The application service builds a domain `NewsItem`.
4. The domain layer preserves `raw_content` exactly and calculates
   `raw_content_hash` from normalized `source_id` plus the raw content.
5. The SQLAlchemy repository inserts the row and commits the transaction.
6. If a unique constraint conflict occurs, the repository rolls back and reads the
   existing row with the same idempotency key.
7. The API returns `201 Created` for a new row or `200 OK` for an existing row.

## News Deduplication

The database owns the final news deduplication guarantee. The unique constraint
covers:

- `source_id`
- `source_url`
- `raw_content_hash`

The repository intentionally does not rely on only a preliminary select, so two
concurrent identical requests cannot create duplicate rows.

## Instrument Matching Flow

1. `POST /api/v1/news/{news_id}/match-instruments` loads the stored news item.
2. The instrument repository loads active instruments and active aliases.
3. The domain normalizer creates a normalized text representation while keeping a
   map back to original character positions.
4. `InstrumentMatcher` checks exact ticker and exact alias token-boundary
   matches.
5. If one alias maps to more than one active instrument, all candidates are
   returned with `is_ambiguous=true`.
6. Repeated mentions of the same instrument are merged. The current rule keeps
   the highest-confidence match, then the lower alias priority, then the earliest
   original-text position.
7. Saved matches are replaced for the same `news_id` and matcher version, making
   reruns idempotent while keeping matcher versions explicit.

## Matching Confidence

Confidence is rule-based:

- `EXACT_TICKER`: `1.00`
- `OFFICIAL_NAME`: `0.98`
- `LEGAL_NAME`: `0.97`
- `SHORT_NAME`: `0.95`
- `BRAND`: `0.92`
- `MANUAL`: `0.90`

These values are deterministic metadata, not model probabilities.

## Database Deletion Rules

- `issuer_aliases.instrument_id` uses `ON DELETE CASCADE` because aliases have no
  meaning without their instrument.
- `news_instrument_matches.news_id` uses `ON DELETE CASCADE` because matches are
  analysis artifacts of one news row.
- `news_instrument_matches.instrument_id` uses `ON DELETE RESTRICT` to preserve
  explainability of historical matches and avoid silently orphaning analysis.
- `market_candles.instrument_id`, `market_data_imports.instrument_id`, and
  `news_market_reactions.instrument_id` use `ON DELETE RESTRICT` because candles
  and labels are historical facts tied to the instrument identity used at import
  or calculation time.
- `news_market_reactions.news_id` uses `ON DELETE CASCADE` because reactions are
  analysis artifacts of one stored news row.
- `reaction_points.reaction_id` uses `ON DELETE CASCADE` because points have no
  meaning without their parent reaction version.
- `news_event_analyses.news_id` uses `ON DELETE CASCADE` because event analyses
  are reproducible artifacts of one stored news row.
- `detected_events.analysis_id` and `extracted_financial_facts.analysis_id` use
  `ON DELETE CASCADE` because children have no meaning without their exact
  parent analysis version.

## Market Data Flow

1. `POST /api/v1/instruments/{instrument_id}/candles/backfill` loads the
   instrument.
2. The use case requires a ticker and `primary_board`; seeded MOEX shares use
   `TQBR`, but the field is nullable for existing and future non-share data.
3. A `MarketDataImport` row is created with status `RUNNING`.
4. `MoexIssClient` requests historical minute candles from MOEX ISS with
   bounded pagination and bounded retry.
5. The adapter maps `candles.columns` to indexes, validates each row, converts
   MOEX Moscow-local timestamps to UTC, and returns valid candles plus rejected
   row counts.
6. The repository saves candles in a batch and relies on the unique candle key to
   make repeated or concurrent imports idempotent.
7. The import record finishes as `SUCCEEDED`, `PARTIAL`, or `FAILED` with
   counters and an error code, never a stack trace.

Indexes:

- `instrument_id + interval_minutes + begin_at` supports range reads and reaction
  lookup for one instrument.
- `provider + board + begin_at` supports provider/board auditing and future
  maintenance queries.
- Import indexes by instrument/provider and `started_at` support operational
  status pages without scanning the audit table.

## Time Rules

All domain and database datetimes are timezone-aware. MOEX ISS candle `begin` and
`end` values are interpreted by the adapter as `Europe/Moscow` because the ISS
response does not include offsets. They are converted to UTC before domain
entities are created. News `published_at` and `received_at` are not modified.

## Reaction Calculation Flow

1. `POST /api/v1/news/{news_id}/calculate-reactions` loads the saved news item.
2. Saved `NewsInstrumentMatch` rows are loaded; the matcher is not run again.
3. For each matched instrument, baseline is the close of the last candle with
   `end_at <= published_at`.
4. Effective event time is the first candle `begin_at` with
   `begin_at >= published_at`. Exact equality means the just-started candle is
   the first post-publication candle.
5. For horizons `1`, `5`, `15`, `30`, and `60`, target time is calendar elapsed
   time `effective_event_at + horizon`, and target price is the close of the
   first candle with `end_at >= target_at`.
6. Simple and log returns are calculated with `Decimal`.
7. Missing baseline, effective candle, or target candles are stored explicitly as
   data-quality statuses.

This prevents look-ahead bias because baseline cannot use a candle ending after
publication. Ambiguous instrument matches are not collapsed; SBER and SBERP get
separate reaction rows with `is_ambiguous_instrument=true`.

## Event Analysis Flow

1. `POST /api/v1/news/{news_id}/analyze-event` loads the saved news item.
2. `EventAnalyzer` applies deterministic rule set `event-rules-v2` and fact
   extractor version `financial-facts-v2` to the original `raw_content`.
3. Event rules classify corporate events by explicit keyword and phrase
   patterns. Numeric fact rules extract values near supported metric names and
   normalize scale, currency, unit, period, role, comparison, and direction.
4. The repository replaces the saved analysis for the same `news_id` and
   `analysis_version`, keeping reruns idempotent and future rule versions
   coexistable.
5. `GET /api/v1/news/{news_id}/event-analysis` returns the saved result, with
   warnings for no event, no facts, missing periods, unknown metrics, and
   low-confidence facts.

The analyzer does not use LLMs, ML models, embeddings, fuzzy matching, sentiment
analysis, external AI APIs, or price data. It produces explainable extraction
metadata and stores evidence spans for later validation datasets.

Event and fact rows store `rule_id` plus evidence spans, and child tables have
exact-span uniqueness constraints. Concurrent reruns still rely on database
uniqueness for the parent `news_id + analysis_version` key rather than only a
preliminary read.

## Evaluation Flow

1. `create_annotation_batch` selects stored news, reruns the deterministic
   extractor, and writes `event-gold-v1` JSONL with empty gold labels for manual
   review.
2. `validate_annotation_dataset` checks JSONL shape, schema version, UUIDs,
   splits, review statuses, duplicate news ids, raw-content hashes, Decimal
   strings, and evidence spans when raw content is available.
3. `import_annotation_dataset` persists reviewed files into
   `evaluation_datasets`, `evaluation_examples`, `gold_events`, and
   `gold_financial_facts`. Imports are idempotent by source file hash.
4. `assign_temporal_split` assigns train, validation, and test splits by
   publication date and records the split boundaries on the dataset.
5. `evaluate_event_extraction` reruns the current analyzer on stored raw news,
   computes event and fact metrics, writes report artifacts, and appends an
   `evaluation_runs` audit row.

Event metrics include micro/macro precision, recall, F1, per-class support,
primary-event accuracy, coverage, unknown rate, ambiguous rate, and a
primary-event confusion matrix. Financial fact metrics use deterministic
one-to-one matching and report strict, value, metric, and field-level quality.

## Future Expansion

Later phases can add workers for source ingestion, a separate issuer registry,
MOEX reference data import, broader event rules, market data ingestion,
prediction storage, and evaluation. Issuers and instruments should eventually be
separate entities because one issuer can have common stock, preferred stock,
bonds, depositary receipts, or renamed instruments. Those additions should keep
facts, extracted values, model estimates, and explanations separate so later
market-reaction analysis avoids look-ahead bias.
