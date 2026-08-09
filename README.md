# Trade AI News MVP

This project is the first foundation layer for an AI-assisted public market news
analyzer. The MVP accepts a public news item, preserves the original material and
metadata, deduplicates repeated submissions at the database layer, and exposes the
stored item through a REST API.

The project also includes deterministic matching of Russian market instruments
mentioned in stored news. Matching is based on an explicit local instrument and
issuer alias registry. It does not use LLMs, embeddings, fuzzy matching, MOEX
connectors, external AI APIs, or trading automation.

Stored news can also be analyzed by deterministic corporate-event and financial
fact rules. Version `event-rules-v1` classifies supported corporate event types
and extracts numeric facts with metric, period, unit, currency, scale, role,
comparison, evidence text, and character positions. This layer also avoids LLMs,
ML models, embeddings, sentiment scoring, and market-impact prediction.

## Architecture

The codebase is a modular monolith:

```text
apps/api/                         FastAPI application composition and seed command
src/news/domain/                  Framework-independent news entity and rules
src/news/application/             News use cases and repository ports
src/news/infrastructure/          News SQLAlchemy models and repositories
src/news/presentation/            News HTTP schemas and routes
src/instruments/domain/           Instruments, aliases, normalizer, matcher
src/instruments/application/      Instrument and matching use cases
src/instruments/infrastructure/   SQLAlchemy models, repositories, seed data
src/instruments/presentation/     Instrument HTTP schemas and routes
src/market_data/domain/           Market candles and import audit entities
src/market_data/application/      Backfill and candle query use cases
src/market_data/infrastructure/   MOEX ISS client and SQLAlchemy storage
src/market_data/presentation/     Market data HTTP schemas and routes
src/reactions/domain/             News market reaction labels
src/reactions/application/        Reaction calculation use cases
src/reactions/infrastructure/     SQLAlchemy reaction storage
src/reactions/presentation/       Reaction HTTP schemas and routes
src/events/domain/                Deterministic event and fact extraction
src/events/application/           Event analysis use cases and repository ports
src/events/infrastructure/        SQLAlchemy event analysis storage
src/events/presentation/          Event analysis HTTP schemas and routes
src/evaluation/domain/            Gold event dataset validation and metrics
src/evaluation/application/       Dataset import and evaluation use cases
src/evaluation/infrastructure/    Evaluation dataset and run storage
src/evaluation/presentation/      Evaluation HTTP schemas and routes
src/shared/config/                Environment-driven settings
src/shared/database/              Async SQLAlchemy engine and sessions
src/shared/logging/               JSON structured logging
tests/unit/                       Fast domain and schema tests
tests/integration/                API and repository integration tests
alembic/                          Database migrations
infra/                            Docker image files
docs/                             Architecture notes and ADRs
```

Domain logic does not import FastAPI, SQLAlchemy, or database-specific code.

## Requirements

- Python 3.12
- uv
- Docker and Docker Compose for PostgreSQL-backed local runs

## Local Setup

```bash
uv sync
cp .env.example .env
```

Run the API against the configured database:

```bash
uv run alembic upgrade head
uv run python -m apps.api.seed_instruments
uv run uvicorn apps.api.main:app --reload
```

## Docker Compose

```bash
docker compose up --build
```

The `postgres` service has a healthcheck. The `api` service waits for that
healthcheck, applies Alembic migrations, and starts Uvicorn.

## Migrations And Seed Data

Apply migrations:

```bash
uv run alembic upgrade head
```

Seed the MVP instrument registry:

```bash
just seed
```

Create a new migration:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

## Project Commands

```bash
just install
just format
just lint
just typecheck
just test
just test-unit
just test-integration
just migrate
just seed
just backfill-candles
just moex-smoke
just analyze-event <news-id>
just export-event-dataset
just create-annotation-batch
just validate-annotation-dataset <file>
just import-annotation-dataset <file> <name>
just assign-temporal-split <dataset-id> <train-until> <validation-until>
just evaluate-event-extraction <dataset-id>
just import-seed-event-batch
just run
just docker-up
just docker-down
```

## API Examples

Health:

```bash
curl http://localhost:8000/health
```

Readiness:

```bash
curl http://localhost:8000/ready
```

Create news:

```bash
curl -i -X POST http://localhost:8000/api/v1/news \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "test-source-001",
    "source_name": "Test News",
    "source_url": "https://example.com/news/1",
    "title": "Company published financial results",
    "raw_content": "SBER and Gazprom published updates.",
    "language": "en",
    "published_at": "2026-08-06T08:00:00Z",
    "received_at": "2026-08-06T08:00:01Z"
  }'
```

Get news:

```bash
curl http://localhost:8000/api/v1/news/<news-id>
```

Create an instrument:

```bash
curl -i -X POST http://localhost:8000/api/v1/instruments \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "GAZP",
    "short_name": "Gazprom",
    "full_name": "PAO Gazprom",
    "issuer_name": "PAO Gazprom",
    "exchange": "MOEX",
    "currency": "RUB",
    "instrument_type": "COMMON_STOCK",
    "primary_board": "TQBR"
  }'
```

List instruments:

```bash
curl "http://localhost:8000/api/v1/instruments?limit=100&offset=0"
```

Add an alias:

```bash
curl -i -X POST http://localhost:8000/api/v1/instruments/<instrument-id>/aliases \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "Gazprom",
    "alias_type": "OFFICIAL_NAME",
    "priority": 100
  }'
```

Run matching for an existing news item:

```bash
curl -X POST http://localhost:8000/api/v1/news/<news-id>/match-instruments
```

Read saved matches:

```bash
curl http://localhost:8000/api/v1/news/<news-id>/instruments
```

Backfill MOEX ISS minute candles:

```bash
curl -X POST http://localhost:8000/api/v1/instruments/<instrument-id>/candles/backfill \
  -H "Content-Type: application/json" \
  -d '{
    "date_from": "2026-07-01",
    "date_till": "2026-07-07",
    "interval_minutes": 1
  }'
```

Read saved candles:

```bash
curl "http://localhost:8000/api/v1/instruments/<instrument-id>/candles?from=2026-07-01T00:00:00Z&till=2026-07-08T00:00:00Z&interval_minutes=1&limit=500&offset=0"
```

Read an import audit record:

```bash
curl http://localhost:8000/api/v1/market-data/imports/<import-id>
```

Calculate and read market reactions:

```bash
curl -X POST http://localhost:8000/api/v1/news/<news-id>/calculate-reactions
curl http://localhost:8000/api/v1/news/<news-id>/reactions
```

Analyze and read deterministic corporate events and financial facts:

```bash
curl -X POST "http://localhost:8000/api/v1/news/<news-id>/analyze-event?debug=true"
curl http://localhost:8000/api/v1/news/<news-id>/event-analysis
```

Export a JSONL dataset that links stored news, event analysis, instrument
matches, and market reactions:

```bash
uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl
uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl --include-raw-content
```

Create, validate, import, split, and evaluate a reviewed gold annotation dataset:

```bash
uv run python -m apps.cli.create_annotation_batch --output artifacts/annotation-batch.jsonl --include-raw-content
uv run python -m apps.cli.validate_annotation_dataset --input artifacts/annotation-batch.jsonl --strict
uv run python -m apps.cli.import_annotation_dataset --input artifacts/annotation-batch.jsonl --name "manual-v1"
uv run python -m apps.cli.assign_temporal_split --dataset-id <dataset-id> --train-until 2026-07-31 --validation-until 2026-08-15
uv run python -m apps.cli.evaluate_event_extraction --dataset-id <dataset-id> --split TEST --fail-below-thresholds
```

Run the batch-001 seed curation workflow without creating final gold labels:

```bash
uv run python -m apps.cli.import_seed_event_batch --input artifacts/seed/ru_corporate_events_seed_50.jsonl --dry-run
uv run python -m apps.cli.import_seed_event_batch --input artifacts/seed/ru_corporate_events_seed_50.jsonl
```

The seed importer validates `event-seed-v1`, creates research `NewsItem` rows
with `DATE_ONLY / DO_NOT_USE_FOR_REACTION` in the title and review notes, runs
instrument matching and deterministic event analysis, then writes review
artifacts under `artifacts/seed/`. It keeps examples in `DRAFT`, leaves split
assignment as `UNASSIGNED`, does not calculate market reactions, and does not
import the records as a final gold dataset.

Evaluation API:

```bash
curl http://localhost:8000/api/v1/evaluation/datasets
curl http://localhost:8000/api/v1/evaluation/datasets/<dataset-id>
curl -X POST http://localhost:8000/api/v1/evaluation/datasets/<dataset-id>/runs \
  -H "Content-Type: application/json" \
  -d '{"split":"TEST"}'
curl http://localhost:8000/api/v1/evaluation/runs/<run-id>
```

## Idempotency

News ingestion deduplication is enforced by a unique database constraint across
`source_id`, `source_url`, and `raw_content_hash`.

Instrument matching is idempotent per `news_id` and `matcher_version`. A rerun of
the same matcher version replaces that version's saved rows and returns the
saved set without creating duplicates.

Market candle storage is idempotent through a unique database constraint on
`instrument_id`, `provider`, `board`, `interval_minutes`, and `begin_at`.
Re-importing the same range counts existing candles instead of inserting
duplicates.

Reaction calculation is idempotent by `news_id`, `instrument_id`, and
`reaction_version`. Version `reaction-v1-minute-candles` is replaced for the same
news item while future versions can coexist.

Event analysis is idempotent by `news_id` and `analysis_version`. Version
`event-rules-v1` is replaced for the same news item while future rule versions
can coexist.

Evaluation dataset import is idempotent by the SHA-256 hash of the source JSONL
file. Evaluation runs are append-only audit records tied to a dataset, split,
extractor versions, thresholds, and Git commit SHA.

## Instrument Matching

The matcher normalizes text with deterministic rules: Unicode normalization,
lowercase conversion, `ё` to `е`, quote normalization, whitespace collapse,
newline replacement, and punctuation separation. The original `raw_content` is
never changed.

Matching supports exact ticker and exact alias matches only. Tickers and aliases
must match on token boundaries, so `SBER` does not match inside `SBERP` or a
longer word. Repeated mentions and different aliases for the same instrument are
merged into one saved result using the highest confidence, alias priority, and
earliest original-text position.

Ambiguity is explicit. For example, the alias `Сбербанк` can refer to both
`SBER` and `SBERP`; the API returns both candidates with `is_ambiguous=true`
instead of selecting the more liquid common stock automatically. If the text
contains `SBER` or `SBERP` as an exact ticker token, the ticker match has
confidence `1.00`.

## MOEX Minute Candles

The first market data adapter uses MOEX ISS historical candles:

```text
https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{ticker}/candles.json
```

Only `interval=1` is supported in this phase. MOEX `begin` and `end` values do
not include an offset; this adapter interprets them as `Europe/Moscow` and
stores UTC-aware timestamps. The client validates ticker, board, interval, and a
maximum 31-day request range, uses bounded retry for HTTP 429 and temporary 5xx,
honors `Retry-After`, caps pagination by `MOEX_HTTP_MAX_PAGES`, and does not log
full HTTP responses.

The optional manual smoke command:

```bash
just moex-smoke
```

fetches a small SBER/TQBR historical range and prints received candle counts. It
does not write to the production database.

## Market Reaction Labels

`NewsItem.published_at` is the public event time. `received_at` is kept only to
measure ingestion latency. Baseline is the `close` of the last fully completed
minute candle whose `end_at <= published_at`, which prevents using market data
after the event. The effective event time is the `begin_at` of the first saved
candle whose `begin_at >= published_at`. If publication happens exactly at a
minute boundary, that just-started candle is the first post-publication candle.

For horizons `1`, `5`, `15`, `30`, and `60` minutes, target time is
`effective_event_at + horizon`. The target price is the `close` of the first
candle whose `end_at >= target_at`; the actual `observed_at` is stored because
trading gaps can shift it. Horizons use calendar elapsed time from
`effective_event_at`, not “N trading minutes”. Simple return is
`target_price / baseline_price - 1`. Log return is
`ln(target_price / baseline_price)`. Decimal arithmetic is used for stored prices
and returns.

Minute candles cannot identify the exact price at the second of publication. A
news item can land inside a candle, and some movement between publication and the
next minute boundary cannot be separated without trade-level data.

## Deterministic Event And Fact Extraction

`POST /api/v1/news/{news_id}/analyze-event` loads the stored raw news text,
applies deterministic regex rules, replaces the saved `event-rules-v1` analysis,
and returns detected events plus extracted facts. `GET
/api/v1/news/{news_id}/event-analysis` returns the saved result.

Event types include financial results, dividends, guidance, M&A, buyback,
management change, major contract, production update, sanctions, credit rating,
debt financing, and additional reserved enum values for future rules. Extracted
facts store metric, raw and normalized value, unit, currency, scale, period,
fact role, comparison type, change direction, evidence text, character offsets,
confidence metadata, and matched rule id. Confidence values are deterministic
rule metadata, not model probabilities.

The analysis version is `event-rules-v1`; financial facts store extractor
version `financial-facts-v1`. Event and fact rows include stable `rule_id`
values so later datasets can trace each label back to a deterministic rule.

The analyzer is intentionally conservative. Unknown numbers are kept with
`metric=OTHER`; missing periods are explicit through `period_type=UNKNOWN`; and
API responses include warnings for absent events, absent facts, low-confidence
facts, and facts without periods.

Supported number formats include comma and dot decimals, spaced thousands,
negative values, `%`, `п.п.`, Russian and English scales from thousands through
trillions, and RUB/USD/EUR/CNY symbols or names. Period extraction supports
year, half-year, quarter including Roman numerals and `Q1`, nine months, month,
and `FY2025`/`1H 2026` forms.

## MVP Limitations

- No news source connectors.
- No separate issuer registry yet; issuer fields currently live on instruments.
- No impact scoring, LLM usage, sentiment analysis, or market reaction prediction.
- No real-time polling, WebSocket, trades, order book, market-adjusted return, or
  forecasting.
- No Redis, Kafka, Elasticsearch, frontend, user authentication, or broker integration.
- No fuzzy matching, embeddings, vector database, or automatic MOEX reference data ingestion.
