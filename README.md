# Trade AI News MVP

This project is the first foundation layer for an AI-assisted public market news
analyzer. The MVP accepts a public news item, preserves the original material and
metadata, deduplicates repeated submissions at the database layer, and exposes the
stored item through a REST API.

The project also includes deterministic matching of Russian market instruments
mentioned in stored news. Matching is based on an explicit local instrument and
issuer alias registry. It does not use LLMs, embeddings, fuzzy matching, MOEX
connectors, external AI APIs, or trading automation.

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
    "instrument_type": "COMMON_STOCK"
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

## Idempotency

News ingestion deduplication is enforced by a unique database constraint across
`source_id`, `source_url`, and `raw_content_hash`.

Instrument matching is idempotent per `news_id` and `matcher_version`. A rerun of
the same matcher version replaces that version's saved rows and returns the
saved set without creating duplicates.

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

## MVP Limitations

- No news source connectors.
- No separate issuer registry yet; issuer fields currently live on instruments.
- No event classification, impact scoring, LLM usage, or market reaction checks.
- No Redis, Kafka, Elasticsearch, frontend, user authentication, or broker integration.
- No fuzzy matching, embeddings, vector database, or automatic MOEX reference data ingestion.

