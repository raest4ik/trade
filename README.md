# Trade AI News MVP

This project is the first foundation layer for an AI-assisted public market news
analyzer. The MVP accepts a public news item, preserves the original material and
metadata, deduplicates repeated submissions at the database layer, and exposes the
stored item through a REST API.

The MVP does not forecast stock prices, run sentiment analysis, call LLMs, scrape
websites, connect to exchanges or brokers, send Telegram messages, or execute
trades.

## Architecture

The codebase is a modular monolith:

```text
apps/api/                    FastAPI application composition
src/news/domain/             Framework-independent news entity and rules
src/news/application/        Use cases and repository ports
src/news/infrastructure/     SQLAlchemy models and repositories
src/news/presentation/       HTTP schemas and routes
src/shared/config/           Environment-driven settings
src/shared/database/         Async SQLAlchemy engine and sessions
src/shared/logging/          JSON structured logging
tests/unit/                  Fast domain and schema tests
tests/integration/           API and repository integration tests
alembic/                     Database migrations
infra/                       Docker image files
docs/                        Architecture notes
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
uv run uvicorn apps.api.main:app --reload
```

## Docker Compose

```bash
docker compose up --build
```

The `postgres` service has a healthcheck. The `api` service waits for that
healthcheck, applies Alembic migrations, and starts Uvicorn.

## Migrations

Apply migrations:

```bash
uv run alembic upgrade head
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
    "raw_content": "Original publication text.",
    "language": "en",
    "published_at": "2026-08-06T08:00:00Z",
    "received_at": "2026-08-06T08:00:01Z"
  }'
```

Get news:

```bash
curl http://localhost:8000/api/v1/news/<news-id>
```

## Idempotency

The deterministic content hash is calculated from a normalized `source_id` and
the unmodified `raw_content`. The database enforces uniqueness across
`source_id`, `source_url`, and `raw_content_hash`. A repeated POST returns the
existing row instead of creating a duplicate. Concurrent duplicate inserts are
handled by the database unique constraint and an explicit conflict recovery path.

## MVP Limitations

- No news source connectors.
- No issuer or ticker extraction yet.
- No event classification, impact scoring, LLM usage, or market reaction checks.
- No Redis, Kafka, frontend, user authentication, or broker integration.

