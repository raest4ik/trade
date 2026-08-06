# Architecture

## Layers

The application is a modular monolith split by feature and layer.

- `news.domain` owns the `NewsItem` entity, timestamp normalization, content hash
  calculation, and validation rules that are independent of frameworks.
- `news.application` contains use cases and repository protocols. It coordinates
  domain behavior without knowing how persistence is implemented.
- `news.infrastructure` contains SQLAlchemy table mappings and repository code.
- `news.presentation` contains FastAPI routes and Pydantic request/response
  schemas.
- `shared` contains reusable configuration, database, and logging infrastructure.

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

## Deduplication

The database owns the final deduplication guarantee. The unique constraint covers:

- `source_id`
- `source_url`
- `raw_content_hash`

The repository intentionally does not rely on only a preliminary select, so two
concurrent identical requests cannot create duplicate rows.

## Domain Boundary

Domain code does not import FastAPI, SQLAlchemy, Alembic, or any concrete database
driver. That keeps future event extraction, issuer recognition, and prediction
logic testable without HTTP or database setup.

## Future Expansion

Later phases can add workers for source ingestion, issuer and ticker extraction,
event classification, market data ingestion, prediction storage, and evaluation.
Those additions should keep facts, extracted values, model estimates, and
explanations separate so later market-reaction analysis avoids look-ahead bias.

