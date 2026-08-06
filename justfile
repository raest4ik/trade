install:
    uv sync

format:
    uv run ruff format .

lint:
    uv run ruff check .

typecheck:
    uv run pyright

test:
    uv run pytest

test-unit:
    uv run pytest tests/unit

test-integration:
    uv run pytest tests/integration

migrate:
    uv run alembic upgrade head

seed:
    uv run python -m apps.api.seed_instruments

backfill-candles:
    uv run python -m apps.cli.backfill_candles --ticker SBER --from 2026-07-01 --till 2026-07-07 --interval 1

moex-smoke:
    uv run python -m apps.cli.moex_smoke

export-event-dataset:
    uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl

migration name:
    uv run alembic revision --autogenerate -m "{{name}}"

run:
    uv run uvicorn apps.api.main:app --reload

docker-up:
    docker compose up --build

docker-down:
    docker compose down
