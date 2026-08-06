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

migration name:
    uv run alembic revision --autogenerate -m "{{name}}"

run:
    uv run uvicorn apps.api.main:app --reload

docker-up:
    docker compose up --build

docker-down:
    docker compose down

