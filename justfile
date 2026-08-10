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

analyze-event news_id:
    curl -X POST "http://localhost:8000/api/v1/news/{{news_id}}/analyze-event?debug=true"

export-event-dataset:
    uv run python -m apps.cli.export_event_dataset --output artifacts/event-dataset.jsonl

create-annotation-batch:
    uv run python -m apps.cli.create_annotation_batch --output artifacts/annotation-batch.jsonl

validate-annotation-dataset file:
    uv run python -m apps.cli.validate_annotation_dataset --input {{file}}

import-annotation-dataset file name:
    uv run python -m apps.cli.import_annotation_dataset --input {{file}} --name "{{name}}"

assign-temporal-split dataset_id train_until validation_until:
    uv run python -m apps.cli.assign_temporal_split --dataset-id {{dataset_id}} --train-until {{train_until}} --validation-until {{validation_until}}

evaluate-event-extraction dataset_id:
    uv run python -m apps.cli.evaluate_event_extraction --dataset-id {{dataset_id}} --fail-below-thresholds

analyze-event-ai text:
    uv run python -m apps.cli.analyze_event_ai --text "{{text}}"

evaluate-ai-event-extraction dataset_id:
    uv run python -m apps.cli.evaluate_ai_event_extraction --dataset-id {{dataset_id}} --split VALIDATION

import-seed-event-batch:
    uv run python -m apps.cli.import_seed_event_batch --input artifacts/seed/ru_corporate_events_seed_50.jsonl

migration name:
    uv run alembic revision --autogenerate -m "{{name}}"

run:
    uv run uvicorn apps.api.main:app --reload

docker-up:
    docker compose up --build

docker-down:
    docker compose down
