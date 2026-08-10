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

backfill-benchmark code date_from date_till:
    uv run python -m apps.cli.backfill_benchmark {{code}} --from {{date_from}} --till {{date_till}}

compute-abnormal-reactions news_id:
    uv run python -m apps.cli.compute_abnormal_reactions --news-id {{news_id}}

compute-abnormal-reactions-all limit="100":
    uv run python -m apps.cli.compute_abnormal_reactions --all --limit {{limit}}

export-market-reaction-dataset output="artifacts/market-reaction-dataset-v2.jsonl":
    uv run python -m apps.cli.export_market_reaction_dataset --output {{output}}

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

import-historical-news file source_code date_from date_to limit="1000":
    uv run python -m apps.cli.import_historical_news --input {{file}} --source-code {{source_code}} --storage-policy FULL_TEXT_ALLOWED --from {{date_from}} --to {{date_to}} --limit {{limit}}

backfill-historical-news:
    uv run python -m apps.cli.backfill_historical_news --help

export-historical-corpus:
    uv run python -m apps.cli.export_historical_corpus

historical-news-stats:
    uv run python -m apps.cli.historical_news_stats

build-ml-feature-dataset date_from date_to:
    uv run python -m apps.cli.build_ml_feature_dataset --from {{date_from}} --to {{date_to}}

export-ml-feature-dataset date_from date_to:
    uv run python -m apps.cli.export_ml_feature_dataset --from {{date_from}} --to {{date_to}}

ml-feature-stats:
    uv run python -m apps.cli.ml_feature_stats

ml-feature-smoke:
    uv run python -m apps.cli.ml_feature_dataset_smoke

migration name:
    uv run alembic revision --autogenerate -m "{{name}}"

run:
    uv run uvicorn apps.api.main:app --reload

docker-up:
    docker compose up --build

docker-down:
    docker compose down
