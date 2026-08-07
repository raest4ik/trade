"""create evaluation pipeline tables

Revision ID: 202608060006
Revises: 202608060005
Create Date: 2026-08-06 00:06:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608060006"
down_revision: str | None = "202608060005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_file_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("reviewed_count", sa.Integer(), nullable=False),
        sa.Column("train_count", sa.Integer(), nullable=False),
        sa.Column("validation_count", sa.Integer(), nullable=False),
        sa.Column("test_count", sa.Integer(), nullable=False),
        sa.Column("split_strategy", sa.String(length=64), nullable=True),
        sa.Column("train_until", sa.Date(), nullable=True),
        sa.Column("validation_until", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_file_hash",
            name="uq_evaluation_datasets_source_file_hash",
        ),
    )
    op.create_index("ix_evaluation_datasets_name", "evaluation_datasets", ["name"])

    op.create_table(
        "evaluation_examples",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("news_id", sa.Uuid(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_content_hash", sa.String(length=64), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("annotator", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("predicted_events", sa.JSON(), nullable=False),
        sa.Column("predicted_financial_facts", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["evaluation_datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["news_id"], ["news_items.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "news_id", name="uq_evaluation_examples_dataset_news"),
    )
    op.create_index(
        "ix_evaluation_examples_dataset_split",
        "evaluation_examples",
        ["dataset_id", "split"],
    )
    op.create_index("ix_evaluation_examples_news_id", "evaluation_examples", ["news_id"])

    op.create_table(
        "gold_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("example_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("evidence_text", sa.String(length=1000), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["example_id"], ["evaluation_examples.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_gold_events_example_type", "gold_events", ["example_id", "event_type"])

    op.create_table(
        "gold_financial_facts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("example_id", sa.Uuid(), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("raw_value", sa.Numeric(28, 10), nullable=False),
        sa.Column("normalized_value", sa.Numeric(38, 10), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("scale", sa.String(length=32), nullable=False),
        sa.Column("period_type", sa.String(length=32), nullable=False),
        sa.Column("period_year", sa.Integer(), nullable=True),
        sa.Column("period_quarter", sa.Integer(), nullable=True),
        sa.Column("period_month", sa.Integer(), nullable=True),
        sa.Column("raw_period", sa.String(length=128), nullable=True),
        sa.Column("fact_role", sa.String(length=32), nullable=False),
        sa.Column("comparison_type", sa.String(length=32), nullable=False),
        sa.Column("change_direction", sa.String(length=32), nullable=False),
        sa.Column("change_value", sa.Numeric(28, 10), nullable=True),
        sa.Column("change_unit", sa.String(length=32), nullable=True),
        sa.Column("evidence_text", sa.String(length=1000), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["example_id"], ["evaluation_examples.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_gold_financial_facts_example_metric",
        "gold_financial_facts",
        ["example_id", "metric"],
    )

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("analysis_version", sa.String(length=64), nullable=False),
        sa.Column("extractor_version", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("git_commit_sha", sa.String(length=64), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["evaluation_datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evaluation_runs_dataset_started",
        "evaluation_runs",
        ["dataset_id", "started_at"],
    )
    op.create_index("ix_evaluation_runs_status", "evaluation_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_status", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_dataset_started", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index(
        "ix_gold_financial_facts_example_metric",
        table_name="gold_financial_facts",
    )
    op.drop_table("gold_financial_facts")
    op.drop_index("ix_gold_events_example_type", table_name="gold_events")
    op.drop_table("gold_events")
    op.drop_index("ix_evaluation_examples_news_id", table_name="evaluation_examples")
    op.drop_index(
        "ix_evaluation_examples_dataset_split",
        table_name="evaluation_examples",
    )
    op.drop_table("evaluation_examples")
    op.drop_index("ix_evaluation_datasets_name", table_name="evaluation_datasets")
    op.drop_table("evaluation_datasets")
