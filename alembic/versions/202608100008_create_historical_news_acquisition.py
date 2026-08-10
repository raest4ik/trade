"""create historical news acquisition

Revision ID: 202608100008
Revises: 202608100007
Create Date: 2026-08-10 22:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608100008"
down_revision: str | None = "202608100007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "historical_news_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_code", sa.String(length=128), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("content_storage_policy", sa.String(length=32), nullable=False),
        sa.Column("source_timezone", sa.String(length=128), nullable=True),
        sa.Column("feed_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_code"),
    )
    op.create_index(
        "ix_historical_news_sources_source_code",
        "historical_news_sources",
        ["source_code"],
    )
    op.create_table(
        "historical_news_import_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("date_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("discovered_count", sa.Integer(), nullable=False),
        sa.Column("validated_count", sa.Integer(), nullable=False),
        sa.Column("imported_count", sa.Integer(), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), nullable=False),
        sa.Column("rejected_count", sa.Integer(), nullable=False),
        sa.Column("metadata_only_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["historical_news_sources.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_historical_news_import_runs_source_started",
        "historical_news_import_runs",
        ["source_id", "started_at"],
    )
    op.create_index(
        "ix_historical_news_import_runs_status",
        "historical_news_import_runs",
        ["status"],
    )
    op.create_table(
        "historical_news_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("ingestion_run_id", sa.Uuid(), nullable=False),
        sa.Column("source_item_id", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.String(length=1000), nullable=False),
        sa.Column("source_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_timezone", sa.String(length=128), nullable=True),
        sa.Column("publication_timestamp_quality", sa.String(length=16), nullable=False),
        sa.Column("original_timestamp_text", sa.String(length=256), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("content_storage_policy", sa.String(length=32), nullable=False),
        sa.Column("content_is_excerpt", sa.Boolean(), nullable=False),
        sa.Column("exact_content_duplicate", sa.Boolean(), nullable=False),
        sa.Column("corrects_source_item_id", sa.String(length=512), nullable=True),
        sa.Column("supersedes_candidate_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("rejection_reason", sa.String(length=256), nullable=True),
        sa.Column("imported_news_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["historical_news_sources.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"], ["historical_news_import_runs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_candidate_id"],
            ["historical_news_candidates.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["imported_news_id"], ["news_items.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "source_item_id",
            name="uq_historical_news_candidates_source_item",
        ),
    )
    op.create_index(
        "ix_historical_news_candidates_source_status",
        "historical_news_candidates",
        ["source_id", "status"],
    )
    op.create_index(
        "ix_historical_news_candidates_quality_status",
        "historical_news_candidates",
        ["publication_timestamp_quality", "status"],
    )
    op.create_index(
        "ix_historical_news_candidates_content_hash",
        "historical_news_candidates",
        ["content_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_historical_news_candidates_content_hash",
        table_name="historical_news_candidates",
    )
    op.drop_index(
        "ix_historical_news_candidates_quality_status",
        table_name="historical_news_candidates",
    )
    op.drop_index(
        "ix_historical_news_candidates_source_status",
        table_name="historical_news_candidates",
    )
    op.drop_table("historical_news_candidates")
    op.drop_index(
        "ix_historical_news_import_runs_status",
        table_name="historical_news_import_runs",
    )
    op.drop_index(
        "ix_historical_news_import_runs_source_started",
        table_name="historical_news_import_runs",
    )
    op.drop_table("historical_news_import_runs")
    op.drop_index(
        "ix_historical_news_sources_source_code",
        table_name="historical_news_sources",
    )
    op.drop_table("historical_news_sources")
