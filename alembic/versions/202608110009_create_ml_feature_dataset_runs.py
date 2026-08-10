"""create ml feature dataset runs

Revision ID: 202608110009
Revises: 202608100008
Create Date: 2026-08-11 01:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608110009"
down_revision: str | None = "202608100008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ml_feature_dataset_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.Column("date_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("eligible_count", sa.Integer(), nullable=False),
        sa.Column("built_count", sa.Integer(), nullable=False),
        sa.Column("excluded_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("git_sha", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ml_feature_dataset_runs_started_at",
        "ml_feature_dataset_runs",
        ["started_at"],
    )
    op.create_index(
        "ix_ml_feature_dataset_runs_status",
        "ml_feature_dataset_runs",
        ["status"],
    )
    op.create_index(
        "ix_ml_feature_dataset_runs_config_hash",
        "ml_feature_dataset_runs",
        ["config_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ml_feature_dataset_runs_config_hash",
        table_name="ml_feature_dataset_runs",
    )
    op.drop_index("ix_ml_feature_dataset_runs_status", table_name="ml_feature_dataset_runs")
    op.drop_index(
        "ix_ml_feature_dataset_runs_started_at",
        table_name="ml_feature_dataset_runs",
    )
    op.drop_table("ml_feature_dataset_runs")
