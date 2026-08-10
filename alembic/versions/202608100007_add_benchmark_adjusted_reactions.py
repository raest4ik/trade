"""add benchmark-adjusted market reactions

Revision ID: 202608100007
Revises: 202608060006
Create Date: 2026-08-10 18:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608100007"
down_revision: str | None = "202608060006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "news_items",
        sa.Column(
            "publication_timestamp_quality",
            sa.String(length=16),
            server_default="UNKNOWN",
            nullable=False,
        ),
    )
    op.create_index(
        "ix_news_items_publication_timestamp_quality",
        "news_items",
        ["publication_timestamp_quality"],
    )
    op.execute(
        sa.text(
            "UPDATE news_items SET publication_timestamp_quality = 'DATE_ONLY' "
            "WHERE source_name = 'seed-dataset'"
        )
    )

    op.create_table(
        "market_benchmarks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("market", sa.String(length=32), nullable=False),
        sa.Column("board", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index("ix_market_benchmarks_code", "market_benchmarks", ["code"])

    op.create_table(
        "benchmark_candles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False),
        sa.Column("begin_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(24, 10), nullable=False),
        sa.Column("high", sa.Numeric(24, 10), nullable=False),
        sa.Column("low", sa.Numeric(24, 10), nullable=False),
        sa.Column("close", sa.Numeric(24, 10), nullable=False),
        sa.Column("volume", sa.Numeric(28, 10), nullable=False),
        sa.Column("value", sa.Numeric(28, 10), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("adapter_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["benchmark_id"], ["market_benchmarks.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "benchmark_id",
            "provider",
            "interval_minutes",
            "begin_at",
            name="uq_benchmark_candles_benchmark_provider_interval_begin",
        ),
    )
    op.create_index(
        "ix_benchmark_candles_benchmark_interval_begin",
        "benchmark_candles",
        ["benchmark_id", "interval_minutes", "begin_at"],
    )
    op.create_index("ix_benchmark_candles_end_at", "benchmark_candles", ["end_at"])

    op.alter_column("market_data_imports", "instrument_id", nullable=True)
    op.add_column(
        "market_data_imports",
        sa.Column("benchmark_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "market_data_imports",
        sa.Column(
            "dataset_type",
            sa.String(length=16),
            server_default="SECURITY",
            nullable=False,
        ),
    )
    op.create_foreign_key(
        "fk_market_data_imports_benchmark_id",
        "market_data_imports",
        "market_benchmarks",
        ["benchmark_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_market_data_imports_dataset_reference",
        "market_data_imports",
        "(dataset_type = 'SECURITY' AND instrument_id IS NOT NULL AND benchmark_id IS NULL) "
        "OR (dataset_type = 'BENCHMARK' AND instrument_id IS NULL AND benchmark_id IS NOT NULL)",
    )
    op.create_index(
        "ix_market_data_imports_dataset_symbol_started",
        "market_data_imports",
        ["dataset_type", "ticker", "started_at"],
    )

    op.create_table(
        "reaction_benchmark_adjustments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reaction_point_id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_code", sa.String(length=32), nullable=False),
        sa.Column("baseline_value", sa.Numeric(24, 10), nullable=True),
        sa.Column("target_value", sa.Numeric(24, 10), nullable=True),
        sa.Column("baseline_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("simple_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("log_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("abnormal_simple_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("abnormal_log_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("missing_reason", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["benchmark_id"], ["market_benchmarks.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["reaction_point_id"], ["reaction_points.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "reaction_point_id",
            "benchmark_id",
            name="uq_reaction_benchmark_adjustments_point_benchmark",
        ),
    )
    op.create_index(
        "ix_reaction_benchmark_adjustments_benchmark_status",
        "reaction_benchmark_adjustments",
        ["benchmark_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_reaction_benchmark_adjustments_benchmark_status",
        table_name="reaction_benchmark_adjustments",
    )
    op.drop_table("reaction_benchmark_adjustments")
    op.drop_index(
        "ix_market_data_imports_dataset_symbol_started",
        table_name="market_data_imports",
    )
    op.execute(
        sa.text(
            "ALTER TABLE market_data_imports "
            "DROP CONSTRAINT IF EXISTS ck_market_data_imports_dataset_reference"
        )
    )
    op.drop_constraint(
        "fk_market_data_imports_benchmark_id",
        "market_data_imports",
        type_="foreignkey",
    )
    op.drop_column("market_data_imports", "dataset_type")
    op.drop_column("market_data_imports", "benchmark_id")
    op.alter_column("market_data_imports", "instrument_id", nullable=False)
    op.drop_index("ix_benchmark_candles_end_at", table_name="benchmark_candles")
    op.drop_index(
        "ix_benchmark_candles_benchmark_interval_begin",
        table_name="benchmark_candles",
    )
    op.drop_table("benchmark_candles")
    op.drop_index("ix_market_benchmarks_code", table_name="market_benchmarks")
    op.drop_table("market_benchmarks")
    op.drop_index(
        "ix_news_items_publication_timestamp_quality",
        table_name="news_items",
    )
    op.drop_column("news_items", "publication_timestamp_quality")
