"""create market data and reaction tables

Revision ID: 202608060003
Revises: 202608060002
Create Date: 2026-08-06 14:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "202608060003"
down_revision = "202608060002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("instruments", sa.Column("primary_board", sa.String(length=32), nullable=True))
    op.create_index("ix_instruments_primary_board", "instruments", ["primary_board"])

    op.create_table(
        "market_data_imports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("board", sa.String(length=32), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False),
        sa.Column("requested_from", sa.Date(), nullable=False),
        sa.Column("requested_till", sa.Date(), nullable=False),
        sa.Column("source_timezone", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("pages_received", sa.Integer(), nullable=False),
        sa.Column("rows_received", sa.Integer(), nullable=False),
        sa.Column("rows_valid", sa.Integer(), nullable=False),
        sa.Column("rows_rejected", sa.Integer(), nullable=False),
        sa.Column("rows_inserted", sa.Integer(), nullable=False),
        sa.Column("rows_existing", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("adapter_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_market_data_imports_instrument_started",
        "market_data_imports",
        ["instrument_id", "started_at"],
    )
    op.create_index(
        "ix_market_data_imports_provider_board_started",
        "market_data_imports",
        ["provider", "board", "started_at"],
    )
    op.create_index("ix_market_data_imports_status", "market_data_imports", ["status"])

    op.create_table(
        "market_candles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("market", sa.String(length=32), nullable=False),
        sa.Column("board", sa.String(length=32), nullable=False),
        sa.Column("ticker_snapshot", sa.String(length=32), nullable=False),
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
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "provider",
            "board",
            "interval_minutes",
            "begin_at",
            name="uq_market_candles_instrument_provider_board_interval_begin",
        ),
    )
    op.create_index(
        "ix_market_candles_instrument_interval_begin",
        "market_candles",
        ["instrument_id", "interval_minutes", "begin_at"],
    )
    op.create_index(
        "ix_market_candles_provider_board_begin",
        "market_candles",
        ["provider", "board", "begin_at"],
    )
    op.create_index("ix_market_candles_end_at", "market_candles", ["end_at"])

    op.create_table(
        "news_market_reactions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("news_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("reaction_version", sa.String(length=64), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("baseline_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("baseline_price", sa.Numeric(24, 10), nullable=True),
        sa.Column("publication_to_receipt_ms", sa.Integer(), nullable=False),
        sa.Column("publication_to_effective_event_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_ambiguous_instrument", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["news_id"], ["news_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "news_id",
            "instrument_id",
            "reaction_version",
            name="uq_news_market_reactions_news_instrument_version",
        ),
    )
    op.create_index(
        "ix_news_market_reactions_news_version",
        "news_market_reactions",
        ["news_id", "reaction_version"],
    )
    op.create_index(
        "ix_news_market_reactions_instrument_created",
        "news_market_reactions",
        ["instrument_id", "created_at"],
    )
    op.create_index("ix_news_market_reactions_status", "news_market_reactions", ["status"])

    op.create_table(
        "reaction_points",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reaction_id", sa.Uuid(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("target_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("price", sa.Numeric(24, 10), nullable=True),
        sa.Column("simple_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("log_return", sa.Numeric(28, 18), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["reaction_id"], ["news_market_reactions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "reaction_id",
            "horizon_minutes",
            name="uq_reaction_points_reaction_horizon",
        ),
    )
    op.create_index(
        "ix_reaction_points_reaction_horizon",
        "reaction_points",
        ["reaction_id", "horizon_minutes"],
    )


def downgrade() -> None:
    op.drop_index("ix_reaction_points_reaction_horizon", table_name="reaction_points")
    op.drop_table("reaction_points")
    op.drop_index("ix_news_market_reactions_status", table_name="news_market_reactions")
    op.drop_index("ix_news_market_reactions_instrument_created", table_name="news_market_reactions")
    op.drop_index("ix_news_market_reactions_news_version", table_name="news_market_reactions")
    op.drop_table("news_market_reactions")
    op.drop_index("ix_market_candles_end_at", table_name="market_candles")
    op.drop_index("ix_market_candles_provider_board_begin", table_name="market_candles")
    op.drop_index("ix_market_candles_instrument_interval_begin", table_name="market_candles")
    op.drop_table("market_candles")
    op.drop_index("ix_market_data_imports_status", table_name="market_data_imports")
    op.drop_index("ix_market_data_imports_provider_board_started", table_name="market_data_imports")
    op.drop_index("ix_market_data_imports_instrument_started", table_name="market_data_imports")
    op.drop_table("market_data_imports")
    op.drop_index("ix_instruments_primary_board", table_name="instruments")
    op.drop_column("instruments", "primary_board")
