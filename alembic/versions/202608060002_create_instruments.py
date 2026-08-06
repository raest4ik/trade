"""create instruments and matching tables

Revision ID: 202608060002
Revises: 202608060001
Create Date: 2026-08-06 00:02:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608060002"
down_revision: str | None = "202608060001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("figi", sa.String(length=32), nullable=True),
        sa.Column("isin", sa.String(length=16), nullable=True),
        sa.Column("short_name", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=500), nullable=False),
        sa.Column("issuer_name", sa.String(length=255), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("instrument_type", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("exchange", "ticker", name="uq_instruments_exchange_ticker"),
        sa.UniqueConstraint("isin", name="uq_instruments_isin"),
    )
    op.create_index("ix_instruments_exchange", "instruments", ["exchange"])
    op.create_index("ix_instruments_is_active", "instruments", ["is_active"])
    op.create_index("ix_instruments_issuer_name", "instruments", ["issuer_name"])
    op.create_index("ix_instruments_ticker", "instruments", ["ticker"])

    op.create_table(
        "issuer_aliases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("alias", sa.String(length=500), nullable=False),
        sa.Column("normalized_alias", sa.String(length=500), nullable=False),
        sa.Column("alias_type", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "normalized_alias",
            name="uq_issuer_aliases_instrument_normalized_alias",
        ),
    )
    op.create_index("ix_issuer_aliases_alias_type", "issuer_aliases", ["alias_type"])
    op.create_index("ix_issuer_aliases_instrument_id", "issuer_aliases", ["instrument_id"])
    op.create_index("ix_issuer_aliases_is_active", "issuer_aliases", ["is_active"])
    op.create_index("ix_issuer_aliases_normalized_alias", "issuer_aliases", ["normalized_alias"])

    op.create_table(
        "news_instrument_matches",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("news_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("matched_alias", sa.String(length=500), nullable=False),
        sa.Column("alias_type", sa.String(length=32), nullable=False),
        sa.Column("match_type", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.Column("is_ambiguous", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("matcher_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["news_id"], ["news_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "news_id",
            "instrument_id",
            "matcher_version",
            name="uq_news_instrument_matches_news_instrument_version",
        ),
    )
    op.create_index(
        "ix_news_instrument_matches_instrument_id",
        "news_instrument_matches",
        ["instrument_id"],
    )
    op.create_index(
        "ix_news_instrument_matches_is_ambiguous",
        "news_instrument_matches",
        ["is_ambiguous"],
    )
    op.create_index(
        "ix_news_instrument_matches_matcher_version",
        "news_instrument_matches",
        ["matcher_version"],
    )
    op.create_index("ix_news_instrument_matches_news_id", "news_instrument_matches", ["news_id"])


def downgrade() -> None:
    op.drop_index("ix_news_instrument_matches_news_id", table_name="news_instrument_matches")
    op.drop_index(
        "ix_news_instrument_matches_matcher_version",
        table_name="news_instrument_matches",
    )
    op.drop_index("ix_news_instrument_matches_is_ambiguous", table_name="news_instrument_matches")
    op.drop_index("ix_news_instrument_matches_instrument_id", table_name="news_instrument_matches")
    op.drop_table("news_instrument_matches")
    op.drop_index("ix_issuer_aliases_normalized_alias", table_name="issuer_aliases")
    op.drop_index("ix_issuer_aliases_is_active", table_name="issuer_aliases")
    op.drop_index("ix_issuer_aliases_instrument_id", table_name="issuer_aliases")
    op.drop_index("ix_issuer_aliases_alias_type", table_name="issuer_aliases")
    op.drop_table("issuer_aliases")
    op.drop_index("ix_instruments_ticker", table_name="instruments")
    op.drop_index("ix_instruments_issuer_name", table_name="instruments")
    op.drop_index("ix_instruments_is_active", table_name="instruments")
    op.drop_index("ix_instruments_exchange", table_name="instruments")
    op.drop_table("instruments")
