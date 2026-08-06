"""create event analysis tables

Revision ID: 202608060004
Revises: 202608060003
Create Date: 2026-08-06 00:04:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608060004"
down_revision: str | None = "202608060003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "news_event_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("news_id", sa.Uuid(), nullable=False),
        sa.Column("analysis_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("primary_event_type", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["news_id"], ["news_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "news_id",
            "analysis_version",
            name="uq_news_event_analyses_news_version",
        ),
    )
    op.create_index(
        "ix_news_event_analyses_news_version",
        "news_event_analyses",
        ["news_id", "analysis_version"],
        unique=False,
    )
    op.create_index(
        "ix_news_event_analyses_status",
        "news_event_analyses",
        ["status"],
        unique=False,
    )

    op.create_table(
        "detected_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Numeric(10, 6), nullable=False),
        sa.Column("matched_rule", sa.String(length=128), nullable=False),
        sa.Column("evidence_text", sa.String(length=1000), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["news_event_analyses.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_detected_events_analysis_type",
        "detected_events",
        ["analysis_id", "event_type"],
        unique=False,
    )
    op.create_index("ix_detected_events_type", "detected_events", ["event_type"], unique=False)

    op.create_table(
        "extracted_financial_facts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("raw_value", sa.Numeric(28, 10), nullable=False),
        sa.Column("normalized_value", sa.Numeric(38, 10), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("currency", sa.String(length=16), nullable=False),
        sa.Column("scale", sa.String(length=32), nullable=False),
        sa.Column("period_type", sa.String(length=32), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("quarter", sa.Integer(), nullable=True),
        sa.Column("month", sa.Integer(), nullable=True),
        sa.Column("date_from", sa.Date(), nullable=True),
        sa.Column("date_to", sa.Date(), nullable=True),
        sa.Column("raw_period", sa.String(length=128), nullable=True),
        sa.Column("comparison_type", sa.String(length=32), nullable=False),
        sa.Column("fact_role", sa.String(length=32), nullable=False),
        sa.Column("change_direction", sa.String(length=32), nullable=False),
        sa.Column("change_value", sa.Numeric(28, 10), nullable=True),
        sa.Column("change_unit", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Numeric(10, 6), nullable=False),
        sa.Column("evidence_text", sa.String(length=1000), nullable=False),
        sa.Column("start_position", sa.Integer(), nullable=False),
        sa.Column("end_position", sa.Integer(), nullable=False),
        sa.Column("extractor_version", sa.String(length=64), nullable=False),
        sa.Column("matched_rule", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["news_event_analyses.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extracted_financial_facts_analysis_metric",
        "extracted_financial_facts",
        ["analysis_id", "metric"],
        unique=False,
    )
    op.create_index(
        "ix_extracted_financial_facts_metric",
        "extracted_financial_facts",
        ["metric"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_extracted_financial_facts_metric",
        table_name="extracted_financial_facts",
    )
    op.drop_index(
        "ix_extracted_financial_facts_analysis_metric",
        table_name="extracted_financial_facts",
    )
    op.drop_table("extracted_financial_facts")
    op.drop_index("ix_detected_events_type", table_name="detected_events")
    op.drop_index("ix_detected_events_analysis_type", table_name="detected_events")
    op.drop_table("detected_events")
    op.drop_index("ix_news_event_analyses_status", table_name="news_event_analyses")
    op.drop_index("ix_news_event_analyses_news_version", table_name="news_event_analyses")
    op.drop_table("news_event_analyses")
