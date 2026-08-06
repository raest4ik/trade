"""add event rule ids and child uniqueness

Revision ID: 202608060005
Revises: 202608060004
Create Date: 2026-08-06 00:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "202608060005"
down_revision: str | None = "202608060004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "detected_events",
        sa.Column("rule_id", sa.String(length=128), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "extracted_financial_facts",
        sa.Column("rule_id", sa.String(length=128), nullable=False, server_default="unknown"),
    )
    op.execute("UPDATE detected_events SET rule_id = matched_rule WHERE rule_id = 'unknown'")
    op.execute(
        "UPDATE extracted_financial_facts SET rule_id = matched_rule WHERE rule_id = 'unknown'"
    )
    op.alter_column("detected_events", "rule_id", server_default=None)
    op.alter_column("extracted_financial_facts", "rule_id", server_default=None)
    op.create_unique_constraint(
        "uq_detected_events_exact_span",
        "detected_events",
        ["analysis_id", "event_type", "rule_id", "start_position", "end_position"],
    )
    op.create_unique_constraint(
        "uq_extracted_financial_facts_exact_span",
        "extracted_financial_facts",
        ["analysis_id", "metric", "rule_id", "start_position", "end_position"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_extracted_financial_facts_exact_span",
        "extracted_financial_facts",
        type_="unique",
    )
    op.drop_constraint("uq_detected_events_exact_span", "detected_events", type_="unique")
    op.drop_column("extracted_financial_facts", "rule_id")
    op.drop_column("detected_events", "rule_id")
