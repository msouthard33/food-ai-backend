"""W2-3c: medication_logs and food_drug_interactions tables

Adds:
- medication_logs: MCAS-differentiator — co-log antihistamines/meds alongside symptoms.
  RLS: user_id = auth.uid()
- food_drug_interactions: reference data for food-drug interaction alerts.
  RLS: PUBLIC SELECT (reference data readable by all authenticated users).

Revision ID: g3b4c5d6e7f8
Revises: f2a3b4c5d6e7
Create Date: 2026-04-12 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "g3b4c5d6e7f8"
down_revision: Union[str, None] = "f2a3b4c5d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── medication_logs ──────────────────────────────────────────────
    op.create_table(
        "medication_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "symptom_log_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("symptom_scores.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("medication_name", sa.String(255), nullable=False),
        sa.Column("dose_mg", sa.Numeric(8, 2)),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_medication_logs_user_id", "medication_logs", ["user_id"])
    op.create_index("ix_medication_logs_symptom_log_id", "medication_logs", ["symptom_log_id"])

    # RLS for medication_logs (user-scoped)
    op.execute("ALTER TABLE medication_logs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE medication_logs FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY medication_logs_user_policy ON medication_logs "
        "FOR ALL USING (user_id = current_setting('app.current_user_id')::uuid)"
    )

    # ── food_drug_interactions ───────────────────────────────────────
    op.create_table(
        "food_drug_interactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("drug_class", sa.String(255), nullable=False),
        sa.Column("interaction_type", sa.String(500), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_food_drug_interactions_food_id", "food_drug_interactions", ["food_id"])

    # RLS for food_drug_interactions (public SELECT — reference data)
    op.execute("ALTER TABLE food_drug_interactions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE food_drug_interactions FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY food_drug_interactions_select_policy ON food_drug_interactions "
        "FOR SELECT USING (true)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS food_drug_interactions_select_policy ON food_drug_interactions")
    op.drop_index("ix_food_drug_interactions_food_id", table_name="food_drug_interactions")
    op.drop_table("food_drug_interactions")
    op.execute("DROP POLICY IF EXISTS medication_logs_user_policy ON medication_logs")
    op.drop_index("ix_medication_logs_symptom_log_id", table_name="medication_logs")
    op.drop_index("ix_medication_logs_user_id", table_name="medication_logs")
    op.drop_table("medication_logs")
