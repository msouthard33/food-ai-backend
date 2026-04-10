"""W2-1b Pillar 1: food_embeddings_oai sidecar for OpenAI 1536-dim embeddings.

Companion to c4e2f1a9d6b0 (offline 384-dim sidecar). Kept as a second table
rather than widening the existing column because:
  - cleanly reversible (drop the table, the 384-dim path is untouched)
  - matches the W2-1 precedent of sidecar-over-mutation
  - lets both embedders co-exist so benchmarks can A/B compare

Revision ID: d7f9a2b1c4e6
Revises: c4e2f1a9d6b0
Create Date: 2026-04-08 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d7f9a2b1c4e6"
down_revision: Union[str, None] = "c4e2f1a9d6b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OAI_EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "food_embeddings_oai",
        sa.Column(
            "food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "embedding",
            sa.dialects.postgresql.ARRAY(sa.Float),  # placeholder, replaced below
            nullable=False,
        ),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("source_text", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.execute("ALTER TABLE food_embeddings_oai DROP COLUMN embedding")
    op.execute(
        f"ALTER TABLE food_embeddings_oai ADD COLUMN embedding vector({OAI_EMBEDDING_DIM}) NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_food_embeddings_oai_embedding_cos "
        "ON food_embeddings_oai USING ivfflat (embedding vector_cosine_ops) WITH (lists = 32)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_food_embeddings_oai_embedding_cos")
    op.drop_table("food_embeddings_oai")
