"""W2-1 Pillar 1: food_embeddings sidecar table for semantic search.

Adds a sidecar `food_embeddings` table (1:1 to food_database) holding a
pgvector embedding for each food. Deliberately NOT an ADD COLUMN on
food_database per W2-1 scope guardrail ('do not touch app/models/food.py
ARRAY columns'). Sidecar keeps ingest code unchanged and lets us rebuild
embeddings idempotently without touching the canonical KB rows.

Uses pgvector 0.8.0 (bootstrapped by conftest.py for tests; prod lift is
OUT OF SCOPE for W2-1 — this migration must not be run against Railway
until a gated foodsci-ingestion sprint approves it).

Revision ID: c4e2f1a9d6b0
Revises: b3d1e2f4a5c8
Create Date: 2026-04-08 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c4e2f1a9d6b0"
down_revision: Union[str, None] = "b3d1e2f4a5c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Embedding dimension. 384 matches common small sentence-transformer sizes
# (e.g. all-MiniLM-L6-v2) and is also what the deterministic offline fallback
# in llm_provider.embed() produces. Any live embedding provider that returns
# a different dim must project/pad to 384 before insert.
EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "food_embeddings",
        sa.Column(
            "food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "embedding",
            sa.dialects.postgresql.ARRAY(sa.Float),  # overridden below with raw DDL
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
    # Replace the ARRAY(Float) placeholder with a real pgvector column — alembic
    # doesn't know the vector type natively.
    op.execute("ALTER TABLE food_embeddings DROP COLUMN embedding")
    op.execute(f"ALTER TABLE food_embeddings ADD COLUMN embedding vector({EMBEDDING_DIM}) NOT NULL")
    # IVFFlat index for cosine distance. Low list count is fine for <10k rows.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_food_embeddings_embedding_cos "
        "ON food_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 32)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_food_embeddings_embedding_cos")
    op.drop_table("food_embeddings")
