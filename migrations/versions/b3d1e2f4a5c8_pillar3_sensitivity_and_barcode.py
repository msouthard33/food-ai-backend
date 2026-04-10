"""pillar3 multi-sensitivity profiles, combined ratings, and barcode cache

Adds:
- user_sensitivity_profiles: per-user weights/thresholds across component types
  to support multi-protocol combined food rating (Pillar 4 dependency).
- food_combined_ratings: cached per-(user, food) combined safety score so the
  trigger UI does not recompute on every render.
- barcode_product_cache: local cache of Open Food Facts lookups so the mobile
  scanner is fast and offline-tolerant.

Revision ID: b3d1e2f4a5c8
Revises: a2c9f0e1d3b7
Create Date: 2026-04-06 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b3d1e2f4a5c8"
down_revision: Union[str, None] = "a2c9f0e1d3b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_sensitivity_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "component_type",
            postgresql.ENUM(name="component_type_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("weight", sa.Numeric(3, 2), nullable=False, server_default="1.00"),
        sa.Column("threshold", sa.Numeric(3, 1), nullable=False, server_default="5.0"),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("user_id", "component_type", name="uq_user_sensitivity_profiles_user_component"),
    )
    op.create_index(
        "ix_user_sensitivity_profiles_user_id",
        "user_sensitivity_profiles",
        ["user_id"],
    )

    op.create_table(
        "food_combined_ratings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("combined_score", sa.Numeric(4, 2), nullable=False),
        sa.Column("rating_label", sa.String(20), nullable=False),
        sa.Column("contributing_components", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("user_id", "food_id", name="uq_food_combined_ratings_user_food"),
    )
    op.create_index(
        "ix_food_combined_ratings_user_id",
        "food_combined_ratings",
        ["user_id"],
    )
    op.create_index(
        "ix_food_combined_ratings_food_id",
        "food_combined_ratings",
        ["food_id"],
    )

    op.create_table(
        "barcode_product_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("barcode", sa.String(32), nullable=False, unique=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="open_food_facts"),
        sa.Column("product_name", sa.String(500)),
        sa.Column("brand", sa.String(255)),
        sa.Column("ingredients_text", sa.Text),
        sa.Column("allergens_tags", postgresql.ARRAY(sa.Text)),
        sa.Column("nutriments", postgresql.JSONB),
        sa.Column("raw_payload", postgresql.JSONB),
        sa.Column(
            "matched_food_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("food_database.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("lookup_status", sa.String(20), nullable=False, server_default="found"),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_barcode_product_cache_barcode",
        "barcode_product_cache",
        ["barcode"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_barcode_product_cache_barcode", table_name="barcode_product_cache")
    op.drop_table("barcode_product_cache")
    op.drop_index("ix_food_combined_ratings_food_id", table_name="food_combined_ratings")
    op.drop_index("ix_food_combined_ratings_user_id", table_name="food_combined_ratings")
    op.drop_table("food_combined_ratings")
    op.drop_index("ix_user_sensitivity_profiles_user_id", table_name="user_sensitivity_profiles")
    op.drop_table("user_sensitivity_profiles")
