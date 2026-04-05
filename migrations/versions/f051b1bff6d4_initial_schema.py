"""initial_schema

Revision ID: f051b1bff6d4
Revises:
Create Date: 2026-04-05 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f051b1bff6d4"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# --- Enum types ---------------------------------------------------------------

meal_type_enum = sa.Enum(
    "breakfast", "lunch", "dinner", "snack", "beverage",
    name="meal_type_enum",
)
symptom_type_enum = sa.Enum(
    "bloating", "pain", "nausea", "brain_fog", "fatigue",
    "skin_reaction", "bowel_changes", "heartburn", "headache",
    "joint_pain", "respiratory", "other",
    name="symptom_type_enum",
)
processing_status_enum = sa.Enum(
    "pending", "processing", "complete", "failed",
    name="processing_status_enum",
)
condition_type_enum = sa.Enum(
    "ibs", "sibo", "histamine_intolerance", "celiac", "mcas",
    "nickel_allergy", "fodmap_sensitivity", "leaky_gut", "dysbiosis",
    "crohns", "ulcerative_colitis", "other",
    name="condition_type_enum",
)
component_type_enum = sa.Enum(
    "shellfish", "fish", "tree_nuts", "peanuts", "milk_dairy", "eggs",
    "wheat", "soy", "sesame", "sulfites", "histamines", "fodmap",
    "nightshades", "gluten", "lectins", "oxalates", "salicylates",
    "amines", "fructose", "lactose", "alcohol", "additives",
    name="component_type_enum",
)
trigger_status_enum = sa.Enum(
    "suspect", "probable", "confirmed", "cleared",
    name="trigger_status_enum",
)
report_type_enum = sa.Enum(
    "weekly", "monthly", "custom", "clinician",
    name="report_type_enum",
)
share_method_enum = sa.Enum(
    "link", "pdf", "in_app",
    name="share_method_enum",
)
severity_enum = sa.Enum(
    "mild", "moderate", "severe",
    name="severity_enum",
)
component_source_enum = sa.Enum(
    "database", "ai_inferred", "user_reported", "research_based",
    name="component_source_enum",
)


def upgrade() -> None:
    # --- Create enum types first ---
    meal_type_enum.create(op.get_bind(), checkfirst=True)
    symptom_type_enum.create(op.get_bind(), checkfirst=True)
    processing_status_enum.create(op.get_bind(), checkfirst=True)
    condition_type_enum.create(op.get_bind(), checkfirst=True)
    component_type_enum.create(op.get_bind(), checkfirst=True)
    trigger_status_enum.create(op.get_bind(), checkfirst=True)
    report_type_enum.create(op.get_bind(), checkfirst=True)
    share_method_enum.create(op.get_bind(), checkfirst=True)
    severity_enum.create(op.get_bind(), checkfirst=True)
    component_source_enum.create(op.get_bind(), checkfirst=True)

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("timezone", sa.String(63), server_default="UTC", nullable=False),
        sa.Column("onboarding_completed", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- user_conditions ---
    op.create_table(
        "user_conditions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("condition_type", condition_type_enum, nullable=False),
        sa.Column("severity", severity_enum, nullable=True),
        sa.Column("diagnosed_by_doctor", sa.Boolean(), server_default="false"),
        sa.Column("diagnosis_date", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "condition_type"),
    )

    # --- user_known_allergens ---
    op.create_table(
        "user_known_allergens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("allergen_type", component_type_enum, nullable=False),
        sa.Column("confirmed", sa.Boolean(), server_default="false"),
        sa.Column("severity", severity_enum, nullable=True),
        sa.Column("reaction_notes", sa.Text(), nullable=True),
        sa.Column("first_reaction_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "allergen_type"),
    )

    # --- user_settings ---
    op.create_table(
        "user_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("notification_prefs", postgresql.JSONB(), server_default="{}"),
        sa.Column("symptom_prompt_intervals", postgresql.JSONB(), server_default="{}"),
        sa.Column("daily_checkin_time", sa.Time(), server_default="09:00:00"),
        sa.Column("preferred_language", sa.String(5), server_default="'en'"),
        sa.Column("units_preference", sa.String(10), server_default="'metric'"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- food_database ---
    op.create_table(
        "food_database",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("subcategory", sa.String(100), nullable=True),
        sa.Column("common_names", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("allergen_profile", postgresql.JSONB(), nullable=True),
        sa.Column("preparation_modifiers", postgresql.JSONB(), nullable=True),
        sa.Column("cross_reactivity_groups", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("source_references", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("last_verified", sa.Date(), nullable=True),
        sa.Column("verified_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- food_component_details ---
    op.create_table(
        "food_component_details",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("food_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("food_database.id", ondelete="CASCADE"), nullable=False),
        sa.Column("component_type", component_type_enum, nullable=False),
        sa.Column("level_score", sa.Numeric(3, 1), nullable=True),
        sa.Column("estimated_value", sa.Numeric(12, 4), nullable=True),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("source_reference", sa.String(500), nullable=True),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("food_id", "component_type"),
    )

    # --- component_definitions ---
    op.create_table(
        "component_definitions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("component_type", component_type_enum, unique=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("common_symptoms", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("typical_onset_hours", sa.Numeric(5, 2), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- meals ---
    op.create_table(
        "meals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("photo_url", sa.Text(), nullable=True),
        sa.Column("raw_description", sa.Text(), nullable=True),
        sa.Column("ai_parsed_description", sa.Text(), nullable=True),
        sa.Column("meal_type", meal_type_enum, nullable=False),
        sa.Column("confidence_score", sa.Numeric(3, 2), server_default="0.0"),
        sa.Column("processing_status", processing_status_enum, server_default="'pending'"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- meal_items ---
    op.create_table(
        "meal_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("meal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("meals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("food_name", sa.String(255), nullable=False),
        sa.Column("food_database_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("food_database.id", ondelete="SET NULL"), nullable=True),
        sa.Column("quantity", sa.Numeric(8, 2), nullable=True),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("preparation_method", sa.String(255), nullable=True),
        sa.Column("brand", sa.String(255), nullable=True),
        sa.Column("confidence_score", sa.Numeric(3, 2), server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- meal_item_components ---
    op.create_table(
        "meal_item_components",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("meal_item_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("meal_items.id", ondelete="CASCADE"), nullable=False),
        sa.Column("component_type", component_type_enum, nullable=False),
        sa.Column("level", sa.Numeric(3, 1), server_default="0"),
        sa.Column("source", component_source_enum, server_default="'database'"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("meal_item_id", "component_type"),
    )

    # --- ai_conversations ---
    op.create_table(
        "ai_conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("meal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("meals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("messages", postgresql.JSONB(), nullable=False),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_estimate", sa.Numeric(8, 6), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- symptom_scores ---
    op.create_table(
        "symptom_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("meal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("meals.id", ondelete="SET NULL"), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symptom_type", symptom_type_enum, nullable=False),
        sa.Column("vas_score", sa.Numeric(3, 0), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("prompt_type", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- daily_checkins ---
    op.create_table(
        "daily_checkins",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("check_date", sa.Date(), nullable=False),
        sa.Column("overall_wellness", sa.Numeric(3, 0), nullable=True),
        sa.Column("stress_level", sa.Numeric(3, 0), nullable=True),
        sa.Column("sleep_quality", sa.Numeric(3, 0), nullable=True),
        sa.Column("sleep_hours", sa.Numeric(3, 1), nullable=True),
        sa.Column("bowel_movements", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "check_date"),
    )

    # --- trigger_predictions ---
    op.create_table(
        "trigger_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("component_type", component_type_enum, nullable=False),
        sa.Column("confidence_score", sa.Numeric(3, 0), nullable=False),
        sa.Column("evidence_count", sa.Integer(), server_default="0"),
        sa.Column("first_detected", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("status", trigger_status_enum, server_default="'suspect'"),
        sa.Column("symptom_types", postgresql.ARRAY(symptom_type_enum), nullable=True),
        sa.Column("average_time_lag_minutes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "component_type"),
    )

    # --- correlation_events ---
    op.create_table(
        "correlation_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trigger_prediction_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("trigger_predictions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("meal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("meals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symptom_score_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("symptom_scores.id", ondelete="CASCADE"), nullable=False),
        sa.Column("time_lag_hours", sa.Numeric(5, 2), nullable=True),
        sa.Column("component_exposure_level", sa.Numeric(3, 1), nullable=True),
        sa.Column("symptom_severity", sa.Numeric(3, 0), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- insight_reports ---
    op.create_table(
        "insight_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_type", report_type_enum, nullable=False),
        sa.Column("date_range_start", sa.Date(), nullable=False),
        sa.Column("date_range_end", sa.Date(), nullable=False),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("json_summary", postgresql.JSONB(), nullable=True),
        sa.Column("shared_with_clinician", sa.Boolean(), server_default="false"),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- report_shares ---
    op.create_table(
        "report_shares",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("report_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("insight_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("share_method", share_method_enum, nullable=False),
        sa.Column("share_token", sa.String(255), unique=True, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- Indexes for common queries ---
    op.create_index("ix_meals_user_id", "meals", ["user_id"])
    op.create_index("ix_meals_timestamp", "meals", ["timestamp"])
    op.create_index("ix_symptom_scores_user_id", "symptom_scores", ["user_id"])
    op.create_index("ix_symptom_scores_timestamp", "symptom_scores", ["timestamp"])
    op.create_index("ix_trigger_predictions_user_id", "trigger_predictions", ["user_id"])
    op.create_index("ix_food_database_name", "food_database", ["name"])
    op.create_index("ix_insight_reports_user_id", "insight_reports", ["user_id"])


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("report_shares")
    op.drop_table("insight_reports")
    op.drop_table("correlation_events")
    op.drop_table("trigger_predictions")
    op.drop_table("daily_checkins")
    op.drop_table("symptom_scores")
    op.drop_table("ai_conversations")
    op.drop_table("meal_item_components")
    op.drop_table("meal_items")
    op.drop_table("meals")
    op.drop_table("component_definitions")
    op.drop_table("food_component_details")
    op.drop_table("food_database")
    op.drop_table("user_settings")
    op.drop_table("user_known_allergens")
    op.drop_table("user_conditions")
    op.drop_table("users")

    # Drop enum types
    component_source_enum.drop(op.get_bind(), checkfirst=True)
    severity_enum.drop(op.get_bind(), checkfirst=True)
    share_method_enum.drop(op.get_bind(), checkfirst=True)
    report_type_enum.drop(op.get_bind(), checkfirst=True)
    trigger_status_enum.drop(op.get_bind(), checkfirst=True)
    component_type_enum.drop(op.get_bind(), checkfirst=True)
    condition_type_enum.drop(op.get_bind(), checkfirst=True)
    processing_status_enum.drop(op.get_bind(), checkfirst=True)
    symptom_type_enum.drop(op.get_bind(), checkfirst=True)
    meal_type_enum.drop(op.get_bind(), checkfirst=True)
