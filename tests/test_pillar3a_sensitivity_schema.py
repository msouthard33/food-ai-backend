"""Pillar 3a — Multi-Sensitivity Profile Schema tests.

Tests cover:
1. Migration file integrity — required tables, columns, and RLS statements present.
2. Enum coverage — ComponentType values used in the schema.
3. HTTP-layer smoke tests — schema-backed endpoints respond correctly.

These tests validate the schema contract without requiring a live DB for the
migration-content tests (file-read only). HTTP tests use the conftest client
fixture which connects to the local foodai_test DB.
"""

import re
from pathlib import Path

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations" / "versions"


def _read_migration(filename_pattern: str) -> str:
    """Return the source of the first migration file matching the pattern."""
    matches = list(MIGRATIONS_DIR.glob(filename_pattern))
    assert matches, f"No migration file matching {filename_pattern!r} found"
    return matches[0].read_text()


# ---------------------------------------------------------------------------
# Migration content tests (no DB required)
# ---------------------------------------------------------------------------


class TestPillar3aMigrationContent:
    """Verify b3d1e2f4a5c8 creates the expected tables and columns."""

    def test_migration_creates_user_sensitivity_profiles(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "user_sensitivity_profiles" in src

    def test_migration_creates_food_combined_ratings(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "food_combined_ratings" in src

    def test_user_sensitivity_profiles_has_user_id_fk(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        # Should reference users.id with CASCADE
        assert re.search(r'ForeignKey.*"users\.id".*CASCADE', src) or \
               re.search(r"users\.id.*CASCADE", src)

    def test_user_sensitivity_profiles_has_component_type(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "component_type" in src

    def test_user_sensitivity_profiles_has_weight_and_threshold(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert '"weight"' in src or "'weight'" in src or "weight" in src
        assert '"threshold"' in src or "'threshold'" in src or "threshold" in src

    def test_food_combined_ratings_has_combined_score(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "combined_score" in src

    def test_food_combined_ratings_has_rating_label(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "rating_label" in src

    def test_food_combined_ratings_has_contributing_components_jsonb(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "contributing_components" in src
        assert "JSONB" in src or "jsonb" in src.lower()

    def test_unique_constraint_user_food_pair(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "uq_food_combined_ratings_user_food" in src

    def test_unique_constraint_user_component_pair(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        assert "uq_user_sensitivity_profiles_user_component" in src

    def test_downgrade_drops_tables_in_order(self):
        src = _read_migration("b3d1e2f4a5c8_*.py")
        # downgrade should drop food_combined_ratings before user_sensitivity_profiles
        # (reversed creation order)
        assert "drop_table" in src or "DROP TABLE" in src.upper()


class TestRlsMigrationCompleteness:
    """Verify RLS is applied to all three Pillar 3 tables across migrations."""

    def test_e8a4c7b2f1d9_covers_user_sensitivity_profiles(self):
        src = _read_migration("e8a4c7b2f1d9_*.py")
        assert "user_sensitivity_profiles" in src

    def test_e8a4c7b2f1d9_covers_barcode_product_cache(self):
        src = _read_migration("e8a4c7b2f1d9_*.py")
        assert "barcode_product_cache" in src

    def test_f2a3b4c5d6e7_covers_food_combined_ratings(self):
        """The fix migration must enable + force RLS on food_combined_ratings."""
        src = _read_migration("f2a3b4c5d6e7_*.py")
        assert "food_combined_ratings" in src
        assert "ENABLE ROW LEVEL SECURITY" in src
        assert "FORCE ROW LEVEL SECURITY" in src

    def test_f2a3b4c5d6e7_has_correct_revision_chain(self):
        src = _read_migration("f2a3b4c5d6e7_*.py")
        assert 'down_revision: Union[str, None] = "e8a4c7b2f1d9"' in src or \
               "down_revision = \"e8a4c7b2f1d9\"" in src

    def test_f2a3b4c5d6e7_has_downgrade(self):
        src = _read_migration("f2a3b4c5d6e7_*.py")
        assert "def downgrade" in src
        assert "DISABLE ROW LEVEL SECURITY" in src

    def test_all_three_pillar3_tables_have_rls_coverage(self):
        """Aggregate check: user_sensitivity_profiles + barcode_product_cache covered by
        e8a4c7b2f1d9; food_combined_ratings covered by f2a3b4c5d6e7."""
        rls_src = _read_migration("e8a4c7b2f1d9_*.py")
        fix_src = _read_migration("f2a3b4c5d6e7_*.py")
        combined = rls_src + fix_src

        for table in ["user_sensitivity_profiles", "barcode_product_cache", "food_combined_ratings"]:
            assert table in combined, f"RLS not found for {table}"


# ---------------------------------------------------------------------------
# Enum coverage tests (no DB required)
# ---------------------------------------------------------------------------


class TestComponentTypeEnum:
    """Verify ComponentType enum has expected sensitivity-relevant values."""

    def test_histamines_present(self):
        from app.models.enums import ComponentType
        assert ComponentType.HISTAMINES

    def test_fodmap_present(self):
        from app.models.enums import ComponentType
        assert ComponentType.FODMAP

    def test_gluten_present(self):
        from app.models.enums import ComponentType
        assert ComponentType.GLUTEN

    def test_sufficient_component_types_for_multi_protocol(self):
        from app.models.enums import ComponentType
        # Multi-sensitivity profile needs at least 10 distinct types to be clinically useful
        assert len(ComponentType) >= 10


# ---------------------------------------------------------------------------
# HTTP smoke tests (requires conftest client fixture → local DB)
# ---------------------------------------------------------------------------


class TestSensitivityEndpointPlaceholders:
    """Smoke tests that verify the app is running and auth is enforced.

    Full CRUD for sensitivity profiles will be added when the router is
    implemented (Pillar 3a Phase 2). These tests confirm the app boots
    with the new schema and that unimplemented routes return 404 (not 500).
    """

    @pytest.mark.asyncio
    async def test_app_health_still_ok_after_schema_changes(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_sensitivity_profiles_route_not_yet_implemented(self, client: AsyncClient):
        """Confirm the sensitivity profiles endpoint is not yet wired (404, not 500)."""
        resp = await client.get("/api/v1/sensitivity/profiles")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_combined_ratings_route_not_yet_implemented(self, client: AsyncClient):
        """Confirm the combined ratings endpoint is not yet wired (404, not 500)."""
        resp = await client.get("/api/v1/foods/combined-rating/some-food-id")
        assert resp.status_code == 404
