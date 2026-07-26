"""Seed a small, hand-tuned DEMO cohort for compelling live-demo screens.

Unlike ``generate_synthetic_data.py`` (a large *stochastic* population prior for
cold-start), this script builds a tiny, **legible, deterministic** cohort of a
few named demo patients. Each patient has a clean ~6-week diary engineered so
that 2 real triggers clearly rise to the top of the suspect-foods leaderboard,
their trigger predictions upgrade from cold-start priors to data-driven
confidence, and the clinician PDF has a real story to tell.

Design goals
------------
* **Legible** — safe foods rotate across a wide variety so no incidental food
  blankets the symptom windows; symptoms occur *only* after trigger meals, so
  the planted triggers dominate the leaderboard with a clear gap.
* **Deterministic** — fixed RNG seed per user; identical output every run.
* **Idempotent** — deterministic user UUIDs (uuid5) + delete-before-insert, so
  re-running never duplicates and always yields the same clean state. Emails are
  stable and recognisable (``demo_ibs@foodai.demo``); the deterministic id — not
  the email — is the collision key we key idempotency on.
* **Marked** — demo users carry ``is_synthetic=True`` so the trigger-decay rule
  and any "exclude non-real users" filter treat them like the synthetic cohort.

Safety
------
``--dry-run`` is the DEFAULT. Nothing is written unless ``--write`` is passed.
The script never targets production on its own — it uses ``--db-url`` or the
``DATABASE_URL`` env var and defaults to the local ``foodai_test`` database.

Usage
-----
    # Preview stats only (no DB writes) — DEFAULT:
    python -m scripts.seed_demo_account --dry-run

    # Write to the local test DB (creates tables if missing):
    python -m scripts.seed_demo_account --write --create-tables \\
        --db-url postgresql+asyncpg://USER@localhost/foodai_test

    # Seed PROD (gated — orchestrator only, with an explicit prod DATABASE_URL):
    DATABASE_URL=<prod-url> python -m scripts.seed_demo_account --write
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Ensure the backend app is importable when run as a script or module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Canonical demo identities — shared with the demo-login endpoint so a demo login
# and this seeded data resolve to the SAME user id + email per persona.
from app.routers.auth import demo_user_email, demo_user_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("seed_demo_account")


def _quiet_sql() -> None:
    """Silence SQLAlchemy's echo=True firehose (dev engine) for legible CLI output.

    Must run *after* the app engine is created, since echo=True forces the
    engine logger to INFO at creation time.
    """
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Stable namespace so demo user UUIDs are deterministic across runs/machines.
DEMO_NAMESPACE = uuid.UUID("d3e0d3e0-0000-4000-8000-000000000000")

# Default KB path — the bundled copy that ships in backend/data/.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_KB_PATH = _SCRIPT_DIR.parent / "data" / "allergen_knowledge_base_complete.json"

_DEFAULT_DB_URL = "postgresql+asyncpg://matthewsouthard@localhost/foodai_test"

NUM_WEEKS = 6

# Which weekday-mod slots (day_index % 7) are "trigger days".
# Primary trigger appears on 2 of them per week, secondary on 1 → a clear
# strong/probable gradient on the leaderboard.
_PRIMARY_DAY_MODS = {1, 5}
_SECONDARY_DAY_MODS = {3}
_TRIGGER_DAY_MODS = _PRIMARY_DAY_MODS | _SECONDARY_DAY_MODS


@dataclass
class DemoUserSpec:
    """A single hand-tuned demo patient."""

    slug: str
    # Canonical demo persona shared with the demo-login endpoint (app.routers.auth)
    # so login and seeded data resolve to the SAME user id + email.
    persona: str
    display_name: str
    conditions: list[str]
    # Ordered [primary, secondary] — recognisable KB food names.
    trigger_foods: list[str]
    # ComponentType *value* string the trigger foods load (for MealItemComponent
    # + data-driven trigger analysis). e.g. "fodmap", "histamines".
    trigger_component: str
    # SymptomType value strings to rotate through for this condition.
    symptom_types: list[str]
    # Lag (hours) from trigger meal to symptom onset.
    symptom_lag_hours: float = 6.0

    @property
    def email(self) -> str:
        # Shared with the demo-login endpoint so the same account is reused.
        return demo_user_email(self.persona)

    @property
    def user_id(self) -> uuid.UUID:
        # Shared with the demo-login endpoint — login and seed hit the SAME user.
        return demo_user_id(self.persona)


# ── The demo cohort ──────────────────────────────────────────────────────────
# Trigger foods are verified against the KB at runtime; recognisable names chosen
# so the leaderboard reads clearly in a demo.
DEMO_COHORT: list[DemoUserSpec] = [
    DemoUserSpec(
        slug="demo_ibs",
        persona="ibs",
        display_name="Demo — IBS (FODMAP)",
        conditions=["ibs"],
        trigger_foods=["Garlic", "Onion"],
        trigger_component="fodmap",
        symptom_types=["bloating", "pain", "bowel_changes"],
        symptom_lag_hours=8.0,
    ),
    DemoUserSpec(
        slug="demo_mcas",
        persona="mcas",
        display_name="Demo — MCAS (Histamine)",
        conditions=["mcas"],
        trigger_foods=["Cheese (Cheddar)", "Salmon (Smoked)"],
        trigger_component="histamines",
        symptom_types=["skin_reaction", "headache", "fatigue"],
        symptom_lag_hours=3.0,
    ),
    DemoUserSpec(
        slug="demo_histamine",
        persona="histamine",
        display_name="Demo — Histamine Intolerance",
        conditions=["histamine_intolerance"],
        trigger_foods=["Tuna (Canned)", "Feta Cheese"],
        trigger_component="histamines",
        symptom_types=["headache", "skin_reaction", "nausea"],
        symptom_lag_hours=4.0,
    ),
]

# A wide, recognisable rotation of safe staples. Filtered against the KB safe
# list at runtime; anything not present is dropped and the rest still rotate.
_PREFERRED_SAFE_FOODS: list[str] = [
    "White Rice", "Chicken Breast (Fresh)", "Beef (Lean)", "Turkey (Fresh)",
    "Pork (Fresh)", "Lamb (Fresh)", "Basmati Rice", "Olive Oil", "Butter (Unsalted)",
    "Black Coffee (Brewed)", "Espresso", "Green Tea (Bottled)", "Sparkling Water (Plain)",
    "Ghee (Clarified Butter)", "Coconut Oil", "Duck (Fresh)", "Ground Beef",
    "Herbal Tea (Chamomile)", "Cumin", "Sunflower Oil", "Black Tea (Bottled)",
    "Canola Oil", "Sesame Oil", "Cold Brew Coffee",
]

_MEAL_SCHEDULE = [
    ("breakfast", 8, 0),
    ("lunch", 12, 30),
    ("dinner", 19, 0),
]


def _resolve_safe_foods(kb_safe: list[str]) -> list[str]:
    """Return a WIDE rotation of safe foods (recognisable staples first).

    We use the *entire* KB safe pool so that, cycled across the diary, no single
    safe food repeats within a 72h window. That keeps incidental foods below the
    suspect-foods sample-size threshold, so only the planted triggers surface —
    the whole point of a legible demo.
    """
    safe_set = set(kb_safe)
    chosen = [f for f in _PREFERRED_SAFE_FOODS if f in safe_set]
    for name in kb_safe:
        if name not in chosen:
            chosen.append(name)
    return chosen or list(kb_safe)


def build_demo_diary(
    spec: DemoUserSpec,
    safe_foods: list[str],
    *,
    start_date: datetime,
    num_weeks: int = NUM_WEEKS,
) -> dict:
    """Build a deterministic, hand-tuned diary for one demo patient.

    Returns ``{"meals": [...], "symptoms": [...]}`` where each meal is
    ``{"timestamp", "meal_type", "foods": [{"name", "components"}], "raw_description"}``
    and each symptom is ``{"timestamp", "symptom_type", "vas_score"}``.

    Symptoms occur *only* after trigger meals, so the planted trigger foods
    dominate the suspect-foods leaderboard with a clear gap over safe staples.
    """
    rng = random.Random(f"demo::{spec.slug}")
    safe_pool = _resolve_safe_foods(safe_foods)
    # Per-user deterministic shuffle so each demo patient eats a distinct rotation.
    rotation = safe_pool[:]
    rng.shuffle(rotation)
    safe_idx = 0

    from app.models.enums import ComponentType, MealType, SymptomType

    component_enum = ComponentType(spec.trigger_component)
    primary_food, secondary_food = spec.trigger_foods[0], spec.trigger_foods[1]

    base = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    total_days = num_weeks * 7

    meals: list[dict] = []
    symptoms: list[dict] = []
    vas_cycle = [6, 7, 8, 7]
    sym_cycle = 0
    vas_cycle_idx = 0

    def next_safe() -> str:
        nonlocal safe_idx
        food = rotation[safe_idx % len(rotation)]
        safe_idx += 1
        return food

    for day in range(total_days):
        day_base = base + timedelta(days=day)
        is_trigger_day = (day % 7) in _TRIGGER_DAY_MODS

        for label, hour, minute in _MEAL_SCHEDULE:
            # Deterministic +/- jitter (minutes) so timestamps look human.
            jitter = rng.randint(-12, 12)
            ts = day_base + timedelta(hours=hour, minutes=minute + jitter)

            foods: list[dict] = [
                {"name": next_safe(), "components": []},
                {"name": next_safe(), "components": []},
            ]

            trigger_in_meal: str | None = None
            if is_trigger_day and label == "dinner":
                mod = day % 7
                if mod in _PRIMARY_DAY_MODS:
                    trigger_in_meal = primary_food
                else:
                    trigger_in_meal = secondary_food
                foods.append(
                    {
                        "name": trigger_in_meal,
                        # High load so daily-load + per-meal analysis both fire.
                        "components": [(component_enum, Decimal("80"))],
                    }
                )

            meals.append(
                {
                    "timestamp": ts,
                    "meal_type": MealType(label),
                    "foods": foods,
                    "raw_description": ", ".join(f["name"] for f in foods),
                }
            )

            if trigger_in_meal is not None:
                sym_type = spec.symptom_types[sym_cycle % len(spec.symptom_types)]
                sym_cycle += 1
                vas = vas_cycle[vas_cycle_idx % len(vas_cycle)]
                vas_cycle_idx += 1
                symptoms.append(
                    {
                        "timestamp": ts + timedelta(hours=spec.symptom_lag_hours),
                        "symptom_type": SymptomType(sym_type),
                        "vas_score": vas,
                    }
                )

    return {"meals": meals, "symptoms": symptoms}


def summarize_diary(spec: DemoUserSpec, diary: dict) -> dict:
    """Return per-user dry-run stats + planted-trigger exposure counts."""
    meals = diary["meals"]
    symptoms = diary["symptoms"]
    trigger_exposures: dict[str, int] = {f: 0 for f in spec.trigger_foods}
    for meal in meals:
        for food in meal["foods"]:
            if food["name"] in trigger_exposures:
                trigger_exposures[food["name"]] += 1
    return {
        "slug": spec.slug,
        "email": spec.email,
        "user_id": str(spec.user_id),
        "conditions": spec.conditions,
        "planted_triggers": spec.trigger_foods,
        "trigger_component": spec.trigger_component,
        "n_meals": len(meals),
        "n_symptoms": len(symptoms),
        "trigger_exposures": trigger_exposures,
    }


# ── DB write path ─────────────────────────────────────────────────────────────

async def _create_tables_if_needed() -> None:
    import app.models  # noqa: F401 — register all ORM models
    from app.database import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def write_demo_user(
    session,
    spec: DemoUserSpec,
    diary: dict,
    food_name_to_id: dict[str, uuid.UUID],
    *,
    run_analysis: bool = True,
) -> dict:
    """Delete-then-insert one demo patient and their diary. Idempotent."""
    from sqlalchemy import delete, or_

    from app.models.enums import MealType, ProcessingStatus
    from app.models.meal import Meal, MealItem, MealItemComponent
    from app.models.symptom import SymptomScore
    from app.models.user import User
    from app.services.trigger_service import run_full_analysis, seed_condition_priors

    # Idempotency: drop any prior instance (cascades to all PHI rows). Match on
    # BOTH the deterministic id and the stable email so a demo row seeded by an
    # earlier scheme (different id, same email) is also replaced cleanly.
    await session.execute(
        delete(User).where(or_(User.id == spec.user_id, User.email == spec.email))
    )
    await session.commit()

    user = User(
        id=spec.user_id,
        email=spec.email,
        display_name=spec.display_name,
        is_synthetic=True,
        onboarding_completed=True,
    )
    session.add(user)
    await session.flush()

    # Cold-start priors, exactly as onboarding would seed them (commits internally).
    await seed_condition_priors(session, spec.user_id, spec.conditions)

    n_items = 0
    n_components = 0
    for meal_data in diary["meals"]:
        meal = Meal(
            user_id=spec.user_id,
            timestamp=meal_data["timestamp"],
            meal_type=MealType(meal_data["meal_type"]),
            raw_description=meal_data.get("raw_description"),
            confidence_score=Decimal("1.0"),
            processing_status=ProcessingStatus.COMPLETE,
        )
        session.add(meal)
        await session.flush()

        for food in meal_data["foods"]:
            item = MealItem(
                meal_id=meal.id,
                food_entry_id=food_name_to_id.get(food["name"]),
                name=food["name"],
                confidence_score=Decimal("1.0"),
            )
            session.add(item)
            await session.flush()
            n_items += 1
            for ctype, level in food.get("components", []):
                session.add(
                    MealItemComponent(
                        meal_item_id=item.id,
                        component_type=ctype,
                        estimated_level=level,
                        confidence_score=Decimal("0.9"),
                    )
                )
                n_components += 1

    for sym in diary["symptoms"]:
        session.add(
            SymptomScore(
                user_id=spec.user_id,
                timestamp=sym["timestamp"],
                symptom_type=sym["symptom_type"],
                vas_score=int(sym["vas_score"]),
            )
        )

    await session.commit()

    if run_analysis:
        # Look back far enough to cover the full demo history.
        await run_full_analysis(session, spec.user_id, lookback_days=NUM_WEEKS * 7 + 3)

    return {
        "slug": spec.slug,
        "user_id": str(spec.user_id),
        "n_meals": len(diary["meals"]),
        "n_meal_items": n_items,
        "n_components": n_components,
        "n_symptoms": len(diary["symptoms"]),
    }


async def run_write(kb_path: str, db_url: str, create_tables: bool) -> None:
    from app.database import async_session_factory
    from app.services.synthetic_data_generator import (
        build_food_name_to_id_map,
        load_kb_food_index,
    )

    _quiet_sql()

    if create_tables:
        await _create_tables_if_needed()

    kb_index, safe_foods = load_kb_food_index(kb_path)
    _validate_triggers(kb_index)
    start_date = datetime.now(UTC) - timedelta(weeks=NUM_WEEKS)

    results = []
    async with async_session_factory() as session:
        food_name_to_id = await build_food_name_to_id_map(session)
        for spec in DEMO_COHORT:
            diary = build_demo_diary(spec, safe_foods, start_date=start_date)
            res = await write_demo_user(session, spec, diary, food_name_to_id)
            logger.info("Seeded %s: %s", spec.slug, res)
            results.append(res)

    print("\n" + "=" * 64)
    print(f"DEMO SEED COMPLETE — {len(results)} demo users written")
    print(f"  target DB: {_redact(db_url)}")
    print("=" * 64)
    for res in results:
        print(
            f"  {res['slug']:<16} meals={res['n_meals']:>3} "
            f"items={res['n_meal_items']:>3} components={res['n_components']:>3} "
            f"symptoms={res['n_symptoms']:>3}"
        )
    print("=" * 64)


def run_dry(kb_path: str) -> None:
    from app.services.synthetic_data_generator import load_kb_food_index

    _quiet_sql()
    kb_index, safe_foods = load_kb_food_index(kb_path)
    _validate_triggers(kb_index)
    start_date = datetime.now(UTC) - timedelta(weeks=NUM_WEEKS)

    print("\n" + "=" * 64)
    print(f"DRY RUN — {len(DEMO_COHORT)} demo users ({NUM_WEEKS}-week diary each)")
    print("=" * 64)
    for spec in DEMO_COHORT:
        diary = build_demo_diary(spec, safe_foods, start_date=start_date)
        s = summarize_diary(spec, diary)
        print(f"\n  {s['slug']}  <{s['email']}>  id={s['user_id']}")
        print(f"    conditions:       {s['conditions']}")
        print(f"    planted triggers: {s['planted_triggers']}  ({s['trigger_component']})")
        print(f"    meals:            {s['n_meals']}")
        print(f"    symptoms:         {s['n_symptoms']}")
        print(f"    trigger exposures:{s['trigger_exposures']}")
    print("\n" + "=" * 64)
    print("No data written. Re-run with --write to seed the database.")
    print("=" * 64)


def _validate_triggers(kb_index: dict[str, list[str]]) -> None:
    """Fail fast if a planted trigger food is missing from the KB component index."""
    problems: list[str] = []
    for spec in DEMO_COHORT:
        names = set(kb_index.get(spec.trigger_component, []))
        for food in spec.trigger_foods:
            if food not in names:
                problems.append(
                    f"{spec.slug}: '{food}' not in KB '{spec.trigger_component}' index"
                )
    if problems:
        raise SystemExit(
            "Planted trigger foods are not KB-aligned:\n  - " + "\n  - ".join(problems)
        )


def _redact(db_url: str) -> str:
    """Hide any password in a DB URL before printing."""
    if "@" in db_url and "://" in db_url:
        scheme, rest = db_url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return db_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a hand-tuned DEMO cohort for live-demo screens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only (DEFAULT)")
    parser.add_argument("--write", action="store_true", help="Actually write to the DB")
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables before writing (local/test DBs only)",
    )
    parser.add_argument(
        "--kb-path", default=str(_DEFAULT_KB_PATH), help="Path to the KB JSON"
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy async DB URL (default: $DATABASE_URL or local foodai_test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.kb_path):
        raise SystemExit(f"KB file not found: {args.kb_path}")

    if not args.write:
        # Dry-run needs the app importable, so a DB URL must parse even though
        # nothing is written. Use a harmless default.
        os.environ.setdefault("DATABASE_URL", _DEFAULT_DB_URL)
        os.environ.setdefault("APP_ENV", "development")
        run_dry(args.kb_path)
        return

    db_url = args.db_url or os.environ.get("DATABASE_URL") or _DEFAULT_DB_URL
    os.environ["DATABASE_URL"] = db_url
    os.environ.setdefault("APP_ENV", "development")
    asyncio.run(run_write(args.kb_path, db_url, args.create_tables))


if __name__ == "__main__":
    main()
