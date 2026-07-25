"""Synthetic patient data generator for cold-start population prior.

Generates realistic synthetic food diary data with known ground-truth
trigger-symptom pairings for pre-seeding the trigger detection engine.

The synthetic cohort is inserted as separate User rows with is_synthetic=True
so the decay rule can exclude them once a real user accumulates enough data.

Wave 2, Pillar 1: Differentiation — cold-start data prevents the "empty shelf"
problem for new users who have not yet logged enough meals for trigger inference.
"""

import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ComponentType, MealType, ProcessingStatus, SymptomType
from app.models.meal import Meal, MealItem
from app.models.sensitivity import UserSensitivityProfile
from app.models.symptom import SymptomScore
from app.models.user import User

logger = logging.getLogger(__name__)


# ── Condition configurations ──────────────────────────────────────────────────
# Symptom types use canonical SymptomType enum values from enums.py.
# Clinical notes on mappings vs. medical terminology:
#   DIARRHEA / CONSTIPATION → SymptomType.BOWEL_CHANGES
#   FLUSHING / HIVES        → SymptomType.SKIN_REACTION
#   ABDOMINAL_PAIN          → SymptomType.PAIN

CONDITION_CONFIGS: dict[str, dict] = {
    "ibs": {
        "primary_triggers": ["fodmap", "lactose", "fructose", "sorbitol"],
        "lag_hours": {"min": 2.0, "max": 36.0, "peak": 8.0},  # triangular distribution
        "symptom_types": [
            SymptomType.BLOATING,
            SymptomType.PAIN,
            SymptomType.BOWEL_CHANGES,
        ],
        "trigger_probability": 0.70,
    },
    "mcas": {
        "primary_triggers": ["histamines", "salicylates", "oxalates"],
        "lag_hours": {
            "bimodal": True,
            "mode_1": {"min": 0.08, "max": 0.5, "weight": 0.4},   # immediate (5–30 min)
            "mode_2": {"min": 2.0, "max": 6.0, "weight": 0.6},    # delayed
        },
        "symptom_types": [
            SymptomType.SKIN_REACTION,
            SymptomType.PAIN,
            SymptomType.FATIGUE,
        ],
        "trigger_probability": 0.55,
    },
    "histamine_intolerance": {
        "primary_triggers": ["histamines"],
        "lag_hours": {"min": 2.0, "max": 12.0, "peak": 4.0},  # triangular distribution
        "symptom_types": [
            SymptomType.HEADACHE,
            SymptomType.SKIN_REACTION,
            SymptomType.NAUSEA,
        ],
        "trigger_probability": 0.60,
    },
    "food_allergy": {
        "primary_triggers": ["gluten", "dairy", "eggs", "peanuts"],
        "lag_hours": {"min": 0.08, "max": 2.0, "peak": 0.5},  # triangular distribution
        "symptom_types": [
            SymptomType.SKIN_REACTION,
            SymptomType.PAIN,
            SymptomType.NAUSEA,
        ],
        "trigger_probability": 0.85,
    },
}

# Condition mix for cohort generation
SYNTHETIC_CONDITION_MIX: list[tuple[str, float]] = [
    ("ibs", 0.40),
    ("histamine_intolerance", 0.25),
    ("mcas", 0.20),
    ("food_allergy", 0.15),
]

# ── KB allergen profile key → trigger string mapping ─────────────────────────
# Maps the KB JSON allergen_profile keys to the string keys used in CONDITION_CONFIGS.
# "sorbitol" covers fodmap_polyols (sorbitol is a FODMAP polyol).

_KB_KEY_TO_TRIGGER: dict[str, str | None] = {
    "gluten": "gluten",
    "dairy": "dairy",
    "soy": "soy",
    "egg": "eggs",
    "tree_nuts": "tree_nuts",
    "peanuts": "peanuts",
    "fish": "fish",
    "shellfish": "shellfish",
    "histamine": "histamines",   # KB singular → trigger plural
    "fodmap_fructans": "fodmap",
    "fodmap_gos": "fodmap",
    "fodmap_lactose": "lactose",
    "fodmap_fructose": "fructose",
    "fodmap_polyols": "sorbitol",  # polyols include sorbitol
    "salicylates": "salicylates",
    "oxalates": "oxalates",
    "lectins": "lectins",
    "sulfites": "sulfites",
    "amines": "amines",
    "nickel": None,  # no ComponentType for nickel — skip
}

# ── Trigger string → ComponentType enum mapping ───────────────────────────────
# Used when inserting UserSensitivityProfile rows.
# "sorbitol" → FODMAP because sorbitol is a FODMAP polyol (no dedicated ComponentType).

COMPONENT_STR_TO_ENUM: dict[str, ComponentType] = {
    "fodmap": ComponentType.FODMAP,
    "lactose": ComponentType.LACTOSE,
    "fructose": ComponentType.FRUCTOSE,
    "sorbitol": ComponentType.FODMAP,      # polyol, mapped to FODMAP
    "histamines": ComponentType.HISTAMINES,
    "salicylates": ComponentType.SALICYLATES,
    "oxalates": ComponentType.OXALATES,
    "gluten": ComponentType.GLUTEN,
    "dairy": ComponentType.MILK_DAIRY,
    "eggs": ComponentType.EGGS,
    "peanuts": ComponentType.PEANUTS,
    "soy": ComponentType.SOY,
    "tree_nuts": ComponentType.TREE_NUTS,
    "fish": ComponentType.FISH,
    "shellfish": ComponentType.SHELLFISH,
    "amines": ComponentType.AMINES,
    "lectins": ComponentType.LECTINS,
    "sulfites": ComponentType.SULFITES,
}

# Minimum allergen score (0–100) to include a food in the trigger index.
# Score >= 20 catches "low" and above; "very_low" (≈5–15) is excluded
# to keep trigger foods clinically meaningful.
_TRIGGER_SCORE_THRESHOLD = 20


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sample_lag_hours(condition: str, rng: random.Random) -> float:
    """Sample a symptom onset lag (in hours) from the condition's distribution."""
    cfg = CONDITION_CONFIGS.get(condition, CONDITION_CONFIGS["ibs"])
    lag = cfg.get("lag_hours", {})

    if lag.get("bimodal"):
        mode_1 = lag["mode_1"]
        mode_2 = lag["mode_2"]
        if rng.random() < mode_1["weight"]:
            return rng.uniform(mode_1["min"], mode_1["max"])
        else:
            return rng.uniform(mode_2["min"], mode_2["max"])
    else:
        # Triangular distribution
        return rng.triangular(lag["min"], lag["max"], lag["peak"])


def _infer_meal_type(ts: datetime) -> MealType:
    """Infer meal type from timestamp hour."""
    hour = ts.hour
    if 5 <= hour < 11:
        return MealType.BREAKFAST
    elif 11 <= hour < 15:
        return MealType.LUNCH
    elif 15 <= hour < 18:
        return MealType.SNACK
    else:
        return MealType.DINNER


# ── Public API ────────────────────────────────────────────────────────────────

def load_kb_food_index(kb_path: str) -> tuple[dict[str, list[str]], list[str]]:
    """Load the knowledge base and return a component food index plus safe foods.

    Returns:
        (component_index, safe_foods)

        component_index: maps trigger string name → list of food names with that
            component present at a meaningful level (score >= 20).
            Example: {"histamines": ["Aged Cheese", "Red Wine", ...], ...}

        safe_foods: list of food names not indexed under any trigger component.
            Used to fill non-trigger meal slots.
    """
    with open(kb_path, encoding="utf-8") as fh:
        kb = json.load(fh)

    component_index: dict[str, list[str]] = {}
    trigger_food_names: set[str] = set()
    all_food_names: list[str] = []

    for food in kb.get("foods", []):
        name = food.get("name", "").strip()
        if not name:
            continue
        all_food_names.append(name)

        allergen_profile = food.get("allergen_profile", {})
        food_triggers: set[str] = set()

        for kb_key, trigger_str in _KB_KEY_TO_TRIGGER.items():
            if trigger_str is None:
                continue
            entry = allergen_profile.get(kb_key)
            if not entry:
                continue
            score = entry.get("score", 0)
            if score >= _TRIGGER_SCORE_THRESHOLD:
                food_triggers.add(trigger_str)

        for tstr in food_triggers:
            if tstr not in component_index:
                component_index[tstr] = []
            if name not in component_index[tstr]:
                component_index[tstr].append(name)
            trigger_food_names.add(name)

    safe_foods = [n for n in all_food_names if n not in trigger_food_names]
    return component_index, safe_foods


def generate_patient_profile(
    conditions: list[str],
    patient_index: int,
    seed: int,
    kb_food_index: dict[str, list[str]],
    safe_foods: list[str],
) -> dict:
    """Generate a synthetic patient profile with ground-truth trigger components.

    Args:
        conditions: List of condition strings (e.g., ["ibs"] or ["ibs", "mcas"]).
        patient_index: Unique index for this patient (used in email and seeding).
        seed: Random seed for reproducibility.
        kb_food_index: Component trigger index from load_kb_food_index().
        safe_foods: List of safe food names from load_kb_food_index().

    Returns:
        dict with keys:
            index, conditions, trigger_components, sensitivity_weights,
            trigger_foods, safe_foods, logging_consistency
    """
    rng = random.Random(seed + patient_index * 97)

    # Collect all possible triggers from all conditions (deduped, ordered)
    all_triggers: list[str] = []
    for cond in conditions:
        cfg = CONDITION_CONFIGS.get(cond, {})
        for t in cfg.get("primary_triggers", []):
            if t not in all_triggers:
                all_triggers.append(t)

    if not all_triggers:
        all_triggers = ["fodmap"]

    # Select 1–3 trigger components
    n_triggers = rng.randint(1, min(3, len(all_triggers)))
    selected_triggers = rng.sample(all_triggers, n_triggers)

    # Assign sensitivity weights 0.4–1.0 per component
    sensitivity_weights = {
        t: round(rng.uniform(0.4, 1.0), 2)
        for t in selected_triggers
    }

    # Collect trigger foods from KB index for the selected components
    trigger_food_pool: list[str] = []
    for t in selected_triggers:
        for food_name in kb_food_index.get(t, []):
            if food_name not in trigger_food_pool:
                trigger_food_pool.append(food_name)

    n_trigger_foods = min(rng.randint(8, 15), len(trigger_food_pool))
    selected_trigger_foods = (
        rng.sample(trigger_food_pool, n_trigger_foods)
        if trigger_food_pool
        else []
    )

    # Select 20–40 safe foods the patient eats regularly
    n_safe = min(rng.randint(20, 40), len(safe_foods))
    selected_safe_foods = rng.sample(safe_foods, n_safe) if safe_foods else []

    return {
        "index": patient_index,
        "conditions": conditions,
        "trigger_components": selected_triggers,
        "sensitivity_weights": sensitivity_weights,
        "trigger_foods": selected_trigger_foods,
        "safe_foods": selected_safe_foods,
        "logging_consistency": round(rng.uniform(0.65, 0.95), 2),
    }


def generate_patient_diary(
    profile: dict,
    kb_food_index: dict[str, list[str]],
    start_date: datetime,
    num_weeks: int = 8,
) -> dict:
    """Generate a synthetic food diary with meal and symptom events.

    Meal timing uses random.gauss with 30-minute std dev centered on:
        Breakfast: 8 am   (7–9 am window)
        Lunch:     12:30 pm (12–1:30 pm window)
        Dinner:    7 pm   (6–8:30 pm window)
        Snack:     3:30 pm (3–4 pm window, 40 % of days)

    Trigger foods appear in ~30 % of meal item slots. After each meal
    containing a trigger food, a symptom is generated with probability
    trigger_probability × sensitivity_weight. Lag is sampled from the
    condition-specific distribution. VAS score is triangular(2, 10, 6).

    Returns:
        {
          "meals": [
            {
              "timestamp": datetime,
              "foods": ["Food A", "Food B"],
              "logged": bool,
              "meal_type": str,
            },
            ...
          ],
          "symptoms": [
            {
              "timestamp": datetime,
              "symptom_type": SymptomType,
              "vas_score": int,
              "lag_hours": float,                    # for test validation
              "triggering_meal_timestamp": datetime, # for test validation
            },
            ...
          ],
        }
    """
    # Use a different seed offset than profile generation
    rng = random.Random(profile["index"] * 113 + 5000)

    primary_condition = profile["conditions"][0] if profile["conditions"] else "ibs"
    cfg = CONDITION_CONFIGS.get(primary_condition, CONDITION_CONFIGS["ibs"])

    trigger_foods_set: set[str] = set(profile["trigger_foods"])
    safe_food_pool: list[str] = profile["safe_foods"]
    trigger_food_pool: list[str] = profile["trigger_foods"]
    trigger_components: list[str] = profile["trigger_components"]
    sensitivity_weights: dict[str, float] = profile["sensitivity_weights"]
    logging_consistency: float = profile["logging_consistency"]
    trigger_probability: float = cfg["trigger_probability"]

    # Day base — strip to date boundary, preserve tz
    base = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    meals: list[dict] = []
    symptoms: list[dict] = []

    total_days = num_weeks * 7

    # Meal schedule: (center_minutes_from_midnight, window_min, window_max)
    MEAL_SCHEDULE = [
        ("breakfast", 480, 420, 540),   # 8 am, window 7–9 am
        ("lunch",     750, 720, 810),   # 12:30 pm, window 12–1:30 pm
        ("dinner",   1140, 1080, 1230), # 7 pm, window 6–8:30 pm
    ]
    SNACK_SCHED = ("snack", 930, 900, 960)  # 3:30 pm, 3–4 pm

    for day_offset in range(total_days):
        day_base = base + timedelta(days=day_offset)

        day_meal_slots: list[tuple[str, datetime]] = []

        for meal_label, center, mn, mx in MEAL_SCHEDULE:
            offset_min = int(rng.gauss(center - (day_base.replace(hour=0, minute=0) -
                                                  day_base.replace(hour=0, minute=0)).total_seconds() / 60,
                                       30))
            # Simpler: offset in minutes from midnight, clamped
            raw = int(rng.gauss(center, 30))
            clamped = max(mn, min(mx, raw))
            ts = day_base + timedelta(minutes=clamped)
            day_meal_slots.append((meal_label, ts))

        # Optional snack (40 % of days)
        if rng.random() < 0.40:
            label, center, mn, mx = SNACK_SCHED
            raw = int(rng.gauss(center, 15))
            clamped = max(mn, min(mx, raw))
            ts = day_base + timedelta(minutes=clamped)
            day_meal_slots.append((label, ts))

        day_meal_slots.sort(key=lambda x: x[1])

        for meal_label, meal_ts in day_meal_slots:
            n_items = rng.randint(1, 4)
            selected_foods: list[str] = []

            for _ in range(n_items):
                if trigger_food_pool and rng.random() < 0.30:
                    selected_foods.append(rng.choice(trigger_food_pool))
                elif safe_food_pool:
                    selected_foods.append(rng.choice(safe_food_pool))

            if not selected_foods:
                continue

            logged = rng.random() < logging_consistency
            meals.append({
                "timestamp": meal_ts,
                "foods": selected_foods,
                "logged": logged,
                "meal_type": meal_label,
            })

            # Symptom generation for meals with trigger foods
            meal_trigger_foods = [f for f in selected_foods if f in trigger_foods_set]
            if not meal_trigger_foods:
                continue

            # Identify which of the patient's trigger components are relevant
            relevant_components = [
                c for c in trigger_components
                if any(f in kb_food_index.get(c, []) for f in meal_trigger_foods)
            ]
            if not relevant_components:
                continue

            component = rng.choice(relevant_components)
            sensitivity = sensitivity_weights.get(component, 0.5)

            if rng.random() < trigger_probability * sensitivity:
                lag = _sample_lag_hours(primary_condition, rng)
                symptom_ts = meal_ts + timedelta(hours=lag)
                vas = int(rng.triangular(2, 10, 6))
                symptom_type = rng.choice(cfg["symptom_types"])

                symptoms.append({
                    "timestamp": symptom_ts,
                    "symptom_type": symptom_type,
                    "vas_score": vas,
                    "lag_hours": lag,                           # diagnostic metadata
                    "triggering_meal_timestamp": meal_ts,       # diagnostic metadata
                })

    return {"meals": meals, "symptoms": symptoms}


async def build_food_name_to_id_map(db: AsyncSession) -> dict[str, uuid.UUID]:
    """Query food_database table and return {name: UUID} mapping.

    Used in insert_synthetic_patient to resolve KB food names to real DB IDs.
    Foods not present in the DB are simply omitted — never fabricate UUIDs.
    """
    from app.models.food import FoodEntry

    result = await db.execute(select(FoodEntry.name, FoodEntry.id))
    return {row.name: row.id for row in result}


async def insert_synthetic_patient(
    db: AsyncSession,
    profile: dict,
    diary: dict,
    food_name_to_id: dict[str, uuid.UUID],
    component_type_map: dict[str, ComponentType],
) -> uuid.UUID:
    """Insert one synthetic patient and their diary into the database.

    Insertion order maintains FK integrity:
        1. users row (is_synthetic=True)
        2. user_sensitivity_profiles rows (deduped by ComponentType to avoid
           unique-constraint violations when two trigger strings map to the
           same ComponentType, e.g. "fodmap" and "sorbitol" → ComponentType.FODMAP)
        3. meals + meal_items (logged=True meals only; foods not found in
           food_name_to_id are silently skipped — no fabricated UUIDs)
        4. symptom_scores rows

    Returns:
        UUID of the newly created User.
    """
    user_id = uuid.uuid4()
    # Unique email suffix so re-runs / batched top-ups never collide on
    # users_email_key (the deterministic synthetic_{index} scheme collides
    # when resuming a partially-completed seed). is_synthetic remains the
    # authoritative marker.
    user = User(
        id=user_id,
        email=f"synthetic_{profile['index']}_{user_id.hex[:8]}@foodai.internal",
        is_synthetic=True,
    )
    db.add(user)
    await db.flush()

    # Sensitivity profiles — deduped by ComponentType
    seen_ctypes: set[ComponentType] = set()
    for comp_str in profile["trigger_components"]:
        ctype = component_type_map.get(comp_str)
        if ctype is None or ctype in seen_ctypes:
            continue
        seen_ctypes.add(ctype)
        weight = Decimal(str(round(profile["sensitivity_weights"].get(comp_str, 0.70), 2)))
        db.add(
            UserSensitivityProfile(
                user_id=user_id,
                component_type=ctype,
                weight=weight,
            )
        )
    await db.flush()

    # Meals and items
    for meal_data in diary["meals"]:
        if not meal_data.get("logged", False):
            continue

        ts: datetime = meal_data["timestamp"]
        meal_type_str: str = meal_data.get("meal_type", "dinner")
        try:
            meal_type = MealType(meal_type_str)
        except ValueError:
            meal_type = _infer_meal_type(ts)

        meal = Meal(
            user_id=user_id,
            timestamp=ts,
            meal_type=meal_type,
            confidence_score=Decimal("1.0"),
            processing_status=ProcessingStatus.COMPLETE,
        )
        db.add(meal)
        await db.flush()

        for food_name in meal_data["foods"]:
            food_id = food_name_to_id.get(food_name)
            if food_id is None:
                logger.debug("Skipping food '%s': not found in food_database", food_name)
                continue
            db.add(
                MealItem(
                    meal_id=meal.id,
                    food_entry_id=food_id,
                    name=food_name,
                )
            )
        await db.flush()

    # Symptom scores
    for sym in diary["symptoms"]:
        db.add(
            SymptomScore(
                user_id=user_id,
                timestamp=sym["timestamp"],
                symptom_type=sym["symptom_type"],
                vas_score=int(sym["vas_score"]),
            )
        )

    await db.commit()
    logger.info("Inserted synthetic patient %s (user_id=%s)", profile["index"], user_id)
    return user_id


async def generate_synthetic_cohort(
    db: AsyncSession,
    kb_path: str,
    n_patients: int = 150,
) -> dict:
    """Generate and insert a full synthetic cohort. Returns summary stats.

    Condition distribution:
        40 % IBS, 25 % histamine_intolerance, 20 % MCAS, 15 % food_allergy.
    30 % of patients receive a second comorbid condition.
    Data history spans 8 weeks before the current timestamp.
    """
    logger.info("Starting synthetic cohort generation: %d patients", n_patients)

    kb_index, safe_foods = load_kb_food_index(kb_path)
    logger.info(
        "KB loaded: %d trigger component buckets, %d safe foods",
        len(kb_index), len(safe_foods),
    )

    food_name_to_id = await build_food_name_to_id_map(db)
    logger.info("Food DB map: %d entries", len(food_name_to_id))

    start_date = datetime.now(timezone.utc) - timedelta(weeks=8)

    rng = random.Random(42)  # deterministic seed for reproducibility
    condition_options = [c for c, _ in SYNTHETIC_CONDITION_MIX]
    condition_weights = [w for _, w in SYNTHETIC_CONDITION_MIX]

    user_ids: list[uuid.UUID] = []
    total_logged_meals = 0
    total_symptoms = 0

    for i in range(n_patients):
        primary = rng.choices(condition_options, weights=condition_weights, k=1)[0]
        conditions = [primary]

        if rng.random() < 0.30:
            others = [c for c in condition_options if c != primary]
            conditions.append(rng.choice(others))

        seed = rng.randint(0, 2**31 - 1)
        profile = generate_patient_profile(conditions, i, seed, kb_index, safe_foods)
        diary = generate_patient_diary(profile, kb_index, start_date)

        user_id = await insert_synthetic_patient(
            db, profile, diary, food_name_to_id, COMPONENT_STR_TO_ENUM
        )

        user_ids.append(user_id)
        total_logged_meals += sum(1 for m in diary["meals"] if m["logged"])
        total_symptoms += len(diary["symptoms"])

        if (i + 1) % 25 == 0:
            logger.info("Generated %d/%d synthetic patients", i + 1, n_patients)

    summary = {
        "n_patients": n_patients,
        "user_ids": [str(uid) for uid in user_ids],
        "total_logged_meals": total_logged_meals,
        "total_symptoms": total_symptoms,
        "avg_meals_per_patient": round(total_logged_meals / max(n_patients, 1), 1),
        "avg_symptoms_per_patient": round(total_symptoms / max(n_patients, 1), 1),
    }
    logger.info("Cohort generation complete: %s", summary)
    return summary
