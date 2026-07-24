"""mySymptoms CSV import — Pillar 4 / Box 12 (Clinician Trust Layer).

Parses a mySymptoms (SkyGazer Labs) food-diary CSV export and creates the
corresponding meals + symptoms for the authenticated user, so a migrating user does
not lose their history.

Documented mySymptoms export layout
------------------------------------
The mySymptoms "Export data -> CSV" produces one row per diary entry. The columns
this importer recognizes (case-insensitive, with common aliases) are:

    Date       - entry date            (aliases: "day")
    Time       - entry time            (aliases: "time of day")
    Type       - entry category        (aliases: "category", "group", "kind")
    Name       - item / description     (aliases: "items", "item", "food",
                                         "description", "detail", "name")
    Severity   - 0-10 rating for symptoms (aliases: "rating", "score", "value",
                                         "intensity", "quantity")
    Note       - free-text note        (aliases: "notes", "comment", "comments")

Type handling:
  * Food / Drink / Meal / Beverage / Snack  -> a Meal (+ one MealItem per item).
    Multiple foods in one row may be separated by ";" or ",".
  * Symptom                                 -> a SymptomScore. mySymptoms uses a
    0-10 severity scale, mapped to the app's 0-100 VAS (severity * 10). The symptom
    name is mapped to a SymptomType enum (best-effort keyword match; OTHER fallback).
  * Any other Type (Medication, Bowel Movement, Exercise, Environment, ...) is
    recorded as a skipped row with a reason (out of scope for meals/symptoms import).

Validation & idempotency:
  * Each row is validated independently; a bad row is reported in ``errors`` with its
    1-based row number and a reason, and does not abort the rest of the import.
  * Import is idempotent-friendly: a meal is skipped if the user already has a
    non-deleted meal at the same timestamp with the same raw_description; a symptom is
    skipped if the user already has a non-deleted symptom at the same timestamp with
    the same symptom_type and VAS score. Re-importing the same file therefore does not
    duplicate rows.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import MealType, SymptomType
from app.models.meal import Meal, MealItem
from app.models.symptom import SymptomScore

# ── column resolution ────────────────────────────────────────────────────────

_COLUMN_ALIASES: dict[str, set[str]] = {
    "date": {"date", "day"},
    "time": {"time", "time of day"},
    "type": {"type", "category", "group", "kind"},
    "name": {"name", "items", "item", "food", "description", "detail"},
    "severity": {"severity", "rating", "score", "value", "intensity", "quantity"},
    "note": {"note", "notes", "comment", "comments"},
}

_MEAL_TYPES = {"food", "drink", "meal", "beverage", "snack", "breakfast", "lunch", "dinner"}
_SYMPTOM_TYPES = {"symptom", "symptoms"}

# Ordered keyword -> SymptomType mapping (first match wins).
_SYMPTOM_KEYWORDS: list[tuple[str, SymptomType]] = [
    ("bloat", SymptomType.BLOATING),
    ("gas", SymptomType.BLOATING),
    ("nausea", SymptomType.NAUSEA),
    ("brain fog", SymptomType.BRAIN_FOG),
    ("fog", SymptomType.BRAIN_FOG),
    ("fatigue", SymptomType.FATIGUE),
    ("tired", SymptomType.FATIGUE),
    ("skin", SymptomType.SKIN_REACTION),
    ("rash", SymptomType.SKIN_REACTION),
    ("itch", SymptomType.SKIN_REACTION),
    ("hives", SymptomType.SKIN_REACTION),
    ("flush", SymptomType.SKIN_REACTION),
    ("bowel", SymptomType.BOWEL_CHANGES),
    ("diarrh", SymptomType.BOWEL_CHANGES),
    ("constipat", SymptomType.BOWEL_CHANGES),
    ("stool", SymptomType.BOWEL_CHANGES),
    ("heartburn", SymptomType.HEARTBURN),
    ("reflux", SymptomType.HEARTBURN),
    ("acid", SymptomType.HEARTBURN),
    ("headache", SymptomType.HEADACHE),
    ("migraine", SymptomType.HEADACHE),
    ("joint", SymptomType.JOINT_PAIN),
    ("respirat", SymptomType.RESPIRATORY),
    ("breath", SymptomType.RESPIRATORY),
    ("wheez", SymptomType.RESPIRATORY),
    ("congest", SymptomType.RESPIRATORY),
    ("pain", SymptomType.PAIN),  # keep generic "pain" late so "joint pain" wins first
    ("cramp", SymptomType.PAIN),
    ("ache", SymptomType.PAIN),
]

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d %b %Y", "%d %B %Y")
_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p", "%I%p")


@dataclass
class ImportResult:
    total_rows: int = 0
    meals_created: int = 0
    symptoms_created: int = 0
    rows_skipped: int = 0
    errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "meals_created": self.meals_created,
            "symptoms_created": self.symptoms_created,
            "rows_skipped": self.rows_skipped,
            "errors": self.errors,
        }


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical keys -> the actual header present in the CSV (case-insensitive)."""
    resolved: dict[str, str] = {}
    lower_to_actual = {fn.strip().lower(): fn for fn in fieldnames if fn}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_actual:
                resolved[canonical] = lower_to_actual[alias]
                break
    return resolved


def _parse_datetime(date_str: str, time_str: str) -> datetime | None:
    """Parse a mySymptoms date + (optional) time into a UTC-aware datetime."""
    date_str = (date_str or "").strip()
    if not date_str:
        return None
    parsed_date = None
    for fmt in _DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if parsed_date is None:
        return None

    time_str = (time_str or "").strip()
    if time_str:
        for fmt in _TIME_FORMATS:
            try:
                t = datetime.strptime(time_str, fmt)
                parsed_date = parsed_date.replace(hour=t.hour, minute=t.minute, second=t.second)
                break
            except ValueError:
                continue
    return parsed_date.replace(tzinfo=timezone.utc)


def _map_symptom_type(name: str) -> SymptomType:
    lowered = (name or "").lower()
    for keyword, symptom_type in _SYMPTOM_KEYWORDS:
        if keyword in lowered:
            return symptom_type
    return SymptomType.OTHER


def _infer_meal_type(type_value: str, ts: datetime) -> MealType:
    tv = (type_value or "").lower()
    if "drink" in tv or "beverage" in tv:
        return MealType.BEVERAGE
    if "breakfast" in tv:
        return MealType.BREAKFAST
    if "lunch" in tv:
        return MealType.LUNCH
    if "dinner" in tv:
        return MealType.DINNER
    if "snack" in tv:
        return MealType.SNACK
    # Fall back to time-of-day heuristic.
    hour = ts.hour
    if 5 <= hour < 11:
        return MealType.BREAKFAST
    if 11 <= hour < 15:
        return MealType.LUNCH
    if 17 <= hour < 22:
        return MealType.DINNER
    return MealType.SNACK


def _split_items(name: str) -> list[str]:
    """Split a mySymptoms item cell into individual food names."""
    parts = [p.strip() for chunk in name.split(";") for p in chunk.split(",")]
    return [p for p in parts if p]


async def import_mysymptoms_csv(
    db: AsyncSession,
    user_id,
    raw_bytes: bytes,
) -> ImportResult:
    """Parse a mySymptoms CSV export and create meals/symptoms for ``user_id``.

    Returns an :class:`ImportResult` summarizing created/skipped rows and row-level
    errors. The caller owns the transaction commit.
    """
    result = ImportResult()

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        result.errors.append({"row": 0, "reason": "CSV has no header row"})
        return result

    cols = _resolve_columns(list(reader.fieldnames))
    missing = [k for k in ("date", "type", "name") if k not in cols]
    if missing:
        result.errors.append(
            {"row": 0, "reason": f"Missing required column(s): {', '.join(missing)}"}
        )
        return result

    for idx, row in enumerate(reader, start=2):  # row 1 is the header
        result.total_rows += 1
        type_value = (row.get(cols["type"], "") or "").strip()
        name_value = (row.get(cols["name"], "") or "").strip()
        date_value = row.get(cols["date"], "")
        time_value = row.get(cols.get("time", ""), "") if "time" in cols else ""
        note_value = (row.get(cols["note"], "") or "").strip() if "note" in cols else ""

        ts = _parse_datetime(date_value, time_value)
        if ts is None:
            result.rows_skipped += 1
            result.errors.append({"row": idx, "reason": f"Unparseable date/time: {date_value!r}"})
            continue

        type_lower = type_value.lower()

        # ── Meal rows ────────────────────────────────────────────────────────
        if type_lower in _MEAL_TYPES:
            if not name_value:
                result.rows_skipped += 1
                result.errors.append({"row": idx, "reason": "Food row has no item name"})
                continue

            raw_description = name_value
            if await _meal_exists(db, user_id, ts, raw_description):
                result.rows_skipped += 1
                continue

            meal = Meal(
                user_id=user_id,
                timestamp=ts,
                meal_type=_infer_meal_type(type_value, ts),
                raw_description=raw_description,
                ai_parsed_description="Imported from mySymptoms",
            )
            db.add(meal)
            await db.flush()
            for item_name in _split_items(name_value):
                db.add(MealItem(meal_id=meal.id, name=item_name[:255]))
            await db.flush()
            result.meals_created += 1
            continue

        # ── Symptom rows ─────────────────────────────────────────────────────
        if type_lower in _SYMPTOM_TYPES or "symptom" in type_lower:
            symptom_type = _map_symptom_type(name_value)
            severity_raw = row.get(cols["severity"], "") if "severity" in cols else ""
            vas = _severity_to_vas(severity_raw)
            notes = name_value if not note_value else f"{name_value} - {note_value}"

            if await _symptom_exists(db, user_id, ts, symptom_type, vas):
                result.rows_skipped += 1
                continue

            db.add(
                SymptomScore(
                    user_id=user_id,
                    timestamp=ts,
                    symptom_type=symptom_type,
                    vas_score=vas,
                    notes=notes[:1000] if notes else None,
                    prompt_type="csv_import",
                )
            )
            await db.flush()
            result.symptoms_created += 1
            continue

        # ── Unsupported types ────────────────────────────────────────────────
        result.rows_skipped += 1
        result.errors.append(
            {"row": idx, "reason": f"Unsupported entry type {type_value!r} (not imported)"}
        )

    return result


def _severity_to_vas(severity_raw: str) -> int:
    """Map a mySymptoms 0-10 severity to a 0-100 VAS score. Clamps to [0, 100].

    Blank/unparseable severity defaults to a mid-scale 50 so the episode is still
    recorded (rather than dropped) for correlation purposes.
    """
    s = (severity_raw or "").strip()
    if not s:
        return 50
    try:
        val = float(s)
    except ValueError:
        return 50
    # mySymptoms uses a 0-10 scale; scale to 0-100. If a value already looks like a
    # 0-100 score (>10), take it as-is.
    vas = val if val > 10 else val * 10
    return int(max(0, min(100, round(vas))))


async def _meal_exists(db: AsyncSession, user_id, ts: datetime, raw_description: str) -> bool:
    existing = await db.execute(
        select(Meal.id).where(
            and_(
                Meal.user_id == user_id,
                Meal.timestamp == ts,
                Meal.raw_description == raw_description,
                Meal.deleted_at.is_(None),
            )
        )
    )
    return existing.first() is not None


async def _symptom_exists(
    db: AsyncSession, user_id, ts: datetime, symptom_type: SymptomType, vas: int
) -> bool:
    existing = await db.execute(
        select(SymptomScore.id).where(
            and_(
                SymptomScore.user_id == user_id,
                SymptomScore.timestamp == ts,
                SymptomScore.symptom_type == symptom_type,
                SymptomScore.vas_score == vas,
                SymptomScore.deleted_at.is_(None),
            )
        )
    )
    return existing.first() is not None
