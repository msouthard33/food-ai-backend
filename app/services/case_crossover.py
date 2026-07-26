"""Within-person, week-stratified case-crossover trigger engine (bake-off winner).

This is the food-level scorer that won the Phase 3 bake-off
(``scripts/trigger_engine_bakeoff.py``, engine ``E6``): the only candidate that both
**exonerates an innocent food sharing a trigger's component** (the failure mode the
per-component hierarchical model is structurally blind to) and keeps false positives
at/below the component model on every scenario — including a pure-noise diary.

Design
------
Per **food**, a self-matched case-crossover:

* Each meal is attributed to the day its symptoms are expected (``meal_time + onset``,
  the same onset-shift the hierarchical engine uses) — exposure lines up with the
  outcome it causes, not the day the food was eaten.
* Days are **stratified by ISO week**. Within a week, week-level confounders (a bad
  week, travel, a med change) are held constant, so only within-week exposure↔symptom
  discordance carries signal. Cochran–Mantel–Haenszel combines the per-week 2x2s into
  one odds ratio + p-value. This is a standard, self-matched epidemiological test
  (DiGA / NICE-defensible), not a bespoke model.
* A **Benjamini–Hochberg FDR** across the tested foods controls the multiple-comparison
  false-positive rate (the raw per-food test flags far too many foods).

Decoupled rank / flag
---------------------
The 0–100 ``score`` separates two jobs that a single threshold cannot serve at once:

* **Flag** (``score >= SUSPECT_FLOOR``, i.e. surfaced as a suspect): only when the food
  clears the *strict* ``FDR_FLAG_Q``. This suppresses chance coincidences on a
  no-trigger diary.
* **Rank**: a food that is positive but sub-threshold still gets a strength-based score
  ``(1 − p)`` held *strictly below* ``SUSPECT_FLOOR`` — so a real but borderline trigger
  keeps its ranking (sorts above safe foods) without being flagged as a false positive.

Everything is pure Python + ``math`` (no numpy, no scipy — the chi-square survival is
``erfc``), and fully deterministic: identical diary → identical scores, every run.

Cold start
----------
This engine is **data-driven** — a food needs within-person exposure/symptom
discordance to be testable. A brand-new or low-data patient has nothing to score here;
the per-component hierarchical model (``hierarchical_trigger``) remains the cold-start /
onboarding-prior path, and the caller (``insights.get_suspect_foods``) falls back to it
for any food this engine cannot test.
"""

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.meal import Meal
from app.models.symptom import SymptomScore
from app.models.user import UserCondition
from app.services.hierarchical_trigger import (
    DEFAULT_LOOKBACK_DAYS,
    _condition_keys_for,
    _onset_lag_hours,
)

# ── Tunable constants ─────────────────────────────────────────────────────────

#: Strict Benjamini–Hochberg q at/below which a food is FLAGGED as a suspect
#: (score >= SUSPECT_FLOOR). Deliberately tighter than a ranking cutoff — it is what
#: keeps chance coincidences on a no-trigger diary from surfacing.
FDR_FLAG_Q = 0.05

#: A flagged food scores at least this; a positive-but-sub-threshold food is scored
#: strictly below it (rank-only). Matches the leaderboard's "suspect" cut.
SUSPECT_FLOOR = 20.0

#: Minimum days in a week-stratum for it to be informative (a 2x2 needs ≥2 days).
_MIN_STRATUM_DAYS = 2


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FoodCaseCrossoverResult:
    """Case-crossover association result for one (user, food name).

    Fields:
        food_name: the logged food name tested.
        odds_ratio: Mantel–Haenszel combined OR (Haldane-corrected cells; >1 raises
            symptom odds). 1.0 when the food has no informative stratum.
        ci_low / ci_high: 95% CI on the OR (Robins–Breslow–Greenland SE).
        p_value: Cochran–Mantel–Haenszel p-value (continuity-corrected, df=1).
        q_value: Benjamini–Hochberg adjusted p across the tested foods.
        score: 0–100 decoupled rank/flag score (see module docstring).
        flagged: True iff OR>1 and q_value <= FDR_FLAG_Q (surfaced as a suspect).
        n_exposed_symptom_days: exposed-and-symptom day count (informative evidence).
        n_strata: number of informative week-strata contributing to the estimate.
        testable: False when the food had no informative stratum (caller should fall
            back to the component model for this food).
        method: engine tag.
    """

    food_name: str
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float
    score: float
    flagged: bool
    n_exposed_symptom_days: int
    n_strata: int
    testable: bool
    method: str = "case_crossover_cmh_fdr"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Benjamini–Hochberg (mirrors assoc_guardrail.benjamini_hochberg) ────────────

def _bh_adjust(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values in input order. Empty input → empty list."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, p_values[idx] * m / rank)
        adjusted[idx] = min(1.0, running_min)
    return adjusted


# ── Pure scoring core (no DB; independently testable) ──────────────────────────

def score_case_crossover(
    days_food: dict[date, set[str]],
    symptom_days: set[date],
    foods: set[str],
) -> list[FoodCaseCrossoverResult]:
    """Score each food in ``foods`` by a week-stratified CMH case-crossover + BH-FDR.

    Args:
        days_food: onset-shifted exposure — ``day -> set of food names present`` (a meal
            is already binned to ``(meal_time + onset).date()`` by the caller).
        symptom_days: the set of calendar days with a recorded symptom.
        foods: the candidate food names to score (e.g. the ≥3-episode qualifiers).

    Returns:
        One ``FoodCaseCrossoverResult`` per food, sorted by score descending then name
        (deterministic tie-break). Pure and deterministic.
    """
    data_days = set(days_food) | symptom_days
    if not data_days or not foods:
        return [
            FoodCaseCrossoverResult(
                food_name=f, odds_ratio=1.0, ci_low=0.0, ci_high=0.0, p_value=1.0,
                q_value=1.0, score=0.0, flagged=False, n_exposed_symptom_days=0,
                n_strata=0, testable=False,
            )
            for f in sorted(foods)
        ]

    # Every calendar day in the observed span is a row: a day with no trigger meal and
    # no symptom is a genuine unexposed control (same day-grid the component model uses),
    # so a food is compared against real control days, not only the days it was eaten.
    d0, d1 = min(data_days), max(data_days)
    all_days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    # ISO (year, week) time strata.
    strata: dict[tuple[int, int], list[date]] = {}
    for day in all_days:
        strata.setdefault(day.isocalendar()[:2], []).append(day)

    ordered = sorted(foods)
    p_values: list[float] = []
    # food -> (OR, se, n_a, n_strata, informative)
    stats: dict[str, tuple[float, float, int, int, bool]] = {}

    for food in ordered:
        cmh_num = 0.0      # Σ (a − E[a]) on RAW counts
        cmh_var = 0.0      # Σ Var(a)
        R = S = 0.0        # Robins–Breslow–Greenland accumulators (Haldane cells)
        sum_pr = sum_psqr = sum_qs = 0.0
        n_a = 0            # exposed-and-symptom days
        n_strata = 0
        for days in strata.values():
            a = b = c = d = 0
            for day in days:
                exposed = food in days_food.get(day, set())
                symptom = day in symptom_days
                if exposed and symptom:
                    a += 1
                elif exposed:
                    b += 1
                elif symptom:
                    c += 1
                else:
                    d += 1
            n = a + b + c + d
            if n < _MIN_STRATUM_DAYS:
                continue
            n_a += a
            row1 = a + b
            col1 = a + c
            cmh_num += a - row1 * col1 / n
            cmh_var += row1 * (c + d) * col1 * (b + d) / (n * n * (n - 1))
            aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5   # Haldane for OR + SE
            nn = aa + bb + cc + dd
            p_i = (aa + dd) / nn
            q_i = (bb + cc) / nn
            r_i = aa * dd / nn
            s_i = bb * cc / nn
            R += r_i
            S += s_i
            sum_pr += p_i * r_i
            sum_psqr += p_i * s_i + q_i * r_i
            sum_qs += q_i * s_i
            n_strata += 1

        odds_ratio = R / S if S > 0 else 1.0
        chi2 = max(0.0, (abs(cmh_num) - 0.5)) ** 2 / cmh_var if cmh_var > 0 else 0.0
        p_value = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
        if R > 0 and S > 0:
            var = sum_pr / (2 * R * R) + sum_psqr / (2 * R * S) + sum_qs / (2 * S * S)
            se = math.sqrt(max(var, 0.0))
        else:
            se = 0.0
        # "Informative" = at least one stratum had exposure-and-outcome discordance
        # (both row and column margins non-zero), i.e. a real comparison. A food eaten
        # only on symptom days (or only on non-symptom days) has cmh_var == 0 — no
        # contrast — so it is NOT testable and the caller falls back to the component model.
        informative = cmh_var > 0.0
        p_values.append(p_value)
        stats[food] = (odds_ratio, se, n_a, n_strata, informative)

    q_values = _bh_adjust(p_values)

    results: list[FoodCaseCrossoverResult] = []
    for i, food in enumerate(ordered):
        odds_ratio, se, n_a, n_strata, informative = stats[food]
        p_value = p_values[i]
        q_value = q_values[i]
        ci_low = odds_ratio * math.exp(-1.96 * se)
        ci_high = odds_ratio * math.exp(1.96 * se)
        flagged = odds_ratio > 1.0 and q_value <= FDR_FLAG_Q
        if flagged:
            score = SUSPECT_FLOOR + (1.0 - q_value) * (100.0 - SUSPECT_FLOOR)
        elif odds_ratio > 1.0:
            # positive but sub-threshold: rank by strength, held strictly below the floor
            score = (1.0 - p_value) * (SUSPECT_FLOOR - 2.0)
        else:
            score = 0.0
        results.append(
            FoodCaseCrossoverResult(
                food_name=food,
                odds_ratio=odds_ratio,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                q_value=q_value,
                score=score,
                flagged=flagged,
                n_exposed_symptom_days=n_a,
                n_strata=n_strata,
                testable=informative,
            )
        )

    results.sort(key=lambda r: (-r.score, r.food_name))
    return results


# ── DB entry point ─────────────────────────────────────────────────────────────

async def analyze_case_crossover_triggers(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    candidate_foods: set[str] | None = None,
) -> list[FoodCaseCrossoverResult]:
    """Load a user's diary and score every candidate food by the case-crossover engine.

    Args:
        db: async session (read-only).
        user_id: user to analyse.
        lookback_days: window; meals/symptoms older than this are ignored.
        candidate_foods: restrict scoring to these logged food names (e.g. the
            suspect-foods qualifiers). When None, every logged food is scored.

    Returns:
        ``FoodCaseCrossoverResult`` per candidate food, sorted by score. Read-only,
        deterministic. Meals are onset-shifted by the user's condition-typical onset
        (``_onset_lag_hours``) so exposure aligns with the symptom day.
    """
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

    condition_types = (
        await db.execute(
            select(UserCondition.condition_type).where(UserCondition.user_id == user_id)
        )
    ).scalars().all()
    onset_hours = _onset_lag_hours(_condition_keys_for(list(condition_types)))
    shift = timedelta(hours=onset_hours)

    meals = list(
        (
            await db.execute(
                select(Meal)
                .where(
                    and_(
                        Meal.user_id == user_id,
                        Meal.timestamp >= cutoff,
                        Meal.deleted_at.is_(None),
                    )
                )
                .options(selectinload(Meal.items))
            )
        ).scalars().unique().all()
    )

    # Onset-shifted exposure: day -> set of logged food names.
    days_food: dict[date, set[str]] = {}
    for meal in meals:
        day = (meal.timestamp + shift).date()
        bucket = days_food.setdefault(day, set())
        for item in meal.items:
            if item.name:
                bucket.add(item.name)

    symptom_times = (
        await db.execute(
            select(SymptomScore.timestamp).where(
                and_(
                    SymptomScore.user_id == user_id,
                    SymptomScore.timestamp >= cutoff,
                    SymptomScore.deleted_at.is_(None),
                )
            )
        )
    ).scalars().all()
    symptom_days = {ts.date() for ts in symptom_times}

    if candidate_foods is None:
        foods = {name for names in days_food.values() for name in names}
    else:
        foods = set(candidate_foods)

    return score_case_crossover(days_food, symptom_days, foods)
