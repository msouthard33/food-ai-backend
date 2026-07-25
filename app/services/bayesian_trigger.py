"""Beta-Binomial Bayesian trigger-association engine (Wave 2, core stats).

Replaces the frequentist proportion + Wilson-interval approach (see
``trigger_service.calculate_confidence``) with a proper component-level
Beta-Binomial model. For each (user, ComponentType) over a lookback window it
computes:

  * ``trigger_probability`` — P(this component is a real trigger), defined as
    ``P(symptom-rate-when-exposed  >  symptom-rate-when-unexposed)``. This is the
    de-confounding quantity: a food merely eaten a lot no longer scores high, only
    a food whose exposure *raises* the symptom rate above its unexposed baseline.
  * ``score`` — ``trigger_probability * 100`` so the existing D9
    ``confidence_to_tier_label`` (0–1) and 0–100 UI scales still apply.
  * ``ci_low`` / ``ci_high`` — the 95% credible interval on the symptom-rate-given-
    exposure (0–100), an interpretable, non-negative interval.

Design choices worth reviewing (surfaced in the sprint report):

  Exposure unit = CALENDAR DAY, not meal. Days de-confound (a food eaten thrice a
  day is one exposure day, not three) and bound the binomial n by the diary length.

  Exposure derivation = ``MealItem.food_entry_id -> FoodComponentDetail.level``
  (0–4 scale). Synthetic meals carry no ``MealItemComponent`` rows, so we must go
  through the food KB join. A day counts as "exposed" to a component when any meal
  that day contains a food whose ``FoodComponentDetail.level >= 2.0`` ("moderate+"
  on the 0–4 scale). We deliberately do NOT use ``MealItemComponent.estimated_level``
  (a different, 0–100 per-meal scale) — resolving the 0–4 vs 0–100 ambiguity in
  favour of the KB-sourced 0–4 ``level``.

  Outcome = a symptom within a condition-appropriate lag window after the exposure
  day. We use ONE generous window per condition (the max lag from
  ``CONDITION_CONFIGS``); MCAS's bimodal immediate+delayed profile is collapsed to
  its outer bound here and flagged as a future refinement.

Priors:
  * exposed rate ``Beta(alpha0, beta0)`` — mean = population rate that this
    component is a real trigger (cohort aggregate over ``is_synthetic`` users'
    ``UserSensitivityProfile``), nudged up when the user's ``UserCondition``
    implicates the component via ``CONDITION_PRIORS``. Strength ``kappa`` (~6
    pseudo-observations).
  * unexposed rate ``Beta(alpha_u, beta_u)`` — a separate, weak prior at a low
    background symptom rate.

The module is pure/read-only: it issues SELECTs and returns dataclasses. No writes,
no side effects, fully deterministic (same DB state -> identical numbers).
"""

import bisect
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enums import ComponentType, ConditionType
from app.models.food import FoodComponentDetail, FoodEntry
from app.models.meal import Meal
from app.models.sensitivity import UserSensitivityProfile
from app.models.symptom import SymptomScore
from app.models.user import User, UserCondition
from app.services.synthetic_data_generator import CONDITION_CONFIGS
from app.services.trigger_service import CONDITION_PRIORS
from app.utils.confidence import beta_credible_interval, beta_mean, prob_beta_exceeds

# ── Tunable model constants ───────────────────────────────────────────────────

#: FoodComponentDetail.level (0–4 scale) at/above which a food counts as an
#: exposure for its component on that day. 2.0 = "moderate+".
EXPOSURE_LEVEL_THRESHOLD = 2.0

#: Prior strength (pseudo-observations) behind the population exposed-rate prior.
#: Small so a handful of real observations can move the posterior.
PRIOR_STRENGTH_KAPPA = 6.0

#: Extra pseudo-successes added to alpha0 when the user's condition implicates the
#: component (via CONDITION_PRIORS). Shifts the prior mean up without over-committing.
CONDITION_NUDGE = 3.0

#: Floor on prior alpha/beta so both stay strictly positive (Beta requires a,b > 0)
#: even when a population rate is 0.0 or 1.0.
MIN_PRIOR = 0.5

#: Fallback population trigger rate for a component absent from the cohort table.
DEFAULT_POPULATION_RATE = 0.10

#: Weak prior on the UNEXPOSED (background) symptom rate: mean and strength.
BACKGROUND_SYMPTOM_RATE = 0.20
UNEXPOSED_PRIOR_STRENGTH = 2.0

#: Fallback outcome window (hours after the exposure day's start) when the user has
#: no recorded condition. Generous so delayed reactions are not missed.
DEFAULT_MAX_LAG_HOURS = 48.0

#: Default lookback window for an analysis run.
DEFAULT_LOOKBACK_DAYS = 90


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentTriggerResult:
    """Bayesian association result for one (user, ComponentType).

    Fields:
        component_type: the ComponentType analysed.
        trigger_probability: P(exposed rate > unexposed rate), 0–1.
        score: trigger_probability * 100 (0–100; feeds D9 tier mapping).
        ci_low / ci_high: 95% credible interval on the symptom-rate-given-exposure,
            expressed 0–100.
        a / b / c / d: the day-level 2x2 counts —
            a = exposed days with a symptom outcome,
            b = exposed days without,
            c = unexposed days with a symptom outcome,
            d = unexposed days without.
        n_exposed_days: a + b (days the component was consumed above threshold).
        n_symptom_days: total analysed days with a symptom outcome (a + c).
        alpha_post / beta_post: posterior params of the exposed-rate Beta.
        alpha_unexposed_post / beta_unexposed_post: posterior params of the
            unexposed-rate Beta.
        prior_alpha / prior_beta: the exposed-rate PRIOR params (pre-data), retained
            so Sprint 2/3 can persist / audit the prior that was applied.
        is_cold_start: True when there were no observed days (result is the prior).
    """

    component_type: ComponentType
    trigger_probability: float
    score: float
    ci_low: float
    ci_high: float
    a: int
    b: int
    c: int
    d: int
    n_exposed_days: int
    n_symptom_days: int
    alpha_post: float
    beta_post: float
    alpha_unexposed_post: float
    beta_unexposed_post: float
    prior_alpha: float
    prior_beta: float
    is_cold_start: bool

    def to_dict(self) -> dict:
        """Plain-dict view (enum rendered as its string value)."""
        d = asdict(self)
        d["component_type"] = self.component_type.value
        return d


# ── Population prior table ────────────────────────────────────────────────────

async def build_population_prior_table(
    db: AsyncSession,
) -> dict[ComponentType, float]:
    """Aggregate the synthetic cohort into a per-component 'real trigger' rate.

    Rate = (# synthetic users with a UserSensitivityProfile row for the component)
    / (total # synthetic users). ``UserSensitivityProfile`` is the synthetic
    ground truth: a row means that component is genuinely a trigger for that patient.

    Returns an empty dict when there is no synthetic cohort; callers then fall back
    to ``DEFAULT_POPULATION_RATE``. Reusable by Sprint 2/3 (build once, pass in).
    """
    total_synthetic = (
        await db.execute(
            select(User.id).where(User.is_synthetic.is_(True))
        )
    ).scalars().all()
    n_users = len(total_synthetic)
    if n_users == 0:
        return {}

    synthetic_ids = set(total_synthetic)

    rows = (
        await db.execute(
            select(
                UserSensitivityProfile.component_type,
                UserSensitivityProfile.user_id,
            ).where(UserSensitivityProfile.user_id.in_(synthetic_ids))
        )
    ).all()

    # Count DISTINCT users per component (a user may have one row per component).
    users_by_component: dict[ComponentType, set[uuid.UUID]] = {}
    for component_type, user_id in rows:
        users_by_component.setdefault(component_type, set()).add(user_id)

    return {
        comp: len(users) / n_users
        for comp, users in users_by_component.items()
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _condition_keys_for(condition_types: list[ConditionType]) -> list[str]:
    """Map ConditionType enum values to the lowercase keys used by CONDITION_PRIORS
    / CONDITION_CONFIGS (e.g. ConditionType.IBS -> "ibs")."""
    return [ct.value for ct in condition_types]


def _implicated_components(condition_keys: list[str]) -> set[ComponentType]:
    """Union of ComponentTypes implicated by the user's conditions (CONDITION_PRIORS)."""
    implicated: set[ComponentType] = set()
    for key in condition_keys:
        implicated.update(CONDITION_PRIORS.get(key, []))
    return implicated


def _max_lag_hours(condition_keys: list[str]) -> float:
    """Widest symptom-onset lag (hours) across the user's conditions.

    Pulled from CONDITION_CONFIGS: triangular configs use ``max``; the bimodal MCAS
    config uses its later mode's ``max``. No condition -> DEFAULT_MAX_LAG_HOURS.
    """
    lags: list[float] = []
    for key in condition_keys:
        cfg = CONDITION_CONFIGS.get(key)
        if not cfg:
            continue
        lag = cfg.get("lag_hours", {})
        if lag.get("bimodal"):
            # Collapse the bimodal profile to its outer bound (future refinement:
            # score the immediate and delayed modes separately).
            lags.append(float(lag["mode_2"]["max"]))
        elif "max" in lag:
            lags.append(float(lag["max"]))
    return max(lags) if lags else DEFAULT_MAX_LAG_HOURS


def _build_exposed_prior(
    population_rate: float, implicated: bool
) -> tuple[float, float]:
    """Beta(alpha0, beta0) prior on the symptom-rate-when-exposed.

    Mean = population 'real trigger' rate, total strength = kappa; a condition match
    adds CONDITION_NUDGE pseudo-successes. Both params floored at MIN_PRIOR.
    """
    alpha0 = population_rate * PRIOR_STRENGTH_KAPPA
    beta0 = (1.0 - population_rate) * PRIOR_STRENGTH_KAPPA
    if implicated:
        alpha0 += CONDITION_NUDGE
    return max(alpha0, MIN_PRIOR), max(beta0, MIN_PRIOR)


def _unexposed_prior() -> tuple[float, float]:
    """Weak Beta prior on the background (unexposed) symptom rate."""
    alpha_u = BACKGROUND_SYMPTOM_RATE * UNEXPOSED_PRIOR_STRENGTH
    beta_u = (1.0 - BACKGROUND_SYMPTOM_RATE) * UNEXPOSED_PRIOR_STRENGTH
    return max(alpha_u, MIN_PRIOR), max(beta_u, MIN_PRIOR)


async def _load_component_levels(
    db: AsyncSession, food_ids: set[uuid.UUID]
) -> dict[uuid.UUID, dict[ComponentType, float]]:
    """Map each food_entry_id -> {ComponentType: level (0–4)} from the KB."""
    if not food_ids:
        return {}
    rows = (
        await db.execute(
            select(
                FoodComponentDetail.food_entry_id,
                FoodComponentDetail.component_type,
                FoodComponentDetail.level,
            ).where(FoodComponentDetail.food_entry_id.in_(food_ids))
        )
    ).all()
    out: dict[uuid.UUID, dict[ComponentType, float]] = {}
    for food_id, component_type, level in rows:
        if level is None:
            continue
        out.setdefault(food_id, {})[component_type] = float(level)
    return out


def _daily_component_exposure(
    meals: list[Meal],
    component_levels: dict[uuid.UUID, dict[ComponentType, float]],
    name_to_food_id: dict[str, uuid.UUID] | None = None,
) -> tuple[set[date], dict[ComponentType, set[date]]]:
    """Per-day component exposure from meals + KB component levels.

    Returns ``(meal_days, exposed_days)`` where ``meal_days`` is every calendar day
    the user logged at least one meal (the day universe, so unexposed days exist),
    and ``exposed_days[component]`` is the subset of days on which some food carried
    that component at level >= EXPOSURE_LEVEL_THRESHOLD.

    A meal item is resolved to a KB food by ``food_entry_id`` when present, else by a
    case-insensitive name match against the food KB (``name_to_food_id``). The name
    fallback matters because items logged through the API/AI pipeline carry a free-text
    ``name`` but a NULL ``food_entry_id`` — without it, real diaries would register no
    exposure and every component would collapse to its prior.
    """
    name_to_food_id = name_to_food_id or {}
    meal_days: set[date] = set()
    exposed_days: dict[ComponentType, set[date]] = {}
    for meal in meals:
        day = meal.timestamp.date()
        meal_days.add(day)
        for item in meal.items:
            food_id = item.food_entry_id
            if food_id is None:
                food_id = name_to_food_id.get((item.name or "").strip().lower())
            if food_id is None:
                continue
            for comp, level in component_levels.get(food_id, {}).items():
                if level >= EXPOSURE_LEVEL_THRESHOLD:
                    exposed_days.setdefault(comp, set()).add(day)
    return meal_days, exposed_days


async def _resolve_names_to_food_ids(
    db: AsyncSession, names: set[str]
) -> dict[str, uuid.UUID]:
    """Map lowercased food names -> a KB ``FoodEntry.id`` (case-insensitive).

    Used to attribute exposure for meal items that were logged by name only (NULL
    ``food_entry_id``). When several KB rows share a name the smallest id is chosen
    for determinism. Returns an empty dict for an empty input.
    """
    cleaned = {n.strip().lower() for n in names if n and n.strip()}
    if not cleaned:
        return {}
    rows = (
        await db.execute(
            select(FoodEntry.id, FoodEntry.name).where(
                func.lower(FoodEntry.name).in_(cleaned)
            )
        )
    ).all()
    out: dict[str, uuid.UUID] = {}
    for food_id, name in sorted(rows, key=lambda r: str(r[0])):
        key = name.strip().lower()
        # first (smallest id, by the sort above) wins -> deterministic
        out.setdefault(key, food_id)
    return out


def _symptom_outcome_days(
    meal_times: list[datetime],
    symptom_times: list[datetime],
    max_lag_hours: float,
) -> set[date]:
    """Days whose outcome is positive: a symptom followed one of that day's meals.

    Each symptom is attributed to exactly ONE day — the calendar day of the most
    recent meal at or before the symptom, provided the gap is within
    ``max_lag_hours``. Single-attribution (rather than marking every day whose wide
    window happens to contain the symptom) is what prevents an unexposed day from
    "stealing" the symptom that a following exposure day actually caused, which
    would destroy the de-confounding. It also handles delayed, cross-midnight
    reactions correctly: the symptom lands on its triggering meal's day, not the
    calendar day it occurred on.
    """
    if not meal_times:
        return set()
    ordered = sorted(meal_times)
    outcome: set[date] = set()
    for ts in symptom_times:
        idx = bisect.bisect_right(ordered, ts) - 1
        if idx < 0:
            continue
        meal_t = ordered[idx]
        if (ts - meal_t).total_seconds() / 3600.0 <= max_lag_hours:
            outcome.add(meal_t.date())
    return outcome


def _score_component(
    component_type: ComponentType,
    exposed_days: set[date],
    meal_days: set[date],
    outcome_days: set[date],
    exposed_prior: tuple[float, float],
    unexposed_prior: tuple[float, float],
) -> ComponentTriggerResult:
    """Assemble the 2x2, form posteriors, and compute the Bayesian outputs."""
    unexposed_days = meal_days - exposed_days

    a = len(exposed_days & outcome_days)
    b = len(exposed_days) - a
    c = len(unexposed_days & outcome_days)
    d = len(unexposed_days) - c

    alpha0, beta0 = exposed_prior
    alpha_u0, beta_u0 = unexposed_prior

    alpha_post = alpha0 + a
    beta_post = beta0 + b
    alpha_u_post = alpha_u0 + c
    beta_u_post = beta_u0 + d

    trigger_probability = prob_beta_exceeds(
        alpha_post, beta_post, alpha_u_post, beta_u_post
    )
    ci_low, ci_high = beta_credible_interval(alpha_post, beta_post)

    return ComponentTriggerResult(
        component_type=component_type,
        trigger_probability=trigger_probability,
        score=trigger_probability * 100.0,
        ci_low=ci_low * 100.0,
        ci_high=ci_high * 100.0,
        a=a,
        b=b,
        c=c,
        d=d,
        n_exposed_days=len(exposed_days),
        n_symptom_days=len(meal_days & outcome_days),
        alpha_post=alpha_post,
        beta_post=beta_post,
        alpha_unexposed_post=alpha_u_post,
        beta_unexposed_post=beta_u_post,
        prior_alpha=alpha0,
        prior_beta=beta0,
        is_cold_start=(a + b + c + d) == 0,
    )


# ── Public engine entry point ─────────────────────────────────────────────────

async def analyze_bayesian_triggers(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    population_prior: dict[ComponentType, float] | None = None,
) -> list[ComponentTriggerResult]:
    """Compute Beta-Binomial trigger associations for every candidate component.

    Args:
        db: async session (read-only usage).
        user_id: user to analyse.
        lookback_days: window length; meals/symptoms older than this are ignored.
        population_prior: optional precomputed cohort rate table
            (``build_population_prior_table``). Built on demand when omitted.

    Returns:
        List of ``ComponentTriggerResult``, one per candidate component (components
        observed in the user's diary UNION components implicated by their
        conditions), sorted by ``score`` descending. Empty only when there is
        neither data nor an implicating condition. On cold start (no diary days) the
        results carry the priors, so condition-implicated components (e.g. FODMAP /
        lactose for IBS) return an elevated score rather than zero.
    """
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

    # Conditions -> nudge + lag window
    condition_types = (
        await db.execute(
            select(UserCondition.condition_type).where(
                UserCondition.user_id == user_id
            )
        )
    ).scalars().all()
    condition_keys = _condition_keys_for(list(condition_types))
    implicated = _implicated_components(condition_keys)
    max_lag = _max_lag_hours(condition_keys)

    # Population priors (cohort aggregate)
    if population_prior is None:
        population_prior = await build_population_prior_table(db)

    # Meals + items in the window
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

    food_ids = {
        item.food_entry_id
        for meal in meals
        for item in meal.items
        if item.food_entry_id is not None
    }
    # Items logged by name only (NULL food_entry_id) — resolve to KB foods so they
    # still register exposure. Without this, API/AI-logged diaries score prior-only.
    unlinked_names = {
        item.name
        for meal in meals
        for item in meal.items
        if item.food_entry_id is None and item.name
    }
    name_to_food_id = await _resolve_names_to_food_ids(db, unlinked_names)
    food_ids |= set(name_to_food_id.values())
    component_levels = await _load_component_levels(db, food_ids)
    meal_days, exposed_days = _daily_component_exposure(
        meals, component_levels, name_to_food_id
    )

    # Symptoms in the window (+ trailing lag so late symptoms after a late exposure
    # day are still matched).
    symptom_cutoff = cutoff
    symptom_times = list(
        (
            await db.execute(
                select(SymptomScore.timestamp).where(
                    and_(
                        SymptomScore.user_id == user_id,
                        SymptomScore.timestamp >= symptom_cutoff,
                        SymptomScore.deleted_at.is_(None),
                    )
                )
            )
        ).scalars().all()
    )
    meal_times = [meal.timestamp for meal in meals]
    outcome_days = _symptom_outcome_days(meal_times, symptom_times, max_lag)

    # Candidate components: observed in diary ∪ implicated by condition.
    candidates: set[ComponentType] = set(exposed_days) | implicated
    unexposed_prior = _unexposed_prior()

    results: list[ComponentTriggerResult] = []
    for component_type in candidates:
        pop_rate = population_prior.get(component_type, DEFAULT_POPULATION_RATE)
        exposed_prior = _build_exposed_prior(
            pop_rate, component_type in implicated
        )
        results.append(
            _score_component(
                component_type,
                exposed_days.get(component_type, set()),
                meal_days,
                outcome_days,
                exposed_prior,
                unexposed_prior,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def prior_score(
    population_rate: float, implicated: bool
) -> float:
    """Cold-start score (0–100) for a component from priors alone.

    Convenience for callers that want the prior-only score without a DB round-trip
    (e.g. Sprint 2/3 seeding). Equivalent to the ``score`` an
    ``analyze_bayesian_triggers`` result would carry with no observed days.
    """
    alpha0, beta0 = _build_exposed_prior(population_rate, implicated)
    alpha_u, beta_u = _unexposed_prior()
    return prob_beta_exceeds(alpha0, beta0, alpha_u, beta_u) * 100.0


def prior_mean(population_rate: float, implicated: bool) -> float:
    """Prior mean symptom-rate-given-exposure (0–1) for a component. Diagnostic."""
    alpha0, beta0 = _build_exposed_prior(population_rate, implicated)
    return beta_mean(alpha0, beta0)


def cold_start_component_result(
    component_type: ComponentType,
    population_rate: float,
    implicated: bool,
) -> ComponentTriggerResult:
    """Full cold-start (zero-data) ``ComponentTriggerResult`` from priors alone.

    The posterior equals the prior (no observed days), so this is the exact result
    ``analyze_bayesian_triggers`` would return for the component at zero data. Used by
    ``trigger_service.seed_condition_priors`` to persist onboarding priors as real
    Bayesian rows — the continuous prior→data blend, no hard synthetic-decay switch.
    """
    exposed_prior = _build_exposed_prior(population_rate, implicated)
    unexposed_prior = _unexposed_prior()
    return _score_component(
        component_type,
        exposed_days=set(),
        meal_days=set(),
        outcome_days=set(),
        exposed_prior=exposed_prior,
        unexposed_prior=unexposed_prior,
    )


async def food_components_by_name(
    db: AsyncSession, names: set[str]
) -> dict[str, set[ComponentType]]:
    """Map each food NAME to the KB components it carries at exposure level.

    Attribution helper for the suspect-foods leaderboard: it groups per-component
    Bayesian results back onto the foods a patient actually logged (which are keyed
    by free-text name). Only components at level >= EXPOSURE_LEVEL_THRESHOLD are
    included, matching the exposure definition the engine scores against — so a
    food's "driving component" is one that genuinely constituted an exposure, not a
    trace amount. Names that match no KB food map to an empty set.
    """
    name_to_food_id = await _resolve_names_to_food_ids(db, names)
    if not name_to_food_id:
        return {}
    levels = await _load_component_levels(db, set(name_to_food_id.values()))
    out: dict[str, set[ComponentType]] = {}
    for name_key, food_id in name_to_food_id.items():
        comps = {
            comp
            for comp, level in levels.get(food_id, {}).items()
            if level >= EXPOSURE_LEVEL_THRESHOLD
        }
        if comps:
            out[name_key] = comps
    return out
