"""Hierarchical Bayesian logistic-regression trigger engine (Wave 2, Sprint H1).

Where ``bayesian_trigger.py`` scores each ComponentType *independently* with a
Beta-Binomial 2x2 model, this module fits **one joint logistic regression per
user** over all candidate components at once. The joint fit is the whole point:
when two component exposures are collinear (e.g. a user's onion always comes with
garlic, so FODMAP and FRUCTOSE move together), the univariate model credits BOTH.
The multivariate logistic fit instead *partitions* the effect — it credits the
component that actually raises the symptom odds and shrinks the passenger toward
its prior. That de-confounding is the win the flat model structurally cannot get.

Model
-----
Unit of observation = a calendar **day** ``t`` within the lookback window.

    P(y_t = 1) = sigmoid(β0 + Σ_c β_c · x_{t,c})

* ``y_t`` = 1 iff a (non-deleted) symptom was recorded on day ``t``.
* ``x_{t,c}`` = distributed-lag exposure to component ``c`` at day ``t``: the
  daily component loads over a short window *preceding and including* ``t``,
  weighted by a normalized lag kernel (exponential decay by default) whose span
  is the condition-appropriate max lag from ``CONDITION_CONFIGS``. Because the
  kernel is normalized to sum to 1, ``β_c`` reads as the log-odds change per unit
  of (recency-weighted) daily component load.

Hierarchy / partial pooling
---------------------------
* **Population level** — a per-component prior ``β_c ~ Normal(μ_c, 1/λ_c)``:
    (a) *clinical seed*: components implicated by the user's condition
        (``CONDITION_PRIORS``) get a positive prior mean ``μ_c`` (+CLINICAL_SEED_MEAN
        log-odds); everything else starts at the cohort mean (≈0).
    (b) *cohort learning*: ``build_population_prior`` aggregates the synthetic
        cohort (``is_synthetic`` users) into a per-component baseline log-odds mean,
        so a component that is broadly a real trigger across patients starts nudged
        up even before the clinical seed.
* **User level** — β is the MAP of a ridge-to-population penalized logistic
  regression: maximize ``log-lik − ½ Σ_c λ_c (β_c − μ_c)²``. A small-n user is
  shrunk toward ``μ``; a data-rich user overrides the prior and fits their own
  data. Solved by IRLS / Newton–Raphson (see ``fit_penalized_logistic``).

Laplace posterior
-----------------
At the MAP ``β̂`` the posterior is approximated ``Normal(β̂, Σ)`` with
``Σ = (XᵀWX + Λ)⁻¹``, ``W = diag(p_t(1−p_t))``, ``Λ = diag(λ)``. Per component:

* ``trigger_probability = P(β_c > 0) = Φ(β̂_c / SE_c)`` (``SE_c = sqrt(Σ_cc)``).
* ``ci_low``/``ci_high`` — 95% credible interval on the **odds ratio**
  ``exp(β̂_c ± 1.96·SE_c)`` (an interpretable, strictly-positive effect scale;
  OR>1 ⇒ exposure raises symptom odds).
* ``score = trigger_probability · 100`` so the existing D9 tier mapping applies.

Cold start
----------
A user with no diary data yields an empty design; the MAP collapses to the prior
mean ``μ`` and ``Σ = Λ⁻¹``. Condition-implicated components therefore come back
elevated (μ_c > 0) rather than at zero.

The module is pure / read-only: it issues SELECTs and returns frozen dataclasses.
No writes, no side effects, fully deterministic (same DB state → identical output,
no sampling / Monte-Carlo).

Exposure-scale note
-------------------
Exposure is derived from ``FoodComponentDetail.level`` (the KB's **0–4** scale),
NOT ``MealItemComponent.estimated_level`` (a different, 0–100 per-meal scale) —
resolving the 0–4 vs 0–100 ambiguity in favour of the KB-sourced ``level``.
Synthetic meals carry no ``MealItemComponent`` rows, so the KB join is also the
only exposure path that works for the synthetic cohort.
"""

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enums import ComponentType, ConditionType
from app.models.food import FoodComponentDetail
from app.models.meal import Meal
from app.models.sensitivity import UserSensitivityProfile
from app.models.symptom import SymptomScore
from app.models.user import User, UserCondition
from app.services.synthetic_data_generator import CONDITION_CONFIGS
from app.services.trigger_service import CONDITION_PRIORS
from app.utils.confidence import normal_cdf

# ── Tunable model constants ───────────────────────────────────────────────────

#: Default lookback window (days) for an analysis run.
DEFAULT_LOOKBACK_DAYS = 90

#: Fallback max symptom-onset lag (hours) when the user records no condition.
DEFAULT_MAX_LAG_HOURS = 48.0

#: Lag-kernel exponential-decay half-life in DAYS. Recent exposure is weighted more
#: than exposure several days back; span is set by the condition max lag.
LAG_KERNEL_HALFLIFE_DAYS = 1.0

#: Ridge precision λ_c applied to every component coefficient's prior. Prior SD of
#: β_c is 1/sqrt(λ) ≈ 1.0 log-odds — moderate shrinkage a data-rich user overrides.
DEFAULT_COMPONENT_PRECISION = 1.0

#: Precision on the intercept prior. Weak (near-unpenalized) so the baseline
#: symptom rate is driven by the data, not pinned to the prior.
INTERCEPT_PRECISION = 0.05

#: Prior mean for the intercept: log-odds of the background symptom-day rate.
BACKGROUND_SYMPTOM_RATE = 0.20

#: Positive prior mean (log-odds / odds-ratio ≈ e^0.7 ≈ 2) applied to a component
#: implicated by the user's condition via CONDITION_PRIORS (the clinical seed).
CLINICAL_SEED_MEAN = 0.7

#: Scale converting a cohort component "real-trigger" rate into a prior log-odds
#: mean, centred on DEFAULT_POPULATION_RATE so an average-prevalence component ≈ 0.
COHORT_LOGODDS_SCALE = 1.2
DEFAULT_POPULATION_RATE = 0.10

#: Clamp on any single component prior mean (log-odds), keeps priors sane.
MAX_PRIOR_MEAN = 1.5

#: IRLS / Newton controls.
MAX_NEWTON_ITERS = 100
NEWTON_TOL = 1e-8

#: Numerical floors.
_PROB_EPS = 1e-6          # clip p away from {0,1} so W stays positive-definite
_HESSIAN_JITTER = 1e-8    # tiny ridge added to the Hessian diagonal for invertibility


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentTriggerResult:
    """Hierarchical-Bayes association result for one (user, ComponentType).

    Fields:
        component_type: the ComponentType analysed.
        trigger_probability: P(β_c > 0) = Φ(β̂_c / SE_c), in 0–1. Probability the
            component's log-odds effect on symptoms is positive (a real trigger).
        score: trigger_probability * 100 (0–100; feeds the D9 tier mapping).
        ci_low / ci_high: 95% credible interval on the ODDS RATIO exp(β̂_c ± 1.96·SE_c)
            (strictly positive; 1.0 = no effect, >1 = raises symptom odds).
        beta: MAP coefficient β̂_c (log-odds per unit recency-weighted daily load).
        beta_se: Laplace posterior SD sqrt(Σ_cc).
        n_obs: number of day-rows in the joint regression (0 on cold start).
        n_exposed: number of those days with non-zero exposure to this component.
        is_cold_start: True when there were no observation days (result = prior).
        method: engine tag, always "hierarchical_bayes_logistic".
    """

    component_type: ComponentType
    trigger_probability: float
    score: float
    ci_low: float
    ci_high: float
    beta: float
    beta_se: float
    n_obs: int
    n_exposed: int
    is_cold_start: bool
    method: str = "hierarchical_bayes_logistic"

    def to_dict(self) -> dict:
        """Plain-dict view (enum rendered as its string value)."""
        d = asdict(self)
        d["component_type"] = self.component_type.value
        return d


# ── Core numeric solver (pure numpy; independently verifiable) ─────────────────

def fit_penalized_logistic(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    max_iter: int = MAX_NEWTON_ITERS,
    tol: float = NEWTON_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP fit of ridge-to-prior logistic regression + Laplace covariance.

    Maximizes  ``Σ_t [y_t·η_t − log(1+e^{η_t})] − ½ (β−μ)ᵀ Λ (β−μ)``  where
    ``η = Xβ``, ``μ = prior_mean``, ``Λ = diag(prior_precision)``, via Newton–Raphson:

        β ← β + (XᵀWX + Λ)⁻¹ (Xᵀ(y−p) − Λ(β−μ)),   W = diag(p_t(1−p_t)).

    Args:
        X: (n, k) design matrix (caller includes any intercept column).
        y: (n,) binary outcomes in {0, 1}.
        prior_mean: (k,) prior mean μ for each coefficient.
        prior_precision: (k,) prior precision λ (≥ 0) for each coefficient.
        max_iter, tol: Newton controls (converge on max |Δβ| < tol).

    Returns:
        (beta, cov) where ``beta`` is the (k,) MAP estimate and ``cov`` is the (k, k)
        Laplace posterior covariance ``(XᵀWX + Λ)⁻¹`` evaluated at the MAP.

    With ``n == 0`` (no data) the penalty alone governs: β = μ and cov = Λ⁻¹ — the
    cold-start path (prior mean, prior variance). ``prior_precision`` of 0 for a
    coefficient with no data leaves it unidentified; callers give every coefficient
    a strictly-positive component precision (or a small intercept precision) so the
    Hessian is always invertible.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(prior_mean, dtype=float)
    lam = np.asarray(prior_precision, dtype=float)
    k = mu.shape[0]
    Lambda = np.diag(lam)

    # Cold start / no rows: MAP is the prior mean, covariance is the prior covariance.
    if X.size == 0 or X.shape[0] == 0:
        cov = np.linalg.pinv(Lambda + _HESSIAN_JITTER * np.eye(k))
        return mu.copy(), cov

    beta = mu.copy()
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
        W = p * (1.0 - p)

        gradient = X.T @ (y - p) - lam * (beta - mu)
        hessian = X.T @ (X * W[:, None]) + Lambda + _HESSIAN_JITTER * np.eye(k)

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    # Laplace covariance at the MAP.
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), _PROB_EPS, 1.0 - _PROB_EPS)
    W = p * (1.0 - p)
    hessian = X.T @ (X * W[:, None]) + Lambda + _HESSIAN_JITTER * np.eye(k)
    cov = np.linalg.pinv(hessian)
    return beta, cov


def lag_kernel(max_lag_days: int, halflife_days: float = LAG_KERNEL_HALFLIFE_DAYS) -> np.ndarray:
    """Normalized exponential-decay lag kernel over ``0..max_lag_days`` (inclusive).

    ``w_k ∝ 0.5 ** (k / halflife_days)`` for lag ``k`` days, normalized to sum 1 so a
    coefficient stays interpretable as "log-odds per unit recency-weighted daily
    load". ``halflife_days <= 0`` degenerates to an all-weight-on-today kernel.
    """
    n = max(int(max_lag_days), 0) + 1
    if halflife_days <= 0.0:
        w = np.zeros(n)
        w[0] = 1.0
        return w
    ks = np.arange(n, dtype=float)
    w = 0.5 ** (ks / halflife_days)
    return w / w.sum()


# ── Population prior (cohort learning) ─────────────────────────────────────────

async def _cohort_component_rates(db: AsyncSession) -> dict[ComponentType, float]:
    """Per-component 'real trigger' rate across the synthetic cohort.

    Rate = (# synthetic users with a UserSensitivityProfile row for the component)
    / (total # synthetic users). ``UserSensitivityProfile`` is the synthetic ground
    truth: a row means the component is genuinely a trigger for that patient.
    Empty dict when there is no synthetic cohort.
    """
    synthetic_ids = (
        await db.execute(select(User.id).where(User.is_synthetic.is_(True)))
    ).scalars().all()
    n_users = len(synthetic_ids)
    if n_users == 0:
        return {}

    rows = (
        await db.execute(
            select(
                UserSensitivityProfile.component_type,
                UserSensitivityProfile.user_id,
            ).where(UserSensitivityProfile.user_id.in_(set(synthetic_ids)))
        )
    ).all()

    users_by_component: dict[ComponentType, set[uuid.UUID]] = {}
    for component_type, user_id in rows:
        users_by_component.setdefault(component_type, set()).add(user_id)
    return {comp: len(users) / n_users for comp, users in users_by_component.items()}


def _rate_to_prior_mean(rate: float) -> float:
    """Map a cohort trigger rate (0–1) to a prior log-odds mean, clamped."""
    mean = COHORT_LOGODDS_SCALE * (rate - DEFAULT_POPULATION_RATE)
    return float(np.clip(mean, -MAX_PRIOR_MEAN, MAX_PRIOR_MEAN))


async def build_population_prior(
    db: AsyncSession,
) -> dict[ComponentType, tuple[float, float]]:
    """Population-level (condition-agnostic) prior from cohort learning.

    Returns ``{ComponentType: (μ_c, λ_c)}`` — a per-component prior mean (log-odds)
    and precision, derived from the synthetic cohort's component trigger rates
    (``_cohort_component_rates``). The clinical seed (condition-specific positive
    nudge) is layered on TOP of this per-user inside ``analyze_hierarchical_triggers``.

    Empty dict when there is no synthetic cohort; callers then fall back to a zero
    mean / ``DEFAULT_COMPONENT_PRECISION`` for every component. Reusable: build once
    and pass into repeated ``analyze_hierarchical_triggers`` calls.
    """
    rates = await _cohort_component_rates(db)
    return {
        comp: (_rate_to_prior_mean(rate), DEFAULT_COMPONENT_PRECISION)
        for comp, rate in rates.items()
    }


# ── Condition helpers (shared vocabulary with bayesian_trigger.py) ─────────────

def _condition_keys_for(condition_types: list[ConditionType]) -> list[str]:
    """ConditionType enum values → the lowercase keys of CONDITION_PRIORS/CONFIGS."""
    return [ct.value for ct in condition_types]


def _implicated_components(condition_keys: list[str]) -> set[ComponentType]:
    """Union of ComponentTypes implicated by the user's conditions (CONDITION_PRIORS)."""
    implicated: set[ComponentType] = set()
    for key in condition_keys:
        implicated.update(CONDITION_PRIORS.get(key, []))
    return implicated


def _max_lag_hours(condition_keys: list[str]) -> float:
    """Widest symptom-onset lag (hours) across the user's conditions.

    Triangular configs use ``max``; the bimodal MCAS config uses its later mode's
    ``max`` (collapsed to the outer bound — a future refinement would score the
    immediate and delayed modes separately). No condition → DEFAULT_MAX_LAG_HOURS.
    """
    lags: list[float] = []
    for key in condition_keys:
        cfg = CONDITION_CONFIGS.get(key)
        if not cfg:
            continue
        lag = cfg.get("lag_hours", {})
        if lag.get("bimodal"):
            lags.append(float(lag["mode_2"]["max"]))
        elif "max" in lag:
            lags.append(float(lag["max"]))
    return max(lags) if lags else DEFAULT_MAX_LAG_HOURS


# ── Exposure / design-matrix construction ──────────────────────────────────────

async def _load_component_levels(
    db: AsyncSession, food_ids: set[uuid.UUID]
) -> dict[uuid.UUID, dict[ComponentType, float]]:
    """Map each food_entry_id → {ComponentType: level (0–4)} from the KB."""
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


def _daily_component_loads(
    meals: list[Meal],
    component_levels: dict[uuid.UUID, dict[ComponentType, float]],
) -> tuple[dict[date, dict[ComponentType, float]], set[ComponentType]]:
    """Aggregate KB component levels into a per-day load per component.

    Returns ``(daily_loads, observed_components)`` where ``daily_loads[day][comp]``
    is the summed ``FoodComponentDetail.level`` (0–4) across every meal item that day
    carrying that component, and ``observed_components`` is every component seen.
    """
    daily_loads: dict[date, dict[ComponentType, float]] = {}
    observed: set[ComponentType] = set()
    for meal in meals:
        day = meal.timestamp.date()
        bucket = daily_loads.setdefault(day, {})
        for item in meal.items:
            if item.food_entry_id is None:
                continue
            for comp, level in component_levels.get(item.food_entry_id, {}).items():
                bucket[comp] = bucket.get(comp, 0.0) + level
                observed.add(comp)
    return daily_loads, observed


def _build_design_matrix(
    daily_loads: dict[date, dict[ComponentType, float]],
    symptom_days: set[date],
    components: list[ComponentType],
    kernel: np.ndarray,
    day_start: date,
    day_end: date,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (n_days, 1+n_components) design matrix X and outcome vector y.

    One row per calendar day in ``[day_start, day_end]``. Column 0 is the intercept
    (ones); column ``1+i`` is the distributed-lag exposure to ``components[i]``:
    ``x_{t,c} = Σ_k kernel[k] · load_c(t-k)`` over the kernel's lag span. ``y_t = 1``
    iff ``t`` is a symptom day. Returns ``(X, y)`` (empty arrays if the range is empty).
    """
    if day_end < day_start:
        return np.empty((0, len(components) + 1)), np.empty((0,))

    n_days = (day_end - day_start).days + 1
    days = [day_start + timedelta(days=i) for i in range(n_days)]

    # Dense load series per component (days x components), then apply the causal
    # lag kernel: x_t = Σ_k kernel[k] * load_{t-k}.
    load = np.zeros((n_days, len(components)))
    comp_index = {c: i for i, c in enumerate(components)}
    for i, day in enumerate(days):
        for comp, val in daily_loads.get(day, {}).items():
            j = comp_index.get(comp)
            if j is not None:
                load[i, j] = val

    feat = np.zeros_like(load)
    for k, w in enumerate(kernel):
        if w == 0.0:
            continue
        # rows k..n-1 receive weight w * load shifted back by k days
        feat[k:, :] += w * load[: n_days - k, :]

    X = np.column_stack([np.ones(n_days), feat])
    y = np.array([1.0 if day in symptom_days else 0.0 for day in days])
    return X, y


# ── Public engine entry point ─────────────────────────────────────────────────

async def analyze_hierarchical_triggers(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    population_prior: dict[ComponentType, tuple[float, float]] | None = None,
) -> list[ComponentTriggerResult]:
    """Fit the joint hierarchical logistic model and return per-component results.

    Args:
        db: async session (read-only usage).
        user_id: user to analyse.
        lookback_days: window length; meals/symptoms older than this are ignored.
        population_prior: optional precomputed ``build_population_prior`` table
            (``{comp: (μ_c, λ_c)}``). Built on demand when omitted.

    Returns:
        List of ``ComponentTriggerResult`` — one per candidate component (components
        observed in the user's diary UNION components implicated by their conditions),
        sorted by ``score`` descending. Empty only when there is neither data nor an
        implicating condition. On cold start (no diary days) each result carries the
        prior, so condition-implicated components come back elevated rather than zero.
    """
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

    # Conditions → clinical seed + lag-kernel span.
    condition_types = (
        await db.execute(
            select(UserCondition.condition_type).where(UserCondition.user_id == user_id)
        )
    ).scalars().all()
    condition_keys = _condition_keys_for(list(condition_types))
    implicated = _implicated_components(condition_keys)
    max_lag_days = int(np.ceil(_max_lag_hours(condition_keys) / 24.0))
    kernel = lag_kernel(max_lag_days)

    if population_prior is None:
        population_prior = await build_population_prior(db)

    # Meals + items in the window (respect soft-deletes).
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
    component_levels = await _load_component_levels(db, food_ids)
    daily_loads, observed = _daily_component_loads(meals, component_levels)

    # Symptom days in the window (respect soft-deletes).
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

    # Candidate components: observed in diary ∪ implicated by condition. Sorted for
    # deterministic column ordering.
    candidates = sorted(observed | implicated, key=lambda c: c.value)
    if not candidates:
        return []

    # Prior mean/precision per coefficient: [intercept, *candidates].
    mu = np.empty(len(candidates) + 1)
    lam = np.empty(len(candidates) + 1)
    mu[0] = float(np.log(BACKGROUND_SYMPTOM_RATE / (1.0 - BACKGROUND_SYMPTOM_RATE)))
    lam[0] = INTERCEPT_PRECISION
    for i, comp in enumerate(candidates):
        pop_mean, pop_prec = population_prior.get(comp, (0.0, DEFAULT_COMPONENT_PRECISION))
        seed = CLINICAL_SEED_MEAN if comp in implicated else 0.0
        mu[i + 1] = float(np.clip(pop_mean + seed, -MAX_PRIOR_MEAN, MAX_PRIOR_MEAN))
        lam[i + 1] = pop_prec

    # Design matrix over the contiguous day range spanned by meals ∪ symptoms.
    all_days = set(daily_loads) | symptom_days
    if all_days:
        day_start, day_end = min(all_days), max(all_days)
        X, y = _build_design_matrix(
            daily_loads, symptom_days, candidates, kernel, day_start, day_end
        )
    else:
        X, y = np.empty((0, len(candidates) + 1)), np.empty((0,))

    beta, cov = fit_penalized_logistic(X, y, mu, lam)
    n_obs = int(X.shape[0])
    is_cold_start = n_obs == 0

    # Per-component exposure counts (days with non-zero raw load for that component).
    exposed_counts = {c: 0 for c in candidates}
    for _day, loads in daily_loads.items():
        for comp in loads:
            if comp in exposed_counts and loads[comp] > 0.0:
                exposed_counts[comp] += 1

    results: list[ComponentTriggerResult] = []
    for i, comp in enumerate(candidates):
        b = float(beta[i + 1])
        var = float(cov[i + 1, i + 1])
        se = float(np.sqrt(var)) if var > 0.0 else 0.0
        trigger_probability = normal_cdf(b, mu=0.0, sigma=se) if se > 0.0 else (
            1.0 if b > 0.0 else 0.0
        )
        ci_low = float(np.exp(b - 1.96 * se))
        ci_high = float(np.exp(b + 1.96 * se))
        results.append(
            ComponentTriggerResult(
                component_type=comp,
                trigger_probability=trigger_probability,
                score=trigger_probability * 100.0,
                ci_low=ci_low,
                ci_high=ci_high,
                beta=b,
                beta_se=se,
                n_obs=n_obs,
                n_exposed=exposed_counts[comp],
                is_cold_start=is_cold_start,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
