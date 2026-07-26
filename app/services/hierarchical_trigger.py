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
from app.utils.confidence import normal_cdf

# ── Tunable model constants ───────────────────────────────────────────────────

#: Default lookback window (days) for an analysis run.
DEFAULT_LOOKBACK_DAYS = 90

#: Fallback max symptom-onset lag (hours) when the user records no condition.
DEFAULT_MAX_LAG_HOURS = 48.0

#: Fallback *typical* onset lag (hours) when the user records no condition. Used to
#: align exposure to the day symptoms actually land on (see ``_onset_lag_hours`` and
#: the onset-shift model note below). Deliberately much shorter than the max-lag tail.
DEFAULT_ONSET_LAG_HOURS = 12.0

#: Extra whole-day tolerance on either side of the onset-aligned exposure day, as the
#: span of a backward lag kernel. 0 = align exactly to the onset day (the tightest
#: window, which preserves the most unexposed control days). A future refinement could
#: widen this to absorb onset variance on real diaries; the bake-off harness (fixed
#: onset) is maximised at 0, so we ship the tight window and keep this as the knob.
ONSET_TOLERANCE_DAYS = 0

#: Lag-kernel exponential-decay half-life in DAYS. Recent exposure is weighted more
#: than exposure several days back; span is set by ``ONSET_TOLERANCE_DAYS``.
LAG_KERNEL_HALFLIFE_DAYS = 1.0

#: Background/confounding floor: the minimum odds ratio at which a component is called
#: a trigger. ``trigger_probability = P(OR_c > BACKGROUND_EFFECT_OR)`` instead of the
#: naive ``P(OR_c > 1)``. A ubiquitous *background* component (present at similar
#: levels on symptom and non-symptom days) has β ≈ 0, which ``P(OR>1)`` scores at ~50
#: — enough to flag every innocent food that merely carries it. Requiring a *meaningful*
#: 1.5× odds increase collapses those background/near-null components toward 0 while
#: leaving genuine triggers (OR ≫ 1.5) essentially unchanged. This is the confounding
#: adjustment that stops trace component loads from safe foods inflating innocent foods.
BACKGROUND_EFFECT_OR = 1.5
_MEANINGFUL_LOG_OR = float(np.log(BACKGROUND_EFFECT_OR))

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

#: Max backtracking halvings per Newton step (line search globalization). Under
#: quasi-separation W = p(1−p) → 0, the raw Newton step can overshoot massively and
#: the iteration diverges (β → tens/hundreds, exp(β) overflows the odds-ratio
#: interval to ~1e37). A backtracking line search on the penalized log-likelihood
#: only accepts a step that does not *decrease* the objective, so the fit converges
#: to the true finite MAP instead of running away.
_MAX_BACKTRACK = 40

#: Hard clamp on |β_c| (log-odds) as a final backstop. exp(±BETA_CLAMP) bounds every
#: odds-ratio interval well below the finiteness ceiling (1e12) even at the max SE,
#: and is far outside any legitimate ridge-penalised effect, so it never binds on
#: real data — it only fires on a pathological run the line search somehow misses.
_BETA_CLAMP = 15.0

#: Numerical floors.
_PROB_EPS = 1e-6          # clip p away from {0,1} so W stays positive-definite
_HESSIAN_JITTER = 1e-8    # tiny ridge added to the Hessian diagonal for invertibility

#: FoodComponentDetail.level (0–4) at/above which a food counts as "carrying" a
#: component for leaderboard ATTRIBUTION (food -> driving component). Matches the
#: flat engine's exposure threshold so the suspect-foods join is consistent with a
#: genuine exposure rather than a trace amount.
ATTRIBUTION_LEVEL_THRESHOLD = 2.0


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentTriggerResult:
    """Hierarchical-Bayes association result for one (user, ComponentType).

    Fields:
        component_type: the ComponentType analysed.
        trigger_probability: P(OR_c > BACKGROUND_EFFECT_OR) = Φ((β̂_c − log 1.5) / SE_c),
            in 0–1. Probability the component raises symptom odds by a *meaningful*
            (≥1.5×) amount — the background-adjusted trigger probability (a near-null
            background component sits near 0 here, not at the 0.5 that P(β>0) gives it).
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

def _penalized_loglik(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Objective maximised by ``fit_penalized_logistic``: the merit function that the
    Newton line search backtracks on.

        Σ_t [y_t·η_t − softplus(η_t)] − ½ Σ_c λ_c (β_c − μ_c)²,   η = Xβ

    ``softplus(η) = log(1 + e^η)`` is evaluated in the numerically stable form
    ``max(η, 0) + log1p(e^{−|η|})`` so large |η| neither overflows nor loses the
    tiny-probability tail. Returns ``-inf`` if the design produces a non-finite η,
    which the caller reads as "reject this step".
    """
    eta = X @ beta
    if not np.all(np.isfinite(eta)):
        return -np.inf
    softplus = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    log_lik = float(np.sum(y * eta - softplus))
    penalty = 0.5 * float(np.sum(lam * (beta - mu) ** 2))
    return log_lik - penalty


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

        β ← β + t · (XᵀWX + Λ)⁻¹ (Xᵀ(y−p) − Λ(β−μ)),   W = diag(p_t(1−p_t)).

    The Newton direction is globalized by a **backtracking line search**: the step
    scale ``t`` starts at 1 and is halved (up to ``_MAX_BACKTRACK`` times) until the
    candidate does not decrease the penalized log-likelihood (``_penalized_loglik``).
    Plain Newton has no such safeguard, so under quasi-separation (W → 0, common for a
    condition-seeded component whose exposure almost perfectly predicts symptoms) it
    overshoots and diverges — β blows up to tens/hundreds and the reported odds-ratio
    interval ``exp(β̂ ± 1.96·SE)`` overflows to ~1e37. With the line search the fit
    settles at the true finite MAP. Each accepted β is additionally clamped to
    ``±_BETA_CLAMP`` as a last-resort backstop so the interval is always finite.

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

        # Backtracking line search: accept the largest t ∈ {1, ½, ¼, …} whose step
        # does not decrease the penalized log-likelihood. Prevents the Newton
        # overshoot / divergence that quasi-separation (W → 0) otherwise causes.
        f0 = _penalized_loglik(X, y, beta, mu, lam)
        t = 1.0
        accepted = False
        for _bt in range(_MAX_BACKTRACK):
            candidate = np.clip(beta + t * step, -_BETA_CLAMP, _BETA_CLAMP)
            if _penalized_loglik(X, y, candidate, mu, lam) >= f0:
                beta = candidate
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # No downhill-safe step (already at the MAP to numerical precision).
            break
        if np.max(np.abs(t * step)) < tol:
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


def _onset_lag_hours(condition_keys: list[str]) -> float:
    """*Typical* symptom-onset lag (hours) across the user's conditions.

    This is the centre of onset, NOT the tail max ``_max_lag_hours`` returns. It drives
    the onset-shift exposure model: each meal is attributed to the day its symptoms are
    expected to appear (``meal_time + onset``), so exposure lines up with the outcome
    it causes instead of being smeared across the whole [0, max-lag] tail. Using the
    max-lag window (36h for IBS) instead destroyed contrast — a frequent trigger fell
    inside almost every day's window, leaving no unexposed control days and flipping the
    fitted effect *protective*. The typical onset keeps control days intact.

    Per condition: the triangular configs' ``peak``; the bimodal MCAS config's
    weight-averaged mode midpoints (the dominant delayed mode drives it). Take the
    widest typical onset across conditions; ``DEFAULT_ONSET_LAG_HOURS`` when none.
    """
    onsets: list[float] = []
    for key in condition_keys:
        cfg = CONDITION_CONFIGS.get(key)
        if not cfg:
            continue
        lag = cfg.get("lag_hours", {})
        if lag.get("bimodal"):
            weighted = 0.0
            weight_sum = 0.0
            for mode_key in ("mode_1", "mode_2"):
                mode = lag.get(mode_key)
                if not mode:
                    continue
                midpoint = 0.5 * (float(mode["min"]) + float(mode["max"]))
                weight = float(mode.get("weight", 0.5))
                weighted += weight * midpoint
                weight_sum += weight
            if weight_sum > 0:
                onsets.append(weighted / weight_sum)
        elif "peak" in lag:
            onsets.append(float(lag["peak"]))
        elif "max" in lag:
            onsets.append(float(lag["max"]))
    return max(onsets) if onsets else DEFAULT_ONSET_LAG_HOURS


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


async def _resolve_names_to_food_ids(
    db: AsyncSession, names: set[str]
) -> dict[str, uuid.UUID]:
    """Map lowercased food names -> a KB ``FoodEntry.id`` (case-insensitive).

    Meal items logged through the API/AI pipeline carry a free-text ``name`` but a
    NULL ``food_entry_id``; without this resolution those items would register no
    exposure and every component would collapse to its prior on real diaries. When
    several KB rows share a name the smallest id is chosen for determinism. Returns
    an empty dict for empty input.
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
        out.setdefault(name.strip().lower(), food_id)  # first (smallest id) wins
    return out


def _daily_component_loads(
    meals: list[Meal],
    component_levels: dict[uuid.UUID, dict[ComponentType, float]],
    name_to_food_id: dict[str, uuid.UUID] | None = None,
    onset_shift_hours: float = 0.0,
) -> tuple[dict[date, dict[ComponentType, float]], set[ComponentType]]:
    """Aggregate KB component levels into a per-day load per component.

    Returns ``(daily_loads, observed_components)`` where ``daily_loads[day][comp]``
    is the summed ``FoodComponentDetail.level`` (0–4) across every meal item that day
    carrying that component, and ``observed_components`` is every component seen.

    **Onset shift**: each meal is binned to ``(meal_time + onset_shift_hours).date()``,
    i.e. the day its symptoms are *expected* to appear, not the day it was eaten. With a
    typical onset that crosses midnight (e.g. an evening trigger with an 8h IBS lag →
    next-morning symptom) this aligns the exposure row with the symptom row it explains;
    binning to the meal day instead put the trigger's load on a non-symptom day and made
    the fitted effect read protective. ``onset_shift_hours = 0`` reproduces meal-day
    binning.

    A meal item is resolved to a KB food by ``food_entry_id`` when present, else by a
    case-insensitive name match (``name_to_food_id``) so API/AI-logged items (NULL
    ``food_entry_id``) still register exposure.
    """
    name_to_food_id = name_to_food_id or {}
    shift = timedelta(hours=onset_shift_hours)
    daily_loads: dict[date, dict[ComponentType, float]] = {}
    observed: set[ComponentType] = set()
    for meal in meals:
        day = (meal.timestamp + shift).date()
        bucket = daily_loads.setdefault(day, {})
        for item in meal.items:
            food_id = item.food_entry_id
            if food_id is None:
                food_id = name_to_food_id.get((item.name or "").strip().lower())
            if food_id is None:
                continue
            for comp, level in component_levels.get(food_id, {}).items():
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
    # Onset-shift exposure model: attribute each meal to the day symptoms are expected
    # (meal_time + typical onset), with only a small whole-day tolerance kernel. This
    # replaces the old wide max-lag window that smeared exposure across [0, 36h] and
    # destroyed the exposed/unexposed contrast for frequent triggers.
    onset_hours = _onset_lag_hours(condition_keys)
    kernel = lag_kernel(ONSET_TOLERANCE_DAYS)

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
    daily_loads, observed = _daily_component_loads(
        meals, component_levels, name_to_food_id, onset_shift_hours=onset_hours
    )

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
        # Background floor: P(β_c > log(BACKGROUND_EFFECT_OR)), not P(β_c > 0). Requires
        # a meaningful (≥1.5×) odds increase so near-null background components don't
        # score ~50 and flag every innocent food that carries them.
        trigger_probability = normal_cdf(b, mu=_MEANINGFUL_LOG_OR, sigma=se) if se > 0.0 else (
            1.0 if b > _MEANINGFUL_LOG_OR else 0.0
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


# ── Attribution + cold-start helpers (endpoint / seeding plumbing) ─────────────

async def food_components_by_name(
    db: AsyncSession, names: set[str]
) -> dict[str, set[ComponentType]]:
    """Map each food NAME to the KB components it carries at the attribution level.

    Attribution helper for the suspect-foods leaderboard: it groups the per-component
    hierarchical results back onto the foods a patient actually logged (which are keyed
    by free-text name). Only components at level >= ``ATTRIBUTION_LEVEL_THRESHOLD`` are
    included, so a food's "driving component" is one that genuinely constituted an
    exposure, not a trace amount. Names matching no KB food map to an empty set.
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
            if level >= ATTRIBUTION_LEVEL_THRESHOLD
        }
        if comps:
            out[name_key] = comps
    return out


async def cold_start_results(
    db: AsyncSession,
    condition_types: list[str],
    population_prior: dict[ComponentType, tuple[float, float]] | None = None,
) -> list[ComponentTriggerResult]:
    """Prior-only (zero-data) results for a set of declared conditions.

    Cold start = the MAP collapses to the prior mean and the Laplace covariance to the
    prior covariance (``fit_penalized_logistic`` with an empty design). For each
    ComponentType implicated by ``condition_types`` (``CONDITION_PRIORS``) this returns
    the exact result ``analyze_hierarchical_triggers`` would produce at zero diary data,
    so condition-implicated components come back elevated (μ_c > 0) rather than at zero.
    Used by ``trigger_service.seed_condition_priors`` to persist onboarding priors
    through the same path the data-driven analysis uses. Empty when no condition is
    recognised.
    """
    condition_keys = [c.lower().strip() for c in condition_types]
    implicated = _implicated_components(condition_keys)
    if not implicated:
        return []

    if population_prior is None:
        population_prior = await build_population_prior(db)

    candidates = sorted(implicated, key=lambda c: c.value)
    mu = np.empty(len(candidates) + 1)
    lam = np.empty(len(candidates) + 1)
    mu[0] = float(np.log(BACKGROUND_SYMPTOM_RATE / (1.0 - BACKGROUND_SYMPTOM_RATE)))
    lam[0] = INTERCEPT_PRECISION
    for i, comp in enumerate(candidates):
        pop_mean, pop_prec = population_prior.get(comp, (0.0, DEFAULT_COMPONENT_PRECISION))
        mu[i + 1] = float(np.clip(pop_mean + CLINICAL_SEED_MEAN, -MAX_PRIOR_MEAN, MAX_PRIOR_MEAN))
        lam[i + 1] = pop_prec

    beta, cov = fit_penalized_logistic(
        np.empty((0, len(candidates) + 1)), np.empty((0,)), mu, lam
    )

    results: list[ComponentTriggerResult] = []
    for i, comp in enumerate(candidates):
        b = float(beta[i + 1])
        var = float(cov[i + 1, i + 1])
        se = float(np.sqrt(var)) if var > 0.0 else 0.0
        # Background floor (see analyze_hierarchical_triggers): P(β > log(1.5)).
        trigger_probability = normal_cdf(b, mu=_MEANINGFUL_LOG_OR, sigma=se) if se > 0.0 else (
            1.0 if b > _MEANINGFUL_LOG_OR else 0.0
        )
        results.append(
            ComponentTriggerResult(
                component_type=comp,
                trigger_probability=trigger_probability,
                score=trigger_probability * 100.0,
                ci_low=float(np.exp(b - 1.96 * se)),
                ci_high=float(np.exp(b + 1.96 * se)),
                beta=b,
                beta_se=se,
                n_obs=0,
                n_exposed=0,
                is_cold_start=True,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results
