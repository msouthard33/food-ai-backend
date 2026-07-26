"""Tests for the hierarchical Bayesian logistic trigger engine (Sprint H1).

Three layers:
  * Pure-math unit tests for the numpy MAP/IRLS + Laplace solver and the normal-CDF
    helper (no DB, no async), including cross-checks against sklearn / statsmodels
    (installed for verification only — skipped if absent).
  * Async engine scenarios that seed a small diary via the ORM and assert the
    recovery / de-confounding / partial-pooling / cold-start / determinism behaviour
    of analyze_hierarchical_triggers.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import text

from app.models.enums import ComponentType, ConditionType, MealType, SymptomType
from app.models.food import FoodComponentDetail, FoodEntry
from app.models.meal import Meal, MealItem
from app.models.symptom import SymptomScore
from app.models.user import UserCondition
from app.services.hierarchical_trigger import (
    DEFAULT_COMPONENT_PRECISION,
    analyze_hierarchical_triggers,
    build_population_prior,
    fit_penalized_logistic,
    lag_kernel,
)
from app.utils.confidence import normal_cdf
from tests.conftest import _ensure_tables, async_session_factory

# ── Pure-math unit tests: normal_cdf ──────────────────────────────────────────

def test_normal_cdf_known_values():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12
    assert abs(normal_cdf(1.96) - 0.975) < 1e-3
    assert abs(normal_cdf(-1.96) - 0.025) < 1e-3
    # Monotone increasing.
    assert normal_cdf(-1.0) < normal_cdf(0.0) < normal_cdf(1.0)


def test_normal_cdf_scaled_and_degenerate():
    # sigma scaling: Φ(2 / 2) = Φ(1).
    assert abs(normal_cdf(2.0, mu=0.0, sigma=2.0) - normal_cdf(1.0)) < 1e-12
    # Degenerate sigma -> step function at mu.
    assert normal_cdf(0.5, mu=0.0, sigma=0.0) == 1.0
    assert normal_cdf(-0.5, mu=0.0, sigma=0.0) == 0.0


# ── Pure-math unit tests: lag kernel ──────────────────────────────────────────

def test_lag_kernel_normalized_and_decaying():
    k = lag_kernel(2, halflife_days=1.0)
    assert len(k) == 3
    assert abs(k.sum() - 1.0) < 1e-12
    # Strictly decaying weights.
    assert k[0] > k[1] > k[2]


def test_lag_kernel_zero_halflife_is_today_only():
    k = lag_kernel(3, halflife_days=0.0)
    assert k[0] == 1.0
    assert k[1:].sum() == 0.0


# ── Pure-math unit tests: MAP solver + cross-checks ───────────────────────────

def _fixed_design(seed: int = 0, n: int = 200, k: int = 3):
    rng = np.random.default_rng(seed)
    Xf = rng.normal(size=(n, k))
    true = np.array([-0.5, 1.2, -0.8, 0.4])
    Xi = np.column_stack([np.ones(n), Xf])
    p = 1.0 / (1.0 + np.exp(-(Xi @ true)))
    y = (rng.uniform(size=n) < p).astype(float)
    return Xf, Xi, y


def test_map_recovers_positive_coefficient_sign():
    """A feature with a truly positive effect recovers a positive β̂."""
    _, Xi, y = _fixed_design()
    mu = np.zeros(4)
    lam = np.array([0.0, 1.0, 1.0, 1.0])
    beta, cov = fit_penalized_logistic(Xi, y, mu, lam)
    # true = [-0.5, +1.2, -0.8, +0.4] -> signs recovered.
    assert beta[1] > 0.5
    assert beta[2] < -0.3
    assert beta[3] > 0.0
    # Covariance is symmetric PD-ish: positive diagonal.
    assert np.all(np.diag(cov) > 0.0)


def test_cold_start_solver_returns_prior():
    """No rows -> MAP == prior mean, cov == prior covariance (diag 1/λ)."""
    mu = np.array([0.1, 0.7, -0.3])
    lam = np.array([0.5, 2.0, 4.0])
    beta, cov = fit_penalized_logistic(np.empty((0, 3)), np.empty((0,)), mu, lam)
    assert np.allclose(beta, mu)
    assert np.allclose(np.diag(cov), 1.0 / lam, rtol=1e-4)


def test_map_matches_sklearn():
    """MAP β̂ matches sklearn LogisticRegression with a matched L2 penalty
    (our λ = 1/C; intercept left unpenalized in both)."""
    sklm = pytest.importorskip("sklearn.linear_model")
    Xf, Xi, y = _fixed_design()
    lam_val = 2.0
    beta, _ = fit_penalized_logistic(
        Xi, y, np.zeros(4), np.array([0.0, lam_val, lam_val, lam_val])
    )
    clf = sklm.LogisticRegression(
        C=1.0 / lam_val, fit_intercept=True, solver="lbfgs", max_iter=5000, tol=1e-10
    )
    clf.fit(Xf, y)
    sk = np.concatenate([clf.intercept_, clf.coef_.ravel()])
    max_dev = float(np.max(np.abs(beta - sk)))
    assert max_dev < 1e-5, f"MAP β deviates from sklearn by {max_dev}"


def test_laplace_se_matches_statsmodels():
    """Laplace SEs (unpenalized fit) match statsmodels Logit .bse."""
    sm = pytest.importorskip("statsmodels.api")
    _, Xi, y = _fixed_design()
    beta, cov = fit_penalized_logistic(Xi, y, np.zeros(4), np.zeros(4))
    our_se = np.sqrt(np.diag(cov))
    res = sm.Logit(y, Xi).fit(disp=0)
    max_se_dev = float(np.max(np.abs(our_se - res.bse)))
    max_beta_dev = float(np.max(np.abs(beta - res.params)))
    assert max_se_dev < 1e-4, f"Laplace SE deviates from statsmodels by {max_se_dev}"
    assert max_beta_dev < 1e-4


# ── Engine test helpers ───────────────────────────────────────────────────────

async def _new_user(condition: ConditionType | None = None) -> uuid.UUID:
    await _ensure_tables()
    uid = uuid.uuid4()
    async with async_session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, email, timezone, onboarding_completed) "
                "VALUES (:id, :email, 'UTC', false)"
            ),
            {"id": uid, "email": f"{uid}@foodai.test"},
        )
        if condition is not None:
            session.add(UserCondition(user_id=uid, condition_type=condition))
        await session.commit()
    return uid


async def _make_food(session, name: str, components: dict[ComponentType, float]) -> uuid.UUID:
    """Create a FoodEntry carrying one or more components at given 0–4 levels."""
    food = FoodEntry(name=name)
    session.add(food)
    await session.flush()
    for comp, level in components.items():
        session.add(
            FoodComponentDetail(
                food_entry_id=food.id,
                component_type=comp,
                level=Decimal(str(level)),
            )
        )
    await session.flush()
    return food.id


async def _add_meal(session, uid: uuid.UUID, ts: datetime, food_id: uuid.UUID, name: str) -> None:
    meal = Meal(user_id=uid, timestamp=ts, meal_type=MealType.LUNCH)
    session.add(meal)
    await session.flush()
    session.add(MealItem(meal_id=meal.id, food_entry_id=food_id, name=name))
    await session.flush()


async def _add_symptom(session, uid: uuid.UUID, ts: datetime) -> None:
    session.add(
        SymptomScore(
            user_id=uid, timestamp=ts, symptom_type=SymptomType.BLOATING, vas_score=70
        )
    )


def _day(days_ago: int, hour: int = 12) -> datetime:
    base = datetime.now(UTC) - timedelta(days=days_ago)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


# ── Engine scenario: recovery ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recovery_strong_trigger_scores_high():
    """A component that consistently precedes symptoms -> positive β and high
    trigger_probability with an odds-ratio CI above 1."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        fodmap = await _make_food(session, "Garlic H", {ComponentType.FODMAP: 3.0})
        safe = await _make_food(session, "Rice H", {ComponentType.OTHER: 0.0})
        for d in range(40):
            if d % 2 == 0:
                await _add_meal(session, uid, _day(d, 9), fodmap, "Garlic H")
                await _add_symptom(session, uid, _day(d, 14))
            else:
                await _add_meal(session, uid, _day(d, 9), safe, "Rice H")
        await session.commit()

    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(session, uid, lookback_days=60)

    fod = {r.component_type: r for r in results}[ComponentType.FODMAP]
    assert fod.beta > 0.0
    assert fod.trigger_probability > 0.9
    assert fod.score > 90.0
    assert fod.ci_low > 1.0  # odds ratio interval excludes "no effect"
    assert not fod.is_cold_start
    assert fod.n_exposed > 0


# ── Engine scenario: DE-CONFOUNDING (the headline win) ────────────────────────

@pytest.mark.asyncio
async def test_deconfounding_collinear_passenger_shrinks():
    """Two collinear components where only ONE is causal: the joint logistic fit
    credits the causal component and shrinks the passenger toward its prior.

    Design (mod-4 day cycle):
      * A-only day  -> symptom          (FODMAP causes)
      * A&B day     -> symptom          (FODMAP causes; SALICYLATES rides along)
      * B-only day  -> NO symptom       (SALICYLATES innocent)
      * A&B day     -> symptom
    A univariate model would credit SALICYLATES (present on 2/3 of its days with a
    symptom); the joint fit must not."""
    uid = await _new_user()  # no condition -> no clinical seed favours either
    async with async_session_factory() as session:
        food_a = await _make_food(session, "Aonly", {ComponentType.FODMAP: 3.0})
        food_b = await _make_food(session, "Bonly", {ComponentType.SALICYLATES: 3.0})
        food_ab = await _make_food(
            session, "AandB", {ComponentType.FODMAP: 3.0, ComponentType.SALICYLATES: 3.0}
        )
        for d in range(64):
            r = d % 4
            if r == 0:
                await _add_meal(session, uid, _day(d, 9), food_a, "Aonly")
                await _add_symptom(session, uid, _day(d, 14))
            elif r == 1:
                await _add_meal(session, uid, _day(d, 9), food_ab, "AandB")
                await _add_symptom(session, uid, _day(d, 14))
            elif r == 2:
                await _add_meal(session, uid, _day(d, 9), food_b, "Bonly")
                # no symptom
            else:
                await _add_meal(session, uid, _day(d, 9), food_ab, "AandB")
                await _add_symptom(session, uid, _day(d, 14))
        await session.commit()

    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(session, uid, lookback_days=90)

    by = {r.component_type: r for r in results}
    causal = by[ComponentType.FODMAP]
    passenger = by[ComponentType.SALICYLATES]

    # The causal component is clearly credited.
    assert causal.beta > 0.5
    assert causal.trigger_probability > 0.9
    # The passenger is shrunk far below the causal component — de-confounded.
    assert passenger.beta < causal.beta - 0.5
    assert passenger.trigger_probability < causal.trigger_probability
    assert passenger.trigger_probability < 0.75
    # Headline ranking: causal outscores passenger.
    assert causal.score > passenger.score


# ── Engine scenario: partial pooling ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_partial_pooling_small_n_shrinks_rich_n_fits():
    """With an identical elevated population prior, a small-n user stays near the
    prior while a data-rich user with contradicting data overrides it."""
    prior = {ComponentType.FODMAP: (1.2, DEFAULT_COMPONENT_PRECISION)}

    # Both users: FODMAP eaten but NEVER followed by a symptom (contradicts prior).
    async def _seed(uid, n_days):
        async with async_session_factory() as session:
            fodmap = await _make_food(session, f"Fd-{uid}", {ComponentType.FODMAP: 3.0})
            for d in range(n_days):
                await _add_meal(session, uid, _day(d, 9), fodmap, "Fd")
            await session.commit()

    small_uid = await _new_user()
    rich_uid = await _new_user()
    await _seed(small_uid, 3)
    await _seed(rich_uid, 60)

    async with async_session_factory() as session:
        small = {
            r.component_type: r
            for r in await analyze_hierarchical_triggers(
                session, small_uid, lookback_days=90, population_prior=prior
            )
        }[ComponentType.FODMAP]
    async with async_session_factory() as session:
        rich = {
            r.component_type: r
            for r in await analyze_hierarchical_triggers(
                session, rich_uid, lookback_days=90, population_prior=prior
            )
        }[ComponentType.FODMAP]

    # Small-n stays close to the positive prior mean; rich-n is pulled down by its
    # (contradicting) data -> strictly lower beta and lower trigger_probability.
    assert small.beta > rich.beta
    assert small.trigger_probability > rich.trigger_probability
    # Small-n remains near the prior (still positive-ish), rich-n moved well away.
    assert small.beta > 0.5
    assert rich.beta < small.beta - 0.3


# ── Engine scenario: cold start ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cold_start_returns_condition_prior():
    """No diary data + an IBS condition -> implicated components (FODMAP, LACTOSE,
    FRUCTOSE) come back elevated from the clinical-seed prior, not zero."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        results = await analyze_hierarchical_triggers(session, uid, lookback_days=60)

    by = {r.component_type: r for r in results}
    assert ComponentType.FODMAP in by
    assert ComponentType.LACTOSE in by
    assert ComponentType.FRUCTOSE in by

    fod = by[ComponentType.FODMAP]
    assert fod.is_cold_start
    assert fod.n_obs == 0
    assert fod.n_exposed == 0
    # Elevated: positive prior beta -> trigger_probability > 0.5, score > 50.
    assert fod.beta > 0.0
    assert fod.trigger_probability > 0.5
    assert fod.score > 50.0
    # Odds-ratio point (exp(beta)) above 1.
    assert fod.ci_high > 1.0


# ── Engine scenario: determinism ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_determinism_identical_across_runs():
    """Same DB state -> byte-identical results (no sampling, deterministic ordering)."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        fodmap = await _make_food(session, "Apple Dt", {ComponentType.FODMAP: 3.0})
        safe = await _make_food(session, "Rice Dt", {ComponentType.OTHER: 0.0})
        for d in range(24):
            if d % 2 == 0:
                await _add_meal(session, uid, _day(d, 9), fodmap, "Apple Dt")
                if d % 4 == 0:
                    await _add_symptom(session, uid, _day(d, 14))
            else:
                await _add_meal(session, uid, _day(d, 9), safe, "Rice Dt")
        await session.commit()

    async with async_session_factory() as session:
        run1 = await analyze_hierarchical_triggers(session, uid, lookback_days=60)
    async with async_session_factory() as session:
        run2 = await analyze_hierarchical_triggers(session, uid, lookback_days=60)

    assert [r.to_dict() for r in run1] == [r.to_dict() for r in run2]


# ── build_population_prior over the synthetic cohort ──────────────────────────

@pytest.mark.asyncio
async def test_build_population_prior_shape_and_empty_default():
    """build_population_prior returns {comp: (mu, lambda)} for cohort components;
    the returned means are finite and clamped, precisions positive."""
    async with async_session_factory() as session:
        prior = await build_population_prior(session)
    assert isinstance(prior, dict)
    for comp, (mu, lam) in prior.items():
        assert isinstance(comp, ComponentType)
        assert np.isfinite(mu)
        assert lam > 0.0
