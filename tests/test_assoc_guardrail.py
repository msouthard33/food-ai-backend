"""Tests for the frequentist association guardrail (H2).

Three layers:
  * Pure-math unit tests for the classical stats (chi-square, Fisher's exact,
    odds ratio, BH-FDR) — no DB, no async. Several p-values are cross-checked
    against values produced by ``scipy.stats`` (a VERIFICATION-ONLY install; scipy
    is NOT a project dependency). The scipy reference numbers are recorded inline.
  * ``agreement`` unit tests over hand-built Bayesian + guardrail results.
  * One async engine scenario that seeds a tiny diary and confirms the end-to-end
    ``analyze_association_guardrail`` path (which reuses the Bayesian 2x2s) flags a
    clear association as FDR-significant.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.models.enums import ComponentType, ConditionType, MealType, SymptomType
from app.models.food import FoodComponentDetail, FoodEntry
from app.models.meal import Meal, MealItem
from app.models.symptom import SymptomScore
from app.models.user import UserCondition
from app.services.assoc_guardrail import (
    AgreementReport,
    GuardrailResult,
    agreement,
    analyze_association_guardrail,
    association_test,
    benjamini_hochberg,
    chi2_sf,
    chi_square_2x2,
    fisher_exact_2x2,
    guardrail_from_bayesian,
    is_degenerate,
    min_expected_cell,
    odds_ratio_2x2,
    run_guardrail,
)
from app.services.bayesian_trigger import ComponentTriggerResult, analyze_bayesian_triggers
from tests.conftest import _ensure_tables, async_session_factory

# ── Chi-square ────────────────────────────────────────────────────────────────

def test_chi2_sf_matches_erfc_at_critical_value():
    import math

    # The df=1 5% critical value is 3.841459; its survival must be ~0.05, and the
    # incomplete-gamma implementation must agree with the exact erfc form.
    x = 3.841458820694124
    assert abs(chi2_sf(x, 1) - 0.05) < 1e-6
    assert abs(chi2_sf(x, 1) - math.erfc(math.sqrt(x / 2.0))) < 1e-12


def test_chi_square_2x2_matches_scipy():
    # scipy.stats.chi2_contingency(tbl, correction=False):
    #   [[10, 2],[3, 15]]  -> stat 13.0316742081, p 0.000306266630184
    #   [[20, 30],[25, 25]]-> stat  1.0101010101, p 0.314878641336
    #   [[40, 10],[20, 30]]-> stat 16.6666666667, p 4.45570906041e-05
    stat, p = chi_square_2x2(10, 2, 3, 15)
    assert abs(stat - 13.0316742081) < 1e-6
    assert abs(p - 0.000306266630184) < 1e-9

    stat, p = chi_square_2x2(20, 30, 25, 25)
    assert abs(stat - 1.0101010101) < 1e-6
    assert abs(p - 0.314878641336) < 1e-9

    stat, p = chi_square_2x2(40, 10, 20, 30)
    assert abs(stat - 16.6666666667) < 1e-6
    assert abs(p - 4.45570906041e-05) < 1e-10


def test_chi_square_degenerate_raises():
    with pytest.raises(ValueError):
        chi_square_2x2(0, 0, 5, 5)  # empty row margin


# ── Fisher's exact ────────────────────────────────────────────────────────────

def test_fisher_exact_matches_scipy():
    # scipy.stats.fisher_exact(tbl) two-sided:
    #   [[3, 1],[1, 3]] -> p 0.485714285714  (classic tea-tasting table)
    #   [[8, 2],[1, 5]] -> p 0.034965034965
    #   [[0, 5],[8, 3]] -> p 0.025641025641
    #   [[5, 0],[1, 9]] -> p 0.001998001998
    assert abs(fisher_exact_2x2(3, 1, 1, 3) - 0.485714285714) < 1e-9
    assert abs(fisher_exact_2x2(8, 2, 1, 5) - 0.034965034965) < 1e-9
    assert abs(fisher_exact_2x2(0, 5, 8, 3) - 0.025641025641) < 1e-9
    assert abs(fisher_exact_2x2(5, 0, 1, 9) - 0.001998001998) < 1e-9


def test_fisher_exact_symmetry_of_row_swap():
    # Swapping the two rows leaves the two-sided p-value unchanged.
    assert abs(fisher_exact_2x2(8, 2, 1, 5) - fisher_exact_2x2(1, 5, 8, 2)) < 1e-12


# ── Odds ratio ────────────────────────────────────────────────────────────────

def test_odds_ratio_plain_and_haldane():
    # No zero cell -> plain (a*d)/(b*c).
    assert abs(odds_ratio_2x2(10, 2, 3, 15) - (10 * 15) / (2 * 3)) < 1e-9
    # Zero cell -> Haldane-Anscombe +0.5 to every cell (finite, defined).
    # scipy reports OR=0.0 (uncorrected) for [[0,5],[8,3]]; we deliberately correct.
    assert abs(odds_ratio_2x2(0, 5, 8, 3) - (0.5 * 3.5) / (5.5 * 8.5)) < 1e-9
    # scipy reports OR=inf for [[5,0],[1,9]]; ours is finite via correction.
    assert odds_ratio_2x2(5, 0, 1, 9) < float("inf")


# ── Test selection + helpers ──────────────────────────────────────────────────

def test_min_expected_and_test_selection():
    # Large, balanced counts -> all expected cells >= 5 -> chi-square chosen.
    assert min_expected_cell(40, 10, 20, 30) >= 5.0
    test, _p, stat = association_test(40, 10, 20, 30)
    assert test == "chi2" and stat is not None

    # Small counts -> an expected cell < 5 -> Fisher chosen.
    assert min_expected_cell(3, 1, 1, 3) < 5.0
    test, _p, stat = association_test(3, 1, 1, 3)
    assert test == "fisher" and stat is None


def test_is_degenerate():
    assert is_degenerate(0, 0, 5, 5)  # empty exposed row
    assert is_degenerate(5, 0, 5, 0)  # empty symptom-free col
    assert not is_degenerate(3, 1, 1, 3)


# ── Benjamini-Hochberg FDR ────────────────────────────────────────────────────

def test_bh_matches_scipy_reference():
    # scipy.stats.false_discovery_control(p, method='bh'):
    #   [0.001, 0.008, 0.039, 0.041, 0.9] -> [0.005, 0.02, 0.05125, 0.05125, 0.9]
    adj, sig = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.9], q=0.05)
    expected = [0.005, 0.02, 0.05125, 0.05125, 0.9]
    for got, exp in zip(adj, expected, strict=True):
        assert abs(got - exp) < 1e-9
    # At q=0.05 the first two are significant, the rest are not.
    assert sig == [True, True, False, False, False]


def test_bh_preserves_input_order():
    # Same p-values, shuffled -> adjusted values track their inputs, not position.
    adj, _ = benjamini_hochberg([0.9, 0.041, 0.001, 0.039, 0.008], q=0.05)
    # index 2 held 0.001 -> smallest adjusted; index 0 held 0.9 -> largest.
    assert abs(adj[2] - 0.005) < 1e-9
    assert abs(adj[0] - 0.9) < 1e-9


def test_bh_monotonic_in_rank():
    # Adjusted q-values, taken in ascending-p order, are non-decreasing.
    pvals = [0.002, 0.01, 0.03, 0.2, 0.5, 0.7]
    adj, _ = benjamini_hochberg(pvals, q=0.05)
    ordered = [adj[i] for i in sorted(range(len(pvals)), key=lambda i: pvals[i])]
    assert all(ordered[i] <= ordered[i + 1] + 1e-12 for i in range(len(ordered) - 1))
    assert all(0.0 <= v <= 1.0 for v in adj)


def test_bh_empty():
    assert benjamini_hochberg([], q=0.05) == ([], [])


# ── run_guardrail ─────────────────────────────────────────────────────────────

def test_run_guardrail_flags_clear_association_not_null():
    # FODMAP: strong association (all exposed days symptomatic, few unexposed).
    # OTHER: null (symptoms independent of exposure).
    counts = [
        (ComponentType.FODMAP, 18, 2, 3, 17),   # strong
        (ComponentType.OTHER, 10, 10, 10, 10),  # null
    ]
    results = run_guardrail(counts, q=0.05)
    by = {r.component_type: r for r in results}

    assert by[ComponentType.FODMAP].significant is True
    assert by[ComponentType.FODMAP].odds_ratio > 1.0
    assert by[ComponentType.OTHER].significant is False
    # q-values were assigned for every tested component.
    assert by[ComponentType.FODMAP].q_value is not None
    assert by[ComponentType.OTHER].q_value is not None


def test_run_guardrail_skips_degenerate_and_excludes_from_fdr():
    counts = [
        (ComponentType.FODMAP, 18, 2, 3, 17),  # testable, strongly significant
        (ComponentType.LACTOSE, 0, 0, 0, 0),   # cold-start / degenerate -> skipped
    ]
    results = run_guardrail(counts, q=0.05)
    by = {r.component_type: r for r in results}

    assert by[ComponentType.LACTOSE].test == "skipped"
    assert by[ComponentType.LACTOSE].q_value is None
    assert by[ComponentType.LACTOSE].significant is False
    # The lone testable component has an FDR family size of 1 -> q == raw p.
    strong = by[ComponentType.FODMAP]
    assert abs(strong.q_value - strong.p_value) < 1e-12
    assert strong.significant is True


# ── agreement ─────────────────────────────────────────────────────────────────

def _bayes(component: ComponentType, prob: float) -> ComponentTriggerResult:
    """A minimal ComponentTriggerResult carrying just what agreement() reads."""
    return ComponentTriggerResult(
        component_type=component,
        trigger_probability=prob,
        score=prob * 100.0,
        ci_low=0.0,
        ci_high=0.0,
        a=0, b=0, c=0, d=0,
        n_exposed_days=0,
        n_symptom_days=0,
        alpha_post=1.0, beta_post=1.0,
        alpha_unexposed_post=1.0, beta_unexposed_post=1.0,
        prior_alpha=1.0, prior_beta=1.0,
        is_cold_start=False,
    )


def _guard(component: ComponentType, significant: bool, test: str = "chi2") -> GuardrailResult:
    return GuardrailResult(
        component_type=component,
        a=0, b=0, c=0, d=0,
        test=test,
        p_value=0.01 if significant else 0.5,
        odds_ratio=2.0,
        chi2_stat=1.0,
        min_expected=10.0,
        q_value=0.01 if significant else 0.5,
        significant=significant,
    )


def test_agreement_full_concordance():
    bayes = [_bayes(ComponentType.FODMAP, 0.9), _bayes(ComponentType.LACTOSE, 0.2)]
    guard = [_guard(ComponentType.FODMAP, True), _guard(ComponentType.LACTOSE, False)]
    rep = agreement(bayes, guard)

    assert isinstance(rep, AgreementReport)
    assert rep.concordant == ["fodmap"]
    assert rep.bayesian_only == []
    assert rep.frequentist_only == []
    assert rep.jaccard == 1.0
    assert rep.overall_concordance == 1.0
    assert rep.n_components == 2


def test_agreement_disagreements_split():
    # FODMAP: both agree (trigger). HISTAMINES: Bayesian-only. LACTOSE: freq-only.
    bayes = [
        _bayes(ComponentType.FODMAP, 0.95),
        _bayes(ComponentType.HISTAMINES, 0.8),
        _bayes(ComponentType.LACTOSE, 0.3),
    ]
    guard = [
        _guard(ComponentType.FODMAP, True),
        _guard(ComponentType.HISTAMINES, False),
        _guard(ComponentType.LACTOSE, True),
    ]
    rep = agreement(bayes, guard)

    assert rep.concordant == ["fodmap"]
    assert rep.bayesian_only == ["histamines"]
    assert rep.frequentist_only == ["lactose"]
    # positive sets: bayes {fodmap,histamines}, freq {fodmap,lactose} -> J = 1/3.
    assert abs(rep.jaccard - 1.0 / 3.0) < 1e-9
    # 3 shared components, agree only on fodmap -> 1/3 overall concordance.
    assert abs(rep.overall_concordance - 1.0 / 3.0) < 1e-9


def test_agreement_skipped_excluded_from_overall():
    bayes = [_bayes(ComponentType.FODMAP, 0.9), _bayes(ComponentType.LACTOSE, 0.9)]
    guard = [
        _guard(ComponentType.FODMAP, True),
        _guard(ComponentType.LACTOSE, False, test="skipped"),
    ]
    rep = agreement(bayes, guard)
    # LACTOSE has no frequentist verdict -> excluded from the shared universe.
    assert rep.n_components == 1
    assert rep.overall_concordance == 1.0
    # But it still counts as Bayesian-only in the positive-set comparison.
    assert "lactose" in rep.bayesian_only


def test_agreement_empty_positive_sets():
    bayes = [_bayes(ComponentType.FODMAP, 0.1)]
    guard = [_guard(ComponentType.FODMAP, False)]
    rep = agreement(bayes, guard)
    assert rep.jaccard == 0.0          # both positive sets empty
    assert rep.overall_concordance == 1.0  # agree that it is NOT a trigger
    assert rep.n_components == 1


# ── Async end-to-end (reuses the Bayesian 2x2s) ───────────────────────────────

async def _new_user(condition: ConditionType | None = None) -> uuid.UUID:
    from sqlalchemy import text

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


def _day(days_ago: int, hour: int = 12) -> datetime:
    base = datetime.now(UTC) - timedelta(days=days_ago)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


@pytest.mark.asyncio
async def test_end_to_end_guardrail_flags_strong_association():
    """A component that always precedes a symptom -> the classical guardrail marks it
    FDR-significant, and agreement() shows the Bayesian + frequentist verdicts concur."""
    uid = await _new_user(ConditionType.IBS)
    async with async_session_factory() as session:
        food = FoodEntry(name="Garlic G")
        session.add(food)
        await session.flush()
        session.add(
            FoodComponentDetail(
                food_entry_id=food.id,
                component_type=ComponentType.FODMAP,
                level=Decimal("3.0"),
            )
        )
        safe = FoodEntry(name="Rice G")
        session.add(safe)
        await session.flush()
        session.add(
            FoodComponentDetail(
                food_entry_id=safe.id,
                component_type=ComponentType.OTHER,
                level=Decimal("0.0"),
            )
        )
        await session.flush()

        for d in range(28):
            if d % 2 == 0:
                meal = Meal(user_id=uid, timestamp=_day(d, 9), meal_type=MealType.LUNCH)
                session.add(meal)
                await session.flush()
                session.add(MealItem(meal_id=meal.id, food_entry_id=food.id, name="Garlic G"))
                session.add(
                    SymptomScore(
                        user_id=uid, timestamp=_day(d, 14),
                        symptom_type=SymptomType.BLOATING, vas_score=70,
                    )
                )
            else:
                meal = Meal(user_id=uid, timestamp=_day(d, 9), meal_type=MealType.LUNCH)
                session.add(meal)
                await session.flush()
                session.add(MealItem(meal_id=meal.id, food_entry_id=safe.id, name="Rice G"))
        await session.commit()

    async with async_session_factory() as session:
        bayes = await analyze_bayesian_triggers(session, uid, lookback_days=40)
        guard = await analyze_association_guardrail(session, uid, lookback_days=40)

    by = {r.component_type: r for r in guard}
    fodmap = by[ComponentType.FODMAP]
    assert fodmap.test in ("chi2", "fisher")
    assert fodmap.significant is True
    assert fodmap.odds_ratio > 1.0

    # The two engines built the SAME 2x2 for FODMAP.
    b_fodmap = {r.component_type: r for r in bayes}[ComponentType.FODMAP]
    assert (fodmap.a, fodmap.b, fodmap.c, fodmap.d) == (
        b_fodmap.a, b_fodmap.b, b_fodmap.c, b_fodmap.d,
    )

    rep = agreement(bayes, guard)
    assert "fodmap" in rep.concordant


@pytest.mark.asyncio
async def test_guardrail_from_bayesian_consumes_counts():
    """guardrail_from_bayesian must reproduce run_guardrail on the same counts."""
    bayes = [_bayes(ComponentType.FODMAP, 0.9)]
    object.__setattr__(bayes[0], "a", 18)
    object.__setattr__(bayes[0], "b", 2)
    object.__setattr__(bayes[0], "c", 3)
    object.__setattr__(bayes[0], "d", 17)
    from_bayes = guardrail_from_bayesian(bayes, q=0.05)
    direct = run_guardrail([(ComponentType.FODMAP, 18, 2, 3, 17)], q=0.05)
    assert from_bayes[0].p_value == direct[0].p_value
    assert from_bayes[0].significant == direct[0].significant
