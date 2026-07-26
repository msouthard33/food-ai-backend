"""Frequentist association guardrail (Wave 2, H2).

A lightweight *classical* association layer that runs ALONGSIDE the Beta-Binomial
Bayesian engine (``bayesian_trigger.py``) as a sanity check and to back defensible,
cohort-level research claims (DiGA / NICE style): "our personalised Bayesian signal
agrees with a classical association test".

For each candidate ``ComponentType`` we take the SAME day-level 2x2 the Bayesian
engine builds — exposed x symptom-outcome over calendar days (a food consumed at or
above the exposure threshold on a day; a symptom within the condition-appropriate lag
window) — and run a classical test of independence:

  * **chi-square** test of independence (no continuity correction), OR
  * **Fisher's exact** test (two-sided) when any expected cell count < 5.

We report a p-value and an **odds ratio** effect size (Haldane-Anscombe 0.5
correction applied when any cell is zero), then apply a **Benjamini-Hochberg FDR**
correction across all tested components (q = 0.05) to get adjusted q-values and a
significance flag. Finally, ``agreement`` compares the FDR-significant set against the
components the Bayesian model flags as likely triggers.

Consistency with H1 is guaranteed *by construction*: the guardrail consumes the exact
``(a, b, c, d)`` counts carried on each ``ComponentTriggerResult`` produced by
``analyze_bayesian_triggers``. There is no second, drifting copy of the exposure
logic.

Everything here is pure Python + ``math`` — no numpy, no scipy:
  * chi-square p-value  = regularized upper incomplete gamma Q(df/2, x/2)
  * Fisher's exact      = hypergeometric pmf via ``math.lgamma`` (log-space)
  * BH-FDR              = a sort + monotone back-fill
All deterministic: identical counts -> identical numbers.
"""

import math
import uuid
from dataclasses import asdict, dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ComponentType
from app.services.bayesian_trigger import (
    DEFAULT_LOOKBACK_DAYS,
    ComponentTriggerResult,
    analyze_bayesian_triggers,
)

# ── Tunable constants ─────────────────────────────────────────────────────────

#: Expected-cell threshold below which we switch chi-square -> Fisher's exact.
#: The classical rule of thumb for a 2x2.
MIN_EXPECTED_CELL_FOR_CHI2 = 5.0

#: Benjamini-Hochberg target false-discovery rate.
DEFAULT_FDR_Q = 0.05

#: Bayesian trigger_probability at/above which the Bayesian engine is treated as
#: "flagging" a component as a likely trigger, for the agreement comparison.
BAYESIAN_TRIGGER_THRESHOLD = 0.7


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardrailResult:
    """Classical association result for one (user, ComponentType).

    Fields:
        component_type: the ComponentType tested.
        a / b / c / d: the day-level 2x2 (identical to the Bayesian engine's) —
            a = exposed days with a symptom outcome,
            b = exposed days without,
            c = unexposed days with a symptom outcome,
            d = unexposed days without.
        test: which test produced ``p_value`` — "chi2", "fisher", or "skipped"
            (a degenerate 2x2 with an empty margin: untestable, excluded from FDR).
        p_value: raw (uncorrected) p-value, or None when skipped.
        odds_ratio: (a*d)/(b*c), with a Haldane-Anscombe +0.5 correction on every
            cell when any cell is zero. None when skipped. ``inf`` is possible only
            if the correction is somehow bypassed; it never is here.
        chi2_stat: the chi-square statistic (only when ``test == "chi2"``).
        min_expected: smallest expected cell count under independence (drives the
            chi2-vs-Fisher choice). None when skipped.
        q_value: Benjamini-Hochberg adjusted p-value across the tested family, or
            None when skipped.
        significant: True when ``q_value <= q`` (the FDR threshold used).
    """

    component_type: ComponentType
    a: int
    b: int
    c: int
    d: int
    test: str
    p_value: float | None
    odds_ratio: float | None
    chi2_stat: float | None
    min_expected: float | None
    q_value: float | None
    significant: bool

    def to_dict(self) -> dict:
        """Plain-dict view (enum rendered as its string value)."""
        out = asdict(self)
        out["component_type"] = self.component_type.value
        return out


@dataclass(frozen=True)
class AgreementReport:
    """Overlap between the Bayesian trigger set and the FDR-significant set.

    Fields:
        bayesian_trigger_set: component values with trigger_probability >= threshold.
        frequentist_significant_set: component values flagged FDR-significant.
        concordant: components in BOTH sets.
        bayesian_only: flagged by the Bayesian model but NOT FDR-significant.
        frequentist_only: FDR-significant but NOT flagged by the Bayesian model.
        jaccard: |intersection| / |union| of the two positive sets (0.0 if both empty).
        overall_concordance: fraction of jointly-considered components on which the two
            methods agree (both-positive or both-negative), over the components present
            in both result sets.
        n_components: number of components considered in ``overall_concordance``.
        trigger_threshold: the Bayesian threshold used.
        q: the FDR threshold used for the frequentist set.
    """

    bayesian_trigger_set: list[str]
    frequentist_significant_set: list[str]
    concordant: list[str]
    bayesian_only: list[str]
    frequentist_only: list[str]
    jaccard: float
    overall_concordance: float
    n_components: int
    trigger_threshold: float
    q: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Incomplete gamma / chi-square survival ────────────────────────────────────

_GAMMA_MAX_ITER = 300
_GAMMA_EPS = 3.0e-14
_GAMMA_FPMIN = 1.0e-300


def _gser(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) via its series expansion.

    Numerical-Recipes ``gser``; converges quickly for ``x < a + 1``.
    """
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_GAMMA_MAX_ITER):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _GAMMA_EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) via its continued fraction.

    Numerical-Recipes ``gcf`` (Lentz's algorithm); converges quickly for ``x >= a + 1``.
    """
    b = x + 1.0 - a
    c = 1.0 / _GAMMA_FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _GAMMA_MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _GAMMA_FPMIN:
            d = _GAMMA_FPMIN
        c = b + an / c
        if abs(c) < _GAMMA_FPMIN:
            c = _GAMMA_FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).

    Picks the series vs continued-fraction branch by the standard ``x < a + 1``
    switch so each is used only in its fast-converging region.
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError("gammq requires a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P(X > x) for ``df`` degrees of freedom.

    ``= Q(df/2, x/2)`` (the regularized upper incomplete gamma). For the 2x2 case
    (df = 1) this equals ``erfc(sqrt(x/2))`` to numerical precision.
    """
    if x <= 0.0:
        return 1.0
    return gammq(df / 2.0, x / 2.0)


# ── 2x2 primitives ────────────────────────────────────────────────────────────

def _margins(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int, int]:
    """Return (row1, row2, col1, col2, n) for the 2x2 [[a, b], [c, d]]."""
    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    return r1, r2, c1, c2, r1 + r2


def is_degenerate(a: int, b: int, c: int, d: int) -> bool:
    """True when any row or column margin is zero -> no association is testable."""
    r1, r2, c1, c2, _ = _margins(a, b, c, d)
    return r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0


def min_expected_cell(a: int, b: int, c: int, d: int) -> float:
    """Smallest expected cell count under independence: min_ij (row_i * col_j / n)."""
    r1, r2, c1, c2, n = _margins(a, b, c, d)
    if n == 0:
        return 0.0
    return min(r1 * c1, r1 * c2, r2 * c1, r2 * c2) / n


def odds_ratio_2x2(a: int, b: int, c: int, d: int) -> float:
    """Sample odds ratio (a*d)/(b*c).

    When any cell is zero a Haldane-Anscombe correction adds 0.5 to every cell,
    yielding a finite, defined estimate (the standard fix for zero-cell tables).
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return (a * d) / (b * c)


def chi_square_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Pearson chi-square test of independence for the 2x2 [[a, b], [c, d]].

    Returns ``(chi2_stat, p_value)``. No Yates continuity correction (matches
    ``scipy.stats.chi2_contingency(..., correction=False)``). Uses the closed-form
    ``chi2 = n (ad - bc)^2 / (r1 r2 c1 c2)`` and the df=1 chi-square survival function.
    Raises ``ValueError`` on a degenerate (empty-margin) table.
    """
    r1, r2, c1, c2, n = _margins(a, b, c, d)
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        raise ValueError("chi-square undefined for a table with an empty margin")
    stat = n * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)
    return stat, chi2_sf(stat, df=1)


def _hypergeom_logpmf(k: int, r1: int, c1: int, c2: int, n: int) -> float:
    """log P(cell(0,0) = k) under the hypergeometric with fixed margins.

    Population ``n`` with ``c1`` "successes" (col-1 total), ``r1`` draws (row-1
    total); log-space via ``math.lgamma`` so large margins do not overflow.
    """
    def logcomb(nn: int, kk: int) -> float:
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    return logcomb(c1, k) + logcomb(c2, r1 - k) - logcomb(n, r1)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test p-value for the 2x2 [[a, b], [c, d]].

    Enumerates every table consistent with the observed margins and sums the
    probability of those at least as extreme as observed — i.e. with probability
    ``<= P(observed)`` (a small relative tolerance guards the boundary). This is the
    same two-sided definition ``scipy.stats.fisher_exact`` uses by default.
    Raises ``ValueError`` on a degenerate (empty-margin) table.
    """
    r1, r2, c1, c2, n = _margins(a, b, c, d)
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        raise ValueError("Fisher's exact undefined for a table with an empty margin")

    log_p_obs = _hypergeom_logpmf(a, r1, c1, c2, n)
    tol = 1.0 + 1.0e-7
    p_obs = math.exp(log_p_obs)

    k_lo = max(0, r1 - c2)
    k_hi = min(r1, c1)
    total = 0.0
    for k in range(k_lo, k_hi + 1):
        p_k = math.exp(_hypergeom_logpmf(k, r1, c1, c2, n))
        if p_k <= p_obs * tol:
            total += p_k
    return min(1.0, total)


def association_test(a: int, b: int, c: int, d: int) -> tuple[str, float, float | None]:
    """Pick + run the appropriate 2x2 association test.

    Fisher's exact when any expected cell < ``MIN_EXPECTED_CELL_FOR_CHI2``, else
    Pearson chi-square. Returns ``(test_name, p_value, chi2_stat_or_None)``.
    Raises ``ValueError`` on a degenerate (empty-margin) table.
    """
    if is_degenerate(a, b, c, d):
        raise ValueError("association test undefined for a table with an empty margin")
    if min_expected_cell(a, b, c, d) < MIN_EXPECTED_CELL_FOR_CHI2:
        return "fisher", fisher_exact_2x2(a, b, c, d), None
    stat, p = chi_square_2x2(a, b, c, d)
    return "chi2", p, stat


# ── Benjamini-Hochberg FDR ────────────────────────────────────────────────────

def benjamini_hochberg(
    p_values: list[float], q: float = DEFAULT_FDR_Q
) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR across ``p_values``.

    Returns ``(adjusted_q_values, significant_flags)`` in the INPUT order. Adjusted
    values are the standard BH step-up: sort ascending, scale each by ``m / rank``,
    enforce monotonicity from the largest downward, clamp to 1.0. ``significant`` is
    ``adjusted <= q`` — equivalent to the classic "largest k with p_(k) <= k/m * q,
    reject all up to k" rule.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    prev = 1.0
    # Walk from the largest p-value down, carrying the running minimum.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = p_values[idx] * m / rank
        prev = min(prev, val)
        adjusted_sorted[rank - 1] = min(1.0, prev)

    adjusted = [0.0] * m
    for rank in range(m):
        adjusted[order[rank]] = adjusted_sorted[rank]

    significant = [adj <= q for adj in adjusted]
    return adjusted, significant


# ── Guardrail assembly ────────────────────────────────────────────────────────

def run_guardrail(
    counts: list[tuple[ComponentType, int, int, int, int]],
    q: float = DEFAULT_FDR_Q,
) -> list[GuardrailResult]:
    """Run the classical association + FDR pipeline over a list of 2x2 counts.

    ``counts`` is ``[(component_type, a, b, c, d), ...]``. Degenerate tables (an empty
    margin — e.g. a cold-start component with no observed days) are marked
    ``test="skipped"`` and EXCLUDED from the FDR family so they neither inflate ``m``
    nor claim significance. Returns one ``GuardrailResult`` per input, sorted with
    significant results first, then by ascending p-value, skipped last.
    """
    testable: list[int] = []
    p_raw: list[float] = []
    partial: list[dict] = []

    for component_type, a, b, c, d in counts:
        if is_degenerate(a, b, c, d):
            partial.append(
                {
                    "component_type": component_type,
                    "a": a, "b": b, "c": c, "d": d,
                    "test": "skipped",
                    "p_value": None,
                    "odds_ratio": None,
                    "chi2_stat": None,
                    "min_expected": None,
                }
            )
            continue
        test, p, stat = association_test(a, b, c, d)
        partial.append(
            {
                "component_type": component_type,
                "a": a, "b": b, "c": c, "d": d,
                "test": test,
                "p_value": p,
                "odds_ratio": odds_ratio_2x2(a, b, c, d),
                "chi2_stat": stat,
                "min_expected": min_expected_cell(a, b, c, d),
            }
        )
        testable.append(len(partial) - 1)
        p_raw.append(p)

    adjusted, significant = benjamini_hochberg(p_raw, q=q)
    for slot, (adj, sig) in zip(testable, zip(adjusted, significant, strict=True), strict=True):
        partial[slot]["q_value"] = adj
        partial[slot]["significant"] = sig

    results = [
        GuardrailResult(
            q_value=item.get("q_value"),
            significant=item.get("significant", False),
            **{k: item[k] for k in (
                "component_type", "a", "b", "c", "d", "test",
                "p_value", "odds_ratio", "chi2_stat", "min_expected",
            )},
        )
        for item in partial
    ]

    def sort_key(r: GuardrailResult) -> tuple:
        # significant first; then ascending p-value; skipped (None) last.
        return (
            0 if r.significant else 1,
            r.p_value if r.p_value is not None else float("inf"),
        )

    results.sort(key=sort_key)
    return results


def guardrail_from_bayesian(
    bayesian_results: list[ComponentTriggerResult],
    q: float = DEFAULT_FDR_Q,
) -> list[GuardrailResult]:
    """Run the guardrail directly on Bayesian results, reusing their 2x2 counts.

    This is the primary path: it guarantees the classical test sees EXACTLY the same
    exposed x symptom 2x2 the Bayesian engine scored, so the "our Bayesian signal
    agrees with a classical test" claim is apples-to-apples.
    """
    counts = [
        (r.component_type, r.a, r.b, r.c, r.d) for r in bayesian_results
    ]
    return run_guardrail(counts, q=q)


async def analyze_association_guardrail(
    db: AsyncSession,
    user_id: uuid.UUID,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    population_prior: dict[ComponentType, float] | None = None,
    q: float = DEFAULT_FDR_Q,
) -> list[GuardrailResult]:
    """End-to-end guardrail for a user: build the Bayesian 2x2s, then test classically.

    Delegates exposure/symptom/lag derivation entirely to
    ``analyze_bayesian_triggers`` (single source of truth), then runs the classical
    tests + FDR on the resulting counts. Read-only.
    """
    bayesian_results = await analyze_bayesian_triggers(
        db, user_id, lookback_days=lookback_days, population_prior=population_prior
    )
    return guardrail_from_bayesian(bayesian_results, q=q)


# ── Agreement ─────────────────────────────────────────────────────────────────

def agreement(
    bayesian_results: list[ComponentTriggerResult],
    guardrail_results: list[GuardrailResult],
    trigger_threshold: float = BAYESIAN_TRIGGER_THRESHOLD,
    q: float = DEFAULT_FDR_Q,
) -> AgreementReport:
    """Compare the Bayesian trigger set against the FDR-significant guardrail set.

    Bayesian "flags" a component when ``trigger_probability >= trigger_threshold``;
    the frequentist positive set is the FDR-significant components. Reports the
    Jaccard overlap of the two positive sets, the concordant / disagreeing lists,
    and an overall concordance over the components both methods actually scored
    (a "skipped" guardrail component — no testable 2x2 — is excluded from overall
    concordance since the frequentist method offered no verdict).
    """
    bayes_pos = {
        r.component_type.value
        for r in bayesian_results
        if r.trigger_probability >= trigger_threshold
    }
    freq_pos = {r.component_type.value for r in guardrail_results if r.significant}

    inter = bayes_pos & freq_pos
    union = bayes_pos | freq_pos
    jaccard = len(inter) / len(union) if union else 0.0

    # Components with a frequentist verdict (not skipped) that the Bayesian side also
    # scored -> the shared universe for overall concordance.
    freq_verdict = {
        r.component_type.value: r.significant
        for r in guardrail_results
        if r.test != "skipped"
    }
    bayes_flag = {
        r.component_type.value: (r.trigger_probability >= trigger_threshold)
        for r in bayesian_results
    }
    shared = set(freq_verdict) & set(bayes_flag)
    agree_n = sum(
        1 for comp in shared if freq_verdict[comp] == bayes_flag[comp]
    )
    overall = agree_n / len(shared) if shared else 0.0

    return AgreementReport(
        bayesian_trigger_set=sorted(bayes_pos),
        frequentist_significant_set=sorted(freq_pos),
        concordant=sorted(inter),
        bayesian_only=sorted(bayes_pos - freq_pos),
        frequentist_only=sorted(freq_pos - bayes_pos),
        jaccard=jaccard,
        overall_concordance=overall,
        n_components=len(shared),
        trigger_threshold=trigger_threshold,
        q=q,
    )
