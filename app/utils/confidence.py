"""D9 confidence tier mapping — canonical vocabulary for all AI output."""

import math


def confidence_to_tier_label(confidence: float) -> str:
    """Map a 0.0–1.0 confidence score to the D9 canonical tier label.

    D9 mapping:
      >= 0.85 → "Well-established"
      0.55–0.84 → "Some evidence"
      < 0.55 → "AI estimate"
    """
    if confidence >= 0.85:
        return "Well-established"
    if confidence >= 0.55:
        return "Some evidence"
    return "AI estimate"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Returns ``(low, high)`` as proportions in [0.0, 1.0]. The Wilson interval is
    used rather than the naive normal approximation because it stays inside [0, 1]
    and behaves sensibly at the small sample sizes typical of a personal food diary.

    ``z`` defaults to 1.96 (two-sided 95%). ``n <= 0`` returns ``(0.0, 0.0)``.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


# ── Beta-Binomial Bayesian math (pure Python, deterministic) ──────────────────
#
# Hand-rolled Beta distribution helpers so the Bayesian trigger engine can compute
# credible intervals and posterior comparisons WITHOUT numpy/scipy (not installed;
# the team hand-rolls stats — see wilson_interval above). Everything here is
# deterministic: analytic / numerical methods only, no Monte-Carlo, no `random`.
# This matters because these numbers are surfaced to patients and clinicians — the
# same data must always produce the same interval.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function.

    Lentz's algorithm, transcribed from Numerical Recipes (``betacf``). Converges
    rapidly for ``x < (a+1)/(a+b+2)``; callers must apply the symmetry transform
    otherwise (see ``_betai``).
    """
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` = the Beta(a, b) CDF at ``x``.

    Numerical-Recipes formulation: a closed-form prefactor times the continued
    fraction ``_betacf``, with the ``x <-> 1-x`` / ``a <-> b`` symmetry applied so
    the continued fraction is always evaluated in its fast-converging region.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    prefactor = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _betacf(a, b, x) / a
    return 1.0 - prefactor * _betacf(b, a, 1.0 - x) / b


def beta_mean(a: float, b: float) -> float:
    """Mean of Beta(a, b) = ``a / (a + b)``."""
    return a / (a + b)


def beta_ppf(p: float, a: float, b: float, tol: float = 1.0e-6) -> float:
    """Quantile (inverse CDF) of Beta(a, b): the ``x`` with ``I_x(a, b) = p``.

    Solved by bisection on the monotone CDF ``_betai`` (no derivative needed, so it
    is robust for any a, b > 0). ``p`` outside (0, 1) clamps to the support {0, 1}.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    # ~40 halvings would drive the bracket below tol; cap iterations generously.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def beta_credible_interval(a: float, b: float, mass: float = 0.95) -> tuple[float, float]:
    """Equal-tailed ``mass`` credible interval for Beta(a, b).

    Returns ``(lower, upper)`` as the ``(1-mass)/2`` and ``1-(1-mass)/2`` quantiles —
    e.g. the (0.025, 0.975) quantiles for the default 95%. Equal-tailed rather than
    HPD: simpler, deterministic, and for the unimodal posteriors here the difference
    is negligible.
    """
    tail = (1.0 - mass) / 2.0
    return (beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b))


def _beta_pdf(x: float, a: float, b: float) -> float:
    """Density of Beta(a, b) at ``x``, via log-space for numerical stability.

    Endpoints return 0.0. When a < 1 or b < 1 the true density diverges at 0 or 1,
    but the divergence is integrable and, in ``prob_beta_exceeds``, is multiplied by
    a CDF factor that vanishes there — so clamping the endpoint to 0 is safe.
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    ln_pdf = (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - ln_beta
    return math.exp(ln_pdf)


def prob_beta_exceeds(
    a1: float, b1: float, a2: float, b2: float, intervals: int = 256
) -> float:
    """``P(X > Y)`` for independent ``X ~ Beta(a1, b1)``, ``Y ~ Beta(a2, b2)``.

    Deterministic composite-Simpson integration of ``∫₀¹ f_X(x) · F_Y(x) dx`` where
    ``f_X`` is the Beta(a1, b1) density and ``F_Y`` is the Beta(a2, b2) CDF
    (``_betai``). This is the exact identity for P(X > Y); no sampling is involved,
    so the result is reproducible to numerical precision. ``intervals`` is forced
    even (Simpson requires an even number of sub-intervals).
    """
    n = intervals if intervals % 2 == 0 else intervals + 1
    h = 1.0 / n
    total = 0.0
    for i in range(n + 1):
        x = i * h
        value = _beta_pdf(x, a1, b1) * _betai(a2, b2, x)
        if i == 0 or i == n:
            weight = 1.0
        elif i % 2 == 1:
            weight = 4.0
        else:
            weight = 2.0
        total += weight * value
    result = total * h / 3.0
    return max(0.0, min(1.0, result))


def evidence_confidence_label(n_symptom_episodes: int, ci_width: float) -> str:
    """Plain-English confidence label for a suspect-food signal.

    Grounded in *statistical* strength (sample size + interval width), not the point
    score — a high score off two data points is not confidence. ``ci_width`` is the
    95% interval width expressed on the same 0–100 scale as the score.

    - "Strong signal":   >= 5 supporting episodes AND a tight interval (width <= 25)
    - "Emerging signal": >= 3 supporting episodes
    - "Preliminary":     everything below that (surfaced, but caveated)
    """
    if n_symptom_episodes >= 5 and ci_width <= 25.0:
        return "Strong signal"
    if n_symptom_episodes >= 3:
        return "Emerging signal"
    return "Preliminary"
