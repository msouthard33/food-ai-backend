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
