"""Doctor/Patient Summary — trigger-signal provider seam (Wave 2, Pillar 4).

This module is the SINGLE boundary between the shareable doctor/patient summary
report and whatever trigger engine is live underneath it. The report generator,
PDF renderer, and share endpoint import ONLY:

  * ``SummarySignalRow`` (the stable, engine-agnostic row the template renders), and
  * ``build_summary_signal_rows`` (the seam that produces those rows).

Per ``01 - Project Planning/SUMMARY_REPORT_SIGNAL_CONTRACT.md`` the row deliberately
exposes NONE of the current engine's output shape — no 0-100 hierarchical score, no
``P(beta>0)``, no ``combined_score``, no ``_driver_for_food``, and no hard-coded lag
window. Direction is a 3-state readout; confidence is an ordinal tier; lag lives
entirely inside the interim association test (``assoc_guardrail`` -> the day-level 2x2
whose exposure/lag/outcome logic is owned by ``bayesian_trigger``).

INTERIM source of truth: the transparent per-food association test in
``assoc_guardrail`` (Fisher/chi-square 2x2 with a Haldane-Anscombe odds ratio and
BH-FDR significance). When the rewired engine (``TRIGGER_ENGINE_HANDOFF.md``) clears
its Phase-3 gates, only ``build_summary_signal_rows`` changes — the template/export
layer and its snapshot tests stay byte-for-byte identical.

Everything here is pure Python + ``math`` (no numpy/scipy) and deterministic: the
same DB state produces the same rows.
"""

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.meal import Meal
from app.services.assoc_guardrail import GuardrailResult, analyze_association_guardrail
from app.services.bayesian_trigger import DEFAULT_LOOKBACK_DAYS
from app.services.hierarchical_trigger import food_components_by_name

# ── Tunable constants (report-facing thresholds, NOT engine internals) ─────────

#: Wald z for the 95% odds-ratio interval (matches the 95% CIs used elsewhere).
_Z_95 = 1.96

#: Sample-size floor for a "strong" ordinal tier: enough exposed AND control days AND
#: symptom days that a significant association is not resting on a thin margin.
_ADEQUATE_EXPOSED = 5
_ADEQUATE_CONTROL = 5
_ADEQUATE_SYMPTOM_DAYS = 5


# ── The stable contract row ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SummarySignalRow:
    """One food's signal, in the engine-agnostic shape the report layer renders.

    The template/export layer must reference ONLY these fields. See the signal
    contract doc for the field-level rules; ``direction``/``confidence``/``demoted``
    are the honest 3-state / ordinal / honesty-flag readouts, and the OR + interval +
    p-value are supporting detail only (never the headline).
    """

    food_name: str

    # honest 3-state direction (never a raw score)
    direction: str            # "trigger" | "protective" | "inconclusive"

    # ordinal, engine-agnostic confidence tier (never a bare percentage)
    confidence: str           # "strong" | "moderate" | "preliminary" | "insufficient"

    # honesty flag: True when the signal is shown but must be downweighted
    demoted: bool
    demotion_reason: str | None

    # sample-size transparency (always shown alongside the signal)
    exposed_count: int
    control_count: int
    symptom_episodes: int

    # supporting effect detail — nullable; interim association test values
    odds_ratio: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    test: str                 # "fisher" | "chi2" | "skipped"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Pure derivation helpers (unit-testable, no DB) ─────────────────────────────

def odds_ratio_ci(
    a: int, b: int, c: int, d: int, z: float = _Z_95
) -> tuple[float, float, float]:
    """95% Wald confidence interval for the 2x2 odds ratio, on the log scale.

    Returns ``(odds_ratio, ci_low, ci_high)``. A Haldane-Anscombe +0.5 correction is
    applied to every cell when any cell is zero so the log-OR and its standard error
    stay finite (the standard zero-cell fix, identical to ``assoc_guardrail``'s OR).
    Deterministic and dependency-free.
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        af, bf, cf, df = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        af, bf, cf, df = float(a), float(b), float(c), float(d)
    or_ = (af * df) / (bf * cf)
    se = math.sqrt(1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df)
    log_or = math.log(or_)
    return or_, math.exp(log_or - z * se), math.exp(log_or + z * se)


def _direction_from_ci(test: str, ci_low: float | None, ci_high: float | None) -> str:
    """3-state direction from the odds-ratio interval relative to 1.0.

    ``protective`` when the interval lies entirely below 1.0, ``trigger`` when
    entirely above, ``inconclusive`` when it straddles 1.0 (or the test was skipped).
    """
    if test == "skipped" or ci_low is None or ci_high is None:
        return "inconclusive"
    if ci_high < 1.0:
        return "protective"
    if ci_low > 1.0:
        return "trigger"
    return "inconclusive"


def _confidence_tier(
    *, test: str, significant: bool, exposed_count: int, control_count: int,
    symptom_episodes: int,
) -> str:
    """Ordinal confidence tier from FDR significance + sample size.

    ``insufficient`` for a degenerate/skipped test; ``strong`` for an FDR-significant
    association with adequate exposed/control/symptom counts; ``moderate`` for
    significant-but-thin; ``preliminary`` for a testable-but-not-significant signal.
    """
    if test == "skipped":
        return "insufficient"
    if significant:
        adequate = (
            exposed_count >= _ADEQUATE_EXPOSED
            and control_count >= _ADEQUATE_CONTROL
            and symptom_episodes >= _ADEQUATE_SYMPTOM_DAYS
        )
        return "strong" if adequate else "moderate"
    return "preliminary"


def _demotion(direction: str, confidence: str) -> tuple[bool, str | None]:
    """Honesty flag: is this row shown but downweighted, and why.

    A row is a confirmed-trigger headline ONLY when its direction is ``trigger`` AND
    its confidence is ``strong`` or ``moderate``. Everything else is demoted so the
    report never implies a trigger the evidence does not support.
    """
    if direction == "trigger" and confidence in ("strong", "moderate"):
        return False, None
    if direction == "protective":
        return True, "protective association (odds ratio below 1) — not a trigger signal"
    if direction == "inconclusive":
        return True, "confidence interval spans 1.0 — direction not established"
    # direction == "trigger" but confidence preliminary/insufficient
    if confidence == "insufficient":
        return True, "insufficient exposed/control days to test this association"
    return True, "below the significance/sample threshold — preliminary only"


def derive_signal_row(food_name: str, guard: GuardrailResult) -> SummarySignalRow:
    """Turn one food's driving guardrail result into a stable ``SummarySignalRow``.

    Pure and deterministic: all direction/confidence/demotion logic lives here, keyed
    only off the classical 2x2 + FDR verdict, so it can be unit-tested without a DB.
    """
    a, b, c, d = guard.a, guard.b, guard.c, guard.d
    exposed_count = a + b
    control_count = c + d
    symptom_episodes = a + c

    if guard.test == "skipped":
        odds_ratio = ci_low = ci_high = None
    else:
        odds_ratio, ci_low, ci_high = odds_ratio_ci(a, b, c, d)

    direction = _direction_from_ci(guard.test, ci_low, ci_high)
    confidence = _confidence_tier(
        test=guard.test,
        significant=guard.significant,
        exposed_count=exposed_count,
        control_count=control_count,
        symptom_episodes=symptom_episodes,
    )
    demoted, reason = _demotion(direction, confidence)

    return SummarySignalRow(
        food_name=food_name,
        direction=direction,
        confidence=confidence,
        demoted=demoted,
        demotion_reason=reason,
        exposed_count=exposed_count,
        control_count=control_count,
        symptom_episodes=symptom_episodes,
        odds_ratio=round(odds_ratio, 4) if odds_ratio is not None else None,
        ci_low=round(ci_low, 4) if ci_low is not None else None,
        ci_high=round(ci_high, 4) if ci_high is not None else None,
        p_value=round(guard.p_value, 8) if guard.p_value is not None else None,
        test=guard.test,
    )


def _pick_driver(
    guards: list[GuardrailResult],
) -> GuardrailResult | None:
    """Pick the guardrail result that best characterises a food's signal.

    Significant results first, then the smallest raw p-value; a skipped/degenerate
    result is used only when nothing testable is available. Fully deterministic (the
    component value is the final tie-break).
    """
    if not guards:
        return None

    def key(g: GuardrailResult) -> tuple:
        return (
            0 if g.significant else 1,
            g.p_value if g.p_value is not None else float("inf"),
            g.component_type.value,
        )

    return sorted(guards, key=key)[0]


def _row_sort_key(row: SummarySignalRow) -> tuple:
    """Order rows for display: undemoted triggers first, by confidence then p-value."""
    conf_rank = {"strong": 0, "moderate": 1, "preliminary": 2, "insufficient": 3}
    return (
        0 if not row.demoted else 1,
        conf_rank.get(row.confidence, 9),
        row.p_value if row.p_value is not None else float("inf"),
        row.food_name.lower(),
    )


# ── The provider seam ──────────────────────────────────────────────────────────

async def build_summary_signal_rows(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[SummarySignalRow]:
    """Produce the summary report's signal rows for a user (the ONLY report seam).

    Interim implementation: runs the classical association guardrail (which owns all
    exposure/lag/outcome logic via the shared Bayesian 2x2), maps each logged food to
    the KB components it carries, and emits one ``SummarySignalRow`` per food from its
    driving component's classical verdict. Read-only and deterministic.

    NOTE: the ONLY lag/window reference in the whole pipeline is inside
    ``analyze_association_guardrail`` — never here and never in the report layer.
    """
    guardrail = await analyze_association_guardrail(db, user_id, lookback_days=lookback_days)
    by_comp = {g.component_type: g for g in guardrail}

    # Distinct foods the user logged in the reporting window (a plain date filter, not
    # a lag window). We only speak to foods the patient actually ate.
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    meal_result = await db.execute(
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
    meals = list(meal_result.scalars().unique().all())
    names: set[str] = set()
    for meal in meals:
        for item in meal.items:
            if item.name:
                names.add(item.name)

    if not names:
        return []

    name_comps = await food_components_by_name(db, names)

    rows: list[SummarySignalRow] = []
    for name in names:
        comps = name_comps.get(name.strip().lower(), set())
        guards = [by_comp[c] for c in comps if c in by_comp]
        driver = _pick_driver(guards)
        if driver is None:
            # No KB-scored component -> nothing defensible to say about this food.
            continue
        rows.append(derive_signal_row(name, driver))

    rows.sort(key=_row_sort_key)
    return rows
