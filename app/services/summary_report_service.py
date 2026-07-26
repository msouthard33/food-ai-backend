"""Doctor/Patient Summary report — generation + PDF (Wave 2, Pillar 4).

A shareable summary a patient can hand to (or send) their clinician. It reuses the
clinician-report PDF infrastructure (styles, table renderer, symptom timeline, PRO
section, protocol section) but the TRIGGER-SIGNAL SECTION is sourced EXCLUSIVELY from
the stable provider seam ``summary_signal.build_summary_signal_rows`` — never from the
suspect-food leaderboard that is being rewired.

This file (the report/template/export layer) references NONE of the current trigger
engine's output shape and NO hard-coded lag window (see the signal contract doc for
the exact banned identifiers). Everything it knows about the signal arrives as
``SummarySignalRow`` fields. When the engine is rewired, this file does not change.
"""

import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO

from sqlalchemy import func, select

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meal import Meal
from app.models.user import User
from app.services.clinician_report_service import (
    _styles,
    _table,
    build_clinician_report_data,
)
from app.services.summary_signal import (
    DEMOTION_REASON_CODES,
    DemotionReason,
    SummarySignalRow,
    build_summary_signal_rows,
)

# ── §5 canonical copy tables (W2_summary_report_layout.md — TONE GATE CLOSED
#    2026-07-26). The grouped definition-list is CANONICAL over the deprecated
#    7-column table (§8.3). All strings are the FINAL approved (neutralized/hybrid)
#    versions; the pinned verb "line up with", the pinned trigger header, and the
#    pinned protective-strong string are preserved verbatim. A bold tier word always
#    leads; a number never leads and never stands bare (D9 doctrine). ─────────────

#: §5.1 fixed group order + exact subheadings (plain-English, non-color glyph label).
_GROUP_ORDER = ("trigger", "protective", "inconclusive")
_GROUP_HEADING = {
    "trigger": "Worth discussing with your doctor",
    "protective": "Seems to sit well with you so far",
    "inconclusive": "No clear pattern yet",
}

#: §5.2 signal readouts, keyed by (direction, confidence). Bold tier word leads.
#: `{food}` = food_name verbatim. inconclusive collapses across all four tiers.
_READOUT = {
    ("trigger", "strong"): (
        "<b>Worth discussing</b> — across enough logs, {food} consistently "
        "lined up with your higher-symptom days."
    ),
    ("trigger", "moderate"): (
        "<b>Some evidence</b> — {food} tended to line up with your "
        "higher-symptom days."
    ),
    ("trigger", "preliminary"): (
        "<b>Early signal</b> — {food} has begun to line up with your "
        "higher-symptom days, but it's too soon to say."
    ),
    ("trigger", "insufficient"): (
        "<b>Not enough yet</b> — {food} came up, but there aren't enough logs "
        "yet to say whether it lines up with your symptoms."
    ),
    ("protective", "strong"): (
        "<b>Sits well so far</b> — across enough logs, {food} did <i>not</i> "
        "line up with your higher-symptom days. That's a reassuring sign, not a "
        "guarantee."
    ),
    ("protective", "moderate"): (
        "<b>Looks okay so far</b> — {food} tended not to line up with your "
        "higher-symptom days in this window."
    ),
    ("protective", "preliminary"): (
        "<b>Leaning okay</b> — early logs suggest {food} isn't lining up with "
        "your symptoms, but it's still early."
    ),
    ("protective", "insufficient"): (
        "<b>Not enough yet</b> — {food} came up, but there aren't enough logs "
        "yet to say either way."
    ),
}
#: §5.2 inconclusive — one string for any confidence tier (no tier gradient).
_INCONCLUSIVE_READOUT = (
    "<b>No clear pattern</b> — {food} showed up in your logs, but it didn't "
    "clearly line up <i>with</i> or <i>against</i> your symptoms in this window."
)

#: §5.4 demoted-caveat strings, selected by EXACT ``demotion_reason`` code (OQ-4).
#: Plain-language only; never names the guardrail test, FDR, p-value, or the engine.
_CAVEAT_SMALL_SAMPLE = (
    "We're showing this for completeness, but it's based on very few logs — "
    "please read it as a hint, not a finding."
)
_CAVEAT_DISAGREE = (
    "Our check-tests don't fully agree on this one yet, so we've held it back from a "
    "stronger reading. It needs more logs before it means much."
)
_CAVEAT_MIXED = (
    "The signal here is mixed — the pattern points one way but isn't consistent "
    "enough to lean on. Treat it as unsettled."
)
_CAVEAT_GENERIC = (
    "We're showing this with extra caution — the evidence isn't strong enough "
    "yet to read it as more than an early hint."
)

#: §5.6 empty / thin-window state (warm patient voice retained).
_EMPTY_STATE = (
    "<b>No food signals yet — here's why.</b> It usually takes two to three "
    "weeks of logging before patterns are steady enough to show here. Your {meals} "
    "meals and {symptom_entries} symptom entries so far are already building the "
    "picture. Keep logging what you can, and this section fills in."
)

#: §5.4 caveat map — EXACT ``demotion_reason`` code -> approved caveat string (OQ-4).
#: Keyed by the canonical ``DemotionReason`` codes (the ONE source of truth in the
#: seam), so a demoted row's caveat is chosen by exact match, never substring. Every
#: code maps to one of the four approved §5.4 strings; ``PROTECTIVE`` (and a bare
#: ``None``) intentionally take the generic hedge, since the spec defines no
#: protective-specific caveat.
_CAVEAT_BY_REASON: dict[str, str] = {
    DemotionReason.INSUFFICIENT_SAMPLE.value: _CAVEAT_SMALL_SAMPLE,
    DemotionReason.BELOW_THRESHOLD.value: _CAVEAT_SMALL_SAMPLE,
    DemotionReason.GUARDRAIL_DISAGREES.value: _CAVEAT_DISAGREE,
    DemotionReason.INCONCLUSIVE.value: _CAVEAT_MIXED,
    DemotionReason.PROTECTIVE.value: _CAVEAT_GENERIC,
}


def _validate_caveat_map() -> None:
    """Fail loudly at import if the caveat map drifts out of sync with the seam.

    Every canonical ``DemotionReason`` code MUST have an explicit caveat and no caveat
    may key off a code the seam can't emit. This is the import-time half of OQ-4's
    guard (the test suite asserts the same invariant); together they make it impossible
    for a renamed/added code to silently fall through to the generic hedge.
    """
    missing = set(DEMOTION_REASON_CODES) - set(_CAVEAT_BY_REASON)
    unknown = set(_CAVEAT_BY_REASON) - set(DEMOTION_REASON_CODES)
    if missing or unknown:
        raise RuntimeError(
            "§5.4 caveat map out of sync with DemotionReason "
            f"(missing caveats for {missing}; unknown codes {unknown})"
        )


_validate_caveat_map()

#: ordinal confidence rank for within-group sort (never a numeric-score sort).
_CONF_RANK = {"strong": 0, "moderate": 1, "preliminary": 2, "insufficient": 3}


def _readout_string(r: SummarySignalRow) -> str:
    """The FINAL §5.2 readout for a row's (direction, confidence), food interpolated."""
    if r.direction == "inconclusive":
        template = _INCONCLUSIVE_READOUT
    else:
        template = _READOUT.get(
            (r.direction, r.confidence),
            _INCONCLUSIVE_READOUT,
        )
    return template.format(food=r.food_name)


def _count_phrase(n: int, word: str) -> str:
    """Bold-number, singular/plural sample-size fragment (e.g. '<b>1</b> day')."""
    return f"<b>{n}</b> {word}" if n == 1 else f"<b>{n}</b> {word}s"


def _sample_size_line(r: SummarySignalRow) -> str:
    """§5.3 mandatory honest-denominator line — rendered on EVERY row, no exceptions."""
    return (
        f"Based on {_count_phrase(r.exposed_count, 'day')} you logged {r.food_name} "
        f"and {_count_phrase(r.control_count, 'day')} you didn't, across "
        f"{_count_phrase(r.symptom_episodes, 'symptom episode')} in this window."
    )


def _demoted_caveat(reason: str | None) -> str:
    """§5.4 caveat for a demoted row, by EXACT ``demotion_reason`` code (OQ-4 closed).

    The code is a canonical ``summary_signal.DemotionReason`` value, matched exactly —
    never by substring — so every code (including ``guardrail_disagrees`` -> the
    "check-tests disagree" caveat, previously unreachable) resolves to its approved
    §5.4 string. ``None`` (no specific reason) takes the generic hedge; any non-None
    code that is NOT in the canonical set raises rather than silently degrading to the
    generic caveat, so a drifted/renamed code is caught instead of hidden.
    """
    if reason is None:
        return _CAVEAT_GENERIC
    try:
        return _CAVEAT_BY_REASON[reason]
    except KeyError:
        raise ValueError(
            f"unmapped demotion_reason code {reason!r}; every code in "
            "summary_signal.DemotionReason must have a §5.4 caveat"
        ) from None


def _supporting_detail_line(r: SummarySignalRow) -> str | None:
    """§5.5 clinician-only supporting detail. None unless every field is present."""
    if r.test == "skipped":
        return None
    if r.odds_ratio is None or r.ci_low is None or r.ci_high is None:
        return None
    line = (
        f"Supporting detail (for clinicians): odds ratio {r.odds_ratio:.2f} "
        f"(95% CI {r.ci_low:.2f}–{r.ci_high:.2f}), {r.test} test"
    )
    if r.p_value is not None:
        line += f", p = {r.p_value:.3f}"
    return line + "."


def _group_sort_key(r: SummarySignalRow) -> tuple:
    """Within-group order: undemoted first, then confidence tier, then episodes desc.

    Never sorts by any numeric score (there is none on the contract). Demoted rows sink
    to the bottom of their direction group so a demoted trigger never sits among the
    confirmed 'Worth discussing' rows (§5.4).
    """
    return (
        1 if r.demoted else 0,
        _CONF_RANK.get(r.confidence, 9),
        -r.symptom_episodes,
        r.food_name.lower(),
    )


async def build_summary_report_data(
    db: AsyncSession,
    user: User,
    lookback_days: int,
) -> dict:
    """Gather the shareable summary's sections.

    Timeline, protocol status, and patient-reported outcomes are reused verbatim from
    ``build_clinician_report_data``. The signal rows come ONLY from the provider seam.
    """
    base = await build_clinician_report_data(db, user, lookback_days)
    signal_rows = await build_summary_signal_rows(
        db, user.id, lookback_days=lookback_days
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    meals_count = await db.scalar(
        select(func.count())
        .select_from(Meal)
        .where(
            Meal.user_id == user.id,
            Meal.timestamp >= cutoff,
            Meal.deleted_at.is_(None),
        )
    )
    return {
        "user": base["user"],
        "lookback_days": base["lookback_days"],
        "period_start": base["period_start"],
        "period_end": base["period_end"],
        "generated_at": base["generated_at"],
        "timeline": base["timeline"],
        "protocols": base["protocols"],
        "pro_symptom_summary": base["pro_symptom_summary"],
        "pro_wellness": base["pro_wellness"],
        "signal_rows": signal_rows,
        "meals_count": int(meals_count or 0),
        "symptom_entries_count": len(base["timeline"]),
    }


def _signal_section(
    story: list,
    st: dict,
    rows: list[SummarySignalRow],
    *,
    clinician_detail: bool = False,
    meals: int | None = None,
    symptom_entries: int | None = None,
) -> None:
    """Render §2 as the CANONICAL grouped definition-list (not the deprecated table).

    Rows are grouped by ``direction`` in the fixed §5.1 order under the exact
    subheadings; within a group they sort by ordinal confidence then ``symptom_episodes``
    desc (never by a numeric score), with demoted rows sunk to the bottom and rendered at
    reduced weight (§5.4). Each row renders, in linear screen-reader order (§8.3):
    (a) the FINAL §5.2 readout, (b) the mandatory §5.3 sample-size line, (c) the §5.4
    caveat when demoted, and (d) the §5.5 supporting detail ONLY when ``clinician_detail``
    is on (default OFF patient-side). References ONLY ``SummarySignalRow`` fields.
    """
    story.append(Paragraph("2. Food Signals", st["heading"]))

    # §5.6 empty / thin-window state — never a dead end (warm patient voice retained).
    if not rows:
        story.append(
            Paragraph(
                _EMPTY_STATE.format(
                    meals="your" if meals is None else meals,
                    symptom_entries="your" if symptom_entries is None else symptom_entries,
                ),
                st["body"],
            )
        )
        return

    subhead = ParagraphStyle(
        "SigSubhead", parent=st["heading"], fontSize=11,
        spaceBefore=10, spaceAfter=4,
    )
    readout_style = st["body"]
    detail_style = ParagraphStyle(
        "SigDetail", parent=st["small"], leftIndent=12,
    )
    demoted_readout_style = ParagraphStyle(
        "SigDemotedReadout", parent=st["small"], leftIndent=12,
        textColor=colors.HexColor("#6B6B6B"),
    )

    by_direction: dict[str, list[SummarySignalRow]] = {d: [] for d in _GROUP_ORDER}
    for r in rows:
        by_direction.setdefault(r.direction, []).append(r)

    for direction in _GROUP_ORDER:
        group = by_direction.get(direction, [])
        if not group:
            continue
        story.append(Paragraph(_GROUP_HEADING[direction], subhead))
        for r in sorted(group, key=_group_sort_key):
            # (a) readout — demoted rows render at reduced weight (§5.4), never a
            #     confirmed 'Worth discussing' headline.
            story.append(
                Paragraph(
                    _readout_string(r),
                    demoted_readout_style if r.demoted else readout_style,
                )
            )
            # (b) mandatory sample-size line on EVERY row (§5.3).
            story.append(Paragraph(_sample_size_line(r), detail_style))
            # (c) honesty caveat immediately after, when demoted (§5.4).
            if r.demoted:
                story.append(Paragraph(_demoted_caveat(r.demotion_reason), detail_style))
            # (d) clinician-only supporting detail, small aside (§5.5).
            if clinician_detail:
                detail = _supporting_detail_line(r)
                if detail is not None:
                    story.append(Paragraph(detail, detail_style))
            story.append(Spacer(1, 0.04 * inch))


def render_summary_pdf(data: dict, *, clinician_detail: bool = False) -> bytes:
    """Render the assembled summary-report data dict to PDF bytes.

    ``clinician_detail`` gates the §5.5 supporting-detail (odds ratio / CI) line; it is
    OFF by default for the patient-facing share and may be turned on for a
    clinician-requested export.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        title="Food AI Doctor/Patient Summary",
    )
    st = _styles()
    story: list = []
    user: User = data["user"]

    # -- Header + metadata ----------------------------------------------------
    story.append(Paragraph("Food AI - Doctor / Patient Summary", st["title"]))
    story.append(
        Paragraph(
            "A shareable dietary-symptom summary to review together - not a diagnostic instrument",
            st["subtitle"],
        )
    )
    patient_label = user.display_name or user.email or str(user.id)
    story.append(
        Paragraph(
            f"<b>Patient:</b> {patient_label} &nbsp;|&nbsp; "
            f"<b>Reporting period:</b> {data['period_start']} to {data['period_end']} "
            f"({data['lookback_days']} days) &nbsp;|&nbsp; "
            f"<b>Generated:</b> {data['generated_at'].strftime('%Y-%m-%d %H:%M UTC')}",
            st["body"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    # -- 1. Symptom timeline --------------------------------------------------
    story.append(Paragraph("1. Symptom Timeline", st["heading"]))
    timeline = data["timeline"]
    if timeline:
        rows = [["Date / Time", "Symptom", "Severity (VAS 0-100)", "Medicated", "Notes"]]
        for e in timeline:
            note = e["notes"]
            rows.append(
                [
                    e["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    e["symptom"],
                    str(e["vas_score"]),
                    "Yes" if e["medicated"] else "No",
                    (note[:40] + "...") if len(note) > 40 else note,
                ]
            )
        story.append(
            _table(rows, [1.3 * inch, 1.4 * inch, 1.3 * inch, 0.8 * inch, 1.9 * inch])
        )
    else:
        story.append(Paragraph("No symptoms recorded in this period.", st["body"]))

    # -- 2. Food signals (from the provider seam ONLY) ------------------------
    _signal_section(
        story,
        st,
        data["signal_rows"],
        clinician_detail=clinician_detail,
        meals=data.get("meals_count"),
        symptom_entries=data.get("symptom_entries_count"),
    )

    # -- 3. Elimination-protocol status ---------------------------------------
    story.append(Paragraph("3. Elimination Protocol Status", st["heading"]))
    protocols = data["protocols"]
    if protocols:
        rows = [["Component / Protocol", "Status", "Started", "Notes"]]
        for p in protocols:
            started = p["started"].strftime("%Y-%m-%d") if p["started"] else "-"
            note = p["notes"]
            rows.append(
                [
                    p["component"].replace("_", " ").title(),
                    p["status"],
                    started,
                    (note[:45] + "...") if len(note) > 45 else note,
                ]
            )
        story.append(_table(rows, [1.7 * inch, 0.9 * inch, 1.1 * inch, 2.8 * inch]))
    else:
        story.append(Paragraph("No elimination protocols on record.", st["body"]))

    # -- 4. Patient-reported outcomes (PRO) -----------------------------------
    story.append(Paragraph("4. Patient-Reported Outcomes (PRO)", st["heading"]))
    pro = data["pro_symptom_summary"]
    if pro:
        rows = [["Symptom", "Episodes (n)", "Mean VAS", "Peak VAS"]]
        for p in pro:
            rows.append([p["symptom"], str(p["count"]), str(p["avg_vas"]), str(p["peak_vas"])])
        story.append(_table(rows, [2.2 * inch, 1.3 * inch, 1.5 * inch, 1.5 * inch]))
    else:
        story.append(Paragraph("No patient-reported symptom scores in this period.", st["body"]))

    wellness = data["pro_wellness"]
    if wellness:
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            Paragraph(
                f"<b>Daily wellness check-ins ({wellness['n_checkins']}):</b> "
                f"mean overall wellness {wellness['avg_overall_wellness']}, "
                f"stress {wellness['avg_stress_level']}, "
                f"sleep quality {wellness['avg_sleep_quality']}, "
                f"sleep hours {wellness['avg_sleep_hours']} "
                "(patient self-report, 0-10 scales unless noted).",
                st["small"],
            )
        )

    # -- 5. Disclaimer --------------------------------------------------------
    story.append(Paragraph("5. Important Caveats", st["heading"]))
    story.append(
        Paragraph(
            "Food signals are produced by a transparent per-food association test that "
            "compares symptom days on days a food was eaten against days it was not, and "
            "reports the direction (possible trigger / not a trigger / unclear), an "
            "ordinal confidence tier, and the underlying sample sizes. Readings that are "
            "not confirmed are shown but flagged so they are not over-interpreted.",
            st["small"],
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> Food AI is an educational dietary-tracking tool. This "
            "summary is for informational use during a patient-clinician conversation "
            "and is not a diagnosis, medical device output, or substitute for "
            "professional medical judgment.",
            st["small"],
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


async def generate_summary_pdf(
    db: AsyncSession,
    user: User,
    lookback_days: int = 30,
    *,
    clinician_detail: bool = False,
) -> bytes:
    """End-to-end: gather data then render the doctor/patient summary PDF."""
    data = await build_summary_report_data(db, user, lookback_days)
    return render_summary_pdf(data, clinician_detail=clinician_detail)


def summary_pdf_filename(user_id: uuid.UUID) -> str:
    """Stable, human-readable download filename for a user's summary PDF."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"foodai-summary-{str(user_id)[:8]}-{stamp}.pdf"
