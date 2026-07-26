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
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.services.clinician_report_service import (
    _styles,
    _table,
    build_clinician_report_data,
)
from app.services.summary_signal import SummarySignalRow, build_summary_signal_rows

# Plain-English labels for the engine-agnostic enums the seam emits.
_DIRECTION_LABEL = {
    "trigger": "Possible trigger",
    "protective": "Not a trigger (protective)",
    "inconclusive": "Unclear",
}
_CONFIDENCE_LABEL = {
    "strong": "Strong",
    "moderate": "Moderate",
    "preliminary": "Preliminary",
    "insufficient": "Insufficient data",
}


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
    }


def _signal_section(story: list, st: dict, rows: list[SummarySignalRow]) -> None:
    """Render the trigger-signal section from stable ``SummarySignalRow`` fields only."""
    story.append(Paragraph("2. Food Signals", st["heading"]))
    if not rows:
        story.append(
            Paragraph(
                "No food reached the reporting threshold in this period. Signals appear "
                "as more meals and symptoms are logged.",
                st["body"],
            )
        )
        return

    header = [
        "Food", "Reading", "Confidence", "Days eaten",
        "Days not eaten", "Symptom days", "Odds ratio (95% CI)",
    ]
    data = [header]
    for r in rows:
        reading = _DIRECTION_LABEL.get(r.direction, r.direction)
        if r.demoted:
            reading = f"{reading} *"
        if r.odds_ratio is None or r.ci_low is None or r.ci_high is None:
            or_cell = "not testable"
        else:
            or_cell = f"{r.odds_ratio:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f})"
        data.append(
            [
                r.food_name,
                reading,
                _CONFIDENCE_LABEL.get(r.confidence, r.confidence),
                str(r.exposed_count),
                str(r.control_count),
                str(r.symptom_episodes),
                or_cell,
            ]
        )
    story.append(
        _table(
            data,
            [1.3 * inch, 1.35 * inch, 1.0 * inch, 0.7 * inch,
             0.75 * inch, 0.7 * inch, 1.4 * inch],
        )
    )
    story.append(Spacer(1, 0.05 * inch))

    # Plain-English caveat for demoted rows (rows carrying a "*").
    demoted_reasons = sorted(
        {r.demotion_reason for r in rows if r.demoted and r.demotion_reason}
    )
    if demoted_reasons:
        caveat = "  ".join(f"* {reason}." for reason in demoted_reasons)
        story.append(
            Paragraph("<b>Rows marked *</b> are shown but not confirmed: " + caveat, st["small"])
        )
        story.append(Spacer(1, 0.04 * inch))
    story.append(
        Paragraph(
            'A "Reading" describes the direction of the association only (possible '
            "trigger, not a trigger, or unclear); it is not a diagnosis. Confidence is "
            "an ordinal strength tier based on how the food's exposed vs. not-exposed "
            "days line up with symptom days, cross-checked by a classical association "
            "test. Sample sizes are shown so weak signals are not over-read. Association "
            "is not causation.",
            st["small"],
        )
    )


def render_summary_pdf(data: dict) -> bytes:
    """Render the assembled summary-report data dict to PDF bytes."""
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
    _signal_section(story, st, data["signal_rows"])

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
) -> bytes:
    """End-to-end: gather data then render the doctor/patient summary PDF."""
    data = await build_summary_report_data(db, user, lookback_days)
    return render_summary_pdf(data)


def summary_pdf_filename(user_id: uuid.UUID) -> str:
    """Stable, human-readable download filename for a user's summary PDF."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"foodai-summary-{str(user_id)[:8]}-{stamp}.pdf"
