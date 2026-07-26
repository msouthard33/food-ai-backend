"""Clinician PDF report service — Pillar 4 / Box 11 (Clinician Trust Layer).

Assembles a clinician-facing PDF for a patient/doctor review session. The layout
mirrors the structure of a standard GI/allergist symptom-history intake, so a
clinician can read it the way they read their own intake paperwork:

  1. Patient & report metadata (identity, reporting window, disclaimer)
  2. Symptom timeline (chief-complaint history, Rome IV-style symptom framing)
  3. Suspect-foods leaderboard WITH confidence + sample size (Box 8 fields)
  4. Elimination-protocol outcomes
  5. Patient-reported outcomes (PRO) — VAS symptom summary + daily wellness check-ins
  6. Methodology / honest-confidence caveats + non-diagnostic disclaimer

Layout reference (documented, public): the section ordering and symptom-history
framing follow the Rome IV Diagnostic Questionnaire for Functional GI Disorders and
a conventional GI symptom-history intake (chief complaint -> symptom timeline ->
dietary triggers -> interventions -> outcomes). This is NOT yet reviewed against a
real GI/allergist intake form — a real-clinician review remains an owed human gate
before Box 11 is marked fully satisfied (see W2-5 sprint report).

All correlation logic is REUSED from the insights leaderboard endpoint; this service
does not re-implement any scoring.
"""

import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.symptom import DailyCheckin, SymptomScore
from app.models.user import User
from app.services.medication_service import get_medicated_symptom_map

# Shared palette (matches report_service.py)
COLOR_PRIMARY = "#2563EB"
COLOR_TEXT_DARK = "#1F2937"
COLOR_BG_LIGHT = "#F3F4F6"


def _fmt_symptom(symptom_type: object) -> str:
    """Human-readable symptom label from an enum/string value."""
    raw = str(getattr(symptom_type, "value", symptom_type))
    return raw.replace("SymptomType.", "").replace("_", " ").title()


async def build_clinician_report_data(
    db: AsyncSession,
    user: User,
    lookback_days: int,
) -> dict:
    """Gather every section's data for the clinician PDF.

    Reuses the insights suspect-foods leaderboard for the trigger section so the
    confidence + sample-size fields (combined_score, ci_low/ci_high, n_meals,
    n_symptom_episodes, confidence_label, medication_confounded) are identical to
    what the API surfaces.
    """
    # Imported lazily to avoid a service<-router import at module load time.
    from app.models.sensitivity import UserSensitivityProfile
    from app.routers.insights import get_suspect_foods

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # -- Symptom timeline (non-deleted, chronological) ------------------------
    symptom_result = await db.execute(
        select(SymptomScore)
        .where(
            and_(
                SymptomScore.user_id == user.id,
                SymptomScore.timestamp >= cutoff,
                SymptomScore.deleted_at.is_(None),
            )
        )
        .order_by(SymptomScore.timestamp)
    )
    symptoms = list(symptom_result.scalars().all())
    medicated_map = await get_medicated_symptom_map(db, user.id, [s.id for s in symptoms])

    timeline = [
        {
            "timestamp": s.timestamp,
            "symptom": _fmt_symptom(s.symptom_type),
            "vas_score": int(s.vas_score),
            "medicated": s.id in medicated_map,
            "notes": s.notes or "",
        }
        for s in symptoms
    ]

    # -- Suspect-foods leaderboard (REUSED insights logic) --------------------
    suspect = await get_suspect_foods(lookback_days=lookback_days, user=user, db=db)
    suspect_rows = [
        {
            "food_name": row.food_name,
            "combined_score": row.combined_score,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
            "n_meals": row.n_meals,
            "n_symptom_episodes": row.n_symptom_episodes,
            "confidence_label": row.confidence_label,
            "medication_confounded": row.medication_confounded,
        }
        for row in suspect.foods
    ]

    # -- Elimination-protocol outcomes ----------------------------------------
    protocol_result = await db.execute(
        select(UserSensitivityProfile)
        .where(
            UserSensitivityProfile.user_id == user.id,
            UserSensitivityProfile.deleted_at.is_(None),
        )
        .order_by(UserSensitivityProfile.created_at)
    )
    protocols = [
        {
            "component": str(getattr(p.component_type, "value", p.component_type)),
            "status": "Active" if p.active else "Ended",
            "notes": p.notes or "",
            "started": p.created_at,
        }
        for p in protocol_result.scalars().all()
    ]

    # -- Patient-reported outcomes (PRO) --------------------------------------
    # 1) Per-symptom VAS summary (VAS = a patient-reported outcome measure).
    pro_by_symptom: dict[str, dict] = {}
    for s in symptoms:
        label = _fmt_symptom(s.symptom_type)
        agg = pro_by_symptom.setdefault(label, {"count": 0, "sum": 0, "peak": 0})
        agg["count"] += 1
        agg["sum"] += int(s.vas_score)
        agg["peak"] = max(agg["peak"], int(s.vas_score))
    pro_symptom_summary = [
        {
            "symptom": label,
            "count": agg["count"],
            "avg_vas": round(agg["sum"] / agg["count"], 1) if agg["count"] else 0,
            "peak_vas": agg["peak"],
        }
        for label, agg in sorted(pro_by_symptom.items())
    ]

    # 2) Daily wellness check-ins (PRO-style self-report), if any.
    checkin_result = await db.execute(
        select(DailyCheckin)
        .where(
            and_(
                DailyCheckin.user_id == user.id,
                DailyCheckin.check_date >= cutoff.date(),
            )
        )
        .order_by(DailyCheckin.check_date)
    )
    checkins = list(checkin_result.scalars().all())

    def _avg(vals: list) -> float | None:
        nums = [float(v) for v in vals if v is not None]
        return round(sum(nums) / len(nums), 1) if nums else None

    pro_wellness = None
    if checkins:
        pro_wellness = {
            "n_checkins": len(checkins),
            "avg_overall_wellness": _avg([c.overall_wellness for c in checkins]),
            "avg_stress_level": _avg([c.stress_level for c in checkins]),
            "avg_sleep_quality": _avg([c.sleep_quality for c in checkins]),
            "avg_sleep_hours": _avg([c.sleep_hours for c in checkins]),
        }

    return {
        "user": user,
        "lookback_days": lookback_days,
        "period_start": cutoff.date().isoformat(),
        "period_end": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc),
        "timeline": timeline,
        "suspect_foods": suspect_rows,
        "protocols": protocols,
        "pro_symptom_summary": pro_symptom_summary,
        "pro_wellness": pro_wellness,
    }


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ClinTitle", parent=base["Heading1"], fontSize=20,
            textColor=colors.HexColor(COLOR_PRIMARY), spaceAfter=4, alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "ClinSubtitle", parent=base["BodyText"], fontSize=9,
            textColor=colors.HexColor(COLOR_TEXT_DARK), alignment=1, spaceAfter=6,
        ),
        "heading": ParagraphStyle(
            "ClinHeading", parent=base["Heading2"], fontSize=13,
            textColor=colors.HexColor(COLOR_PRIMARY), spaceBefore=14, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "ClinBody", parent=base["BodyText"], fontSize=9.5,
            textColor=colors.HexColor(COLOR_TEXT_DARK),
        ),
        "small": ParagraphStyle(
            "ClinSmall", parent=base["BodyText"], fontSize=8,
            textColor=colors.HexColor(COLOR_TEXT_DARK),
        ),
    }


def _table(data: list[list[str]], col_widths: list[float]) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_PRIMARY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor(COLOR_BG_LIGHT)]),
            ]
        )
    )
    return t


def render_clinician_pdf(data: dict) -> bytes:
    """Render the assembled clinician-report data dict to PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        title="Food AI Clinician Report",
    )
    st = _styles()
    story: list = []

    user: User = data["user"]

    # -- Header + patient/report metadata -------------------------------------
    story.append(Paragraph("Food AI - Clinician Summary Report", st["title"]))
    story.append(
        Paragraph(
            "Educational dietary-symptom tracking summary - not a diagnostic instrument",
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
        story.append(Spacer(1, 0.05 * inch))
        story.append(
            Paragraph(
                "Severity is a patient-reported Visual Analog Scale (0-100). "
                '"Medicated" flags episodes with a co-logged medication in the '
                "surrounding window (severity may be pharmacologically blunted).",
                st["small"],
            )
        )
    else:
        story.append(Paragraph("No symptoms recorded in this period.", st["body"]))

    # -- 2. Suspect-foods leaderboard (confidence + sample size) --------------
    story.append(Paragraph("2. Suspect Foods (Correlation Leaderboard)", st["heading"]))
    suspect = data["suspect_foods"]
    if suspect:
        rows = [
            [
                "Food", "Score", "95% CI", "Meals (n)",
                "Symptom episodes (n)", "Confidence", "Med-confounded",
            ]
        ]
        for f in suspect:
            rows.append(
                [
                    f["food_name"],
                    f"{f['combined_score']:.0f}",
                    f"{f['ci_low']:.0f}-{f['ci_high']:.0f}",
                    str(f["n_meals"]),
                    str(f["n_symptom_episodes"]),
                    f["confidence_label"],
                    "Yes" if f["medication_confounded"] else "No",
                ]
            )
        story.append(
            _table(
                rows,
                [1.5 * inch, 0.6 * inch, 0.9 * inch, 0.75 * inch,
                 1.15 * inch, 1.1 * inch, 1.05 * inch],
            )
        )
        story.append(Spacer(1, 0.05 * inch))
        story.append(
            Paragraph(
                "Score is a medication-adjusted association strength (0-100) from a "
                "hierarchical Bayesian model. The 95% interval shown is the driving "
                "component's odds-ratio credible interval, and the n-of-meals / "
                "n-of-symptom-episode sample sizes are shown so weak signals are not "
                "over-read. Association is not causation; foods here warrant clinical "
                "correlation, not automatic elimination.",
                st["small"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No food reached the reporting threshold (>=3 associated symptom "
                "episodes) in this period.",
                st["body"],
            )
        )

    # -- 3. Elimination-protocol outcomes -------------------------------------
    story.append(Paragraph("3. Elimination Protocol Outcomes", st["heading"]))
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
        story.append(
            _table(rows, [1.7 * inch, 0.9 * inch, 1.1 * inch, 2.8 * inch])
        )
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

    # -- 5. Methodology / disclaimer ------------------------------------------
    story.append(Paragraph("5. Methodology &amp; Important Caveats", st["heading"]))
    story.append(
        Paragraph(
            "Suspect-food scores are computed by a hierarchical Bayesian logistic model "
            "that jointly de-confounds co-occurring food components over a "
            "condition-appropriate symptom-onset lag window, adjusted for co-logged "
            "medication as a confounder, and cross-checked against a classical "
            "association test (chi-square / Fisher's exact with Benjamini-Hochberg FDR). "
            "Confidence labels reflect statistical strength (sample size and interval "
            "width), not the point score alone.",
            st["small"],
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        Paragraph(
            "<b>Report structure</b> follows a conventional GI symptom-history intake "
            "(chief complaint -> symptom timeline -> dietary triggers -> interventions -> "
            "outcomes) and Rome IV functional-GI symptom framing. This layout has NOT "
            "yet been reviewed against a specific clinic's intake form; a clinician "
            "review is a pending validation step.",
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


async def generate_clinician_pdf(
    db: AsyncSession,
    user: User,
    lookback_days: int = 30,
) -> bytes:
    """End-to-end: gather data then render the clinician PDF, returning bytes."""
    data = await build_clinician_report_data(db, user, lookback_days)
    return render_clinician_pdf(data)


def clinician_pdf_filename(user_id: uuid.UUID) -> str:
    """Stable, human-readable download filename for a user's clinician PDF."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"foodai-clinician-report-{str(user_id)[:8]}-{stamp}.pdf"
