"""Report generation service with PDF generation."""

import uuid
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.pdfgen import canvas
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meal import Meal
from app.models.report import InsightReport
from app.models.symptom import SymptomScore
from app.models.trigger import TriggerPrediction
from app.schemas.report import ReportGenerateRequest


# Color palette
COLOR_PRIMARY = "#2563EB"  # Blue
COLOR_TEXT_DARK = "#1F2937"  # Dark gray
COLOR_BG_LIGHT = "#F3F4F6"  # Light gray


def generate_pdf(json_summary: dict, report_type: str, user_id: uuid.UUID, report_id: uuid.UUID) -> bytes:
    """Generate a PDF report from JSON summary data.

    Args:
        json_summary: Dictionary containing report data (period, meals_logged, symptom_summary, top_triggers)
        report_type: Type of report (weekly, monthly, custom, clinician)
        user_id: User ID (for reference)
        report_id: Report ID (for unique naming)

    Returns:
        PDF bytes
    """
    # Create BytesIO buffer for PDF
    pdf_buffer = BytesIO()

    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    # Get sample styles and create custom styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor(COLOR_PRIMARY),
        spaceAfter=6,
        alignment=1,  # Center
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor(COLOR_PRIMARY),
        spaceAfter=12,
        spaceBefore=12,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        textColor=colors.HexColor(COLOR_TEXT_DARK),
        alignment=0,  # Left
    )

    # Build story (content)
    story = []

    # Header with title and metadata
    title = Paragraph("Food AI - Dietary Sensitivity Report", title_style)
    story.append(title)

    # Report type and date range
    period_start = json_summary.get("period", {}).get("start", "N/A")
    period_end = json_summary.get("period", {}).get("end", "N/A")
    subtitle = Paragraph(
        f"<b>Report Type:</b> {report_type.title()} | "
        f"<b>Period:</b> {period_start} to {period_end}",
        body_style,
    )
    story.append(subtitle)
    story.append(Spacer(1, 0.3 * inch))

    # Section 1: Period Summary
    story.append(Paragraph("Period Summary", heading_style))

    meals_logged = json_summary.get("meals_logged", 0)
    period_obj = json_summary.get("period", {})
    start_date = period_obj.get("start")
    end_date = period_obj.get("end")

    if start_date and end_date:
        try:
            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()
            days_covered = (end - start).days + 1
        except (ValueError, TypeError):
            days_covered = 0
    else:
        days_covered = 0

    summary_text = Paragraph(
        f"<b>Meals Logged:</b> {meals_logged} | "
        f"<b>Days Covered:</b> {days_covered} days",
        body_style,
    )
    story.append(summary_text)
    story.append(Spacer(1, 0.2 * inch))

    # Section 2: Symptom Summary Table
    story.append(Paragraph("Symptom Summary", heading_style))

    symptom_data = json_summary.get("symptom_summary", [])
    if symptom_data:
        # Build symptom table
        symptom_table_data = [
            ["Symptom Type", "Count", "Avg Severity", "Peak Severity"]
        ]

        for symptom in symptom_data:
            symptom_type = str(symptom.get("type", "")).replace("SymptomType.", "").replace("_", " ").title()
            count = symptom.get("count", 0)
            avg_sev = symptom.get("avg_severity", 0)
            peak_sev = symptom.get("peak_severity", 0)

            symptom_table_data.append([
                symptom_type,
                str(count),
                f"{avg_sev}/10",
                f"{peak_sev}/10",
            ])

        symptom_table = Table(symptom_table_data, colWidths=[2 * inch, 1 * inch, 1.5 * inch, 1.5 * inch])
        symptom_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(COLOR_BG_LIGHT)),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(COLOR_BG_LIGHT)]),
        ]))
        story.append(symptom_table)
    else:
        story.append(Paragraph("No symptoms recorded during this period.", body_style))

    story.append(Spacer(1, 0.2 * inch))

    # Section 3: Top Triggers Table
    story.append(Paragraph("Top Dietary Triggers", heading_style))

    triggers = json_summary.get("top_triggers", [])
    if triggers:
        # Build triggers table
        triggers_table_data = [
            ["Component", "Confidence", "Status", "Evidence Count"]
        ]

        for trigger in triggers:
            component = str(trigger.get("component", "")).replace("ComponentType.", "").replace("_", " ").title()
            confidence = trigger.get("confidence", 0)
            status = str(trigger.get("status", "")).replace("TriggerStatus.", "").title()
            evidence = trigger.get("evidence_count", 0)

            triggers_table_data.append([
                component,
                f"{confidence}%",
                status,
                str(evidence),
            ])

        triggers_table = Table(triggers_table_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        triggers_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(COLOR_BG_LIGHT)),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(COLOR_BG_LIGHT)]),
        ]))
        story.append(triggers_table)
    else:
        story.append(Paragraph("No triggers identified during this period.", body_style))

    story.append(Spacer(1, 0.2 * inch))

    # Section 4: Recommendations
    story.append(Paragraph("Recommendations", heading_style))

    recommendations = []

    # Generate recommendations based on triggers
    if triggers:
        high_confidence_triggers = [t for t in triggers if t.get("confidence", 0) >= 70]
        if high_confidence_triggers:
            trigger_list = ", ".join([
                str(t.get("component", "")).replace("ComponentType.", "").replace("_", " ").title()
                for t in high_confidence_triggers[:3]
            ])
            recommendations.append(
                f"<b>Avoid High-Confidence Triggers:</b> Consider eliminating {trigger_list} from your diet for 2-4 weeks to assess symptom improvement."
            )

        probable_triggers = [t for t in triggers if t.get("status") == "TriggerStatus.probable"]
        if probable_triggers:
            recommendations.append(
                "<b>Monitor Probable Triggers:</b> Keep a detailed log of these components and correlate with symptom onset times."
            )

    # Symptom-based recommendations
    if symptom_data:
        high_severity_symptoms = [s for s in symptom_data if s.get("peak_severity", 0) >= 7]
        if high_severity_symptoms:
            recommendations.append(
                "<b>Seek Medical Consultation:</b> Symptoms with high severity scores warrant professional assessment."
            )

    # General recommendations
    recommendations.extend([
        "<b>Track Meals Consistently:</b> Continue logging all meals and symptoms to refine trigger identification.",
        "<b>Follow Up Reports:</b> Generate monthly reports to track progress and identify new patterns.",
    ])

    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Spacer(1, 0.3 * inch))

    # Footer
    footer_text = Paragraph(
        f"<i>Generated by Food AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        "For informational purposes only. Not a substitute for professional medical advice.</i>",
        body_style,
    )
    story.append(footer_text)

    # Build PDF
    doc.build(story)

    # Get bytes
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _save_pdf_to_disk(pdf_bytes: bytes, report_id: uuid.UUID) -> str:
    """Save PDF bytes to disk and return the path.

    Args:
        pdf_bytes: PDF content as bytes
        report_id: Report ID for unique filename

    Returns:
        Path to saved PDF file
    """
    # Create reports directory if it doesn't exist
    reports_dir = Path("/tmp/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save PDF
    pdf_path = reports_dir / f"{report_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    return str(pdf_path)


async def generate_report(
    db: AsyncSession,
    user_id: uuid.UUID,
    data: ReportGenerateRequest,
) -> InsightReport:
    """Generate a comprehensive report with JSON summary and PDF.

    Builds a JSON summary from meals, symptoms, and triggers, then generates
    a PDF document and saves it to disk.

    Args:
        db: Database session
        user_id: User ID
        data: Report generation request (report_type, date_range_start, date_range_end)

    Returns:
        InsightReport with pdf_url and json_summary populated
    """
    # Gather meal count
    meal_count = (
        await db.execute(
            select(func.count())
            .select_from(Meal)
            .where(
                Meal.user_id == user_id,
                func.date(Meal.timestamp) >= data.date_range_start,
                func.date(Meal.timestamp) <= data.date_range_end,
            )
        )
    ).scalar_one()

    # Gather symptom stats
    symptom_stats = (
        await db.execute(
            select(
                SymptomScore.symptom_type,
                func.count().label("count"),
                func.avg(SymptomScore.vas_score).label("avg_score"),
                func.max(SymptomScore.vas_score).label("peak_score"),
            )
            .where(
                SymptomScore.user_id == user_id,
                func.date(SymptomScore.timestamp) >= data.date_range_start,
                func.date(SymptomScore.timestamp) <= data.date_range_end,
            )
            .group_by(SymptomScore.symptom_type)
        )
    ).all()

    # Gather active triggers
    triggers = (
        await db.execute(
            select(TriggerPrediction)
            .where(TriggerPrediction.user_id == user_id)
            .order_by(TriggerPrediction.confidence_score.desc())
            .limit(10)
        )
    ).scalars().all()

    # Build JSON summary
    json_summary = {
        "period": {
            "start": data.date_range_start.isoformat(),
            "end": data.date_range_end.isoformat(),
        },
        "meals_logged": meal_count,
        "symptom_summary": [
            {
                "type": str(row.symptom_type),
                "count": row.count,
                "avg_severity": round(float(row.avg_score), 1) if row.avg_score else 0,
                "peak_severity": int(row.peak_score) if row.peak_score else 0,
            }
            for row in symptom_stats
        ],
        "top_triggers": [
            {
                "component": str(t.component_type),
                "confidence": int(t.confidence_score),
                "status": str(t.status),
                "evidence_count": t.evidence_count,
            }
            for t in triggers
        ],
    }

    # Create report record first (to get report_id)
    report = InsightReport(
        user_id=user_id,
        report_type=data.report_type,
        date_range_start=data.date_range_start,
        date_range_end=data.date_range_end,
        json_summary=json_summary,
    )
    db.add(report)
    await db.flush()

    # Generate PDF
    pdf_bytes = generate_pdf(
        json_summary=json_summary,
        report_type=str(data.report_type).replace("ReportType.", ""),
        user_id=user_id,
        report_id=report.id,
    )

    # Save PDF to disk
    pdf_path = _save_pdf_to_disk(pdf_bytes, report.id)

    # Update report with PDF path
    report.pdf_url = pdf_path
    db.add(report)
    await db.flush()

    return report


async def get_report(
    db: AsyncSession,
    report_id: uuid.UUID,
    user_id: uuid.UUID,
) -> InsightReport | None:
    """Retrieve a specific report by ID, ensuring user ownership.

    Args:
        db: Database session
        report_id: Report ID to retrieve
        user_id: User ID (for access control)

    Returns:
        InsightReport if found and owned by user, None otherwise
    """
    result = await db.execute(
        select(InsightReport).where(
            InsightReport.id == report_id,
            InsightReport.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_reports(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[InsightReport], int]:
    """List reports for a user with pagination.

    Args:
        db: Database session
        user_id: User ID
        limit: Maximum number of reports to return
        offset: Number of reports to skip

    Returns:
        Tuple of (list of reports, total count)
    """
    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(InsightReport).where(InsightReport.user_id == user_id)
    )
    total_count = count_result.scalar_one()

    # Get paginated results, ordered by creation date descending
    result = await db.execute(
        select(InsightReport)
        .where(InsightReport.user_id == user_id)
        .order_by(InsightReport.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    reports = result.scalars().all()

    return reports, total_count
