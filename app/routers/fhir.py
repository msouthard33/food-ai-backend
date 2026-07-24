"""FHIR R4 export endpoint — Pillar 4 clinical trust layer skeleton."""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.dependencies import get_current_user
from app.models.meal import Meal
from app.models.symptom import SymptomScore
from app.models.user import User

router = APIRouter(prefix="/api/v1/fhir", tags=["insights"])


@router.get(
    "/export",
    summary="Export user data as a FHIR R4 Bundle",
)
async def fhir_export(
    lookback_days: int = Query(30, ge=1, le=365, description="Days of history to export"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Return a FHIR R4 Bundle containing NutritionIntake and Observation resources.

    This is a skeleton implementation (W2-3c scope). Full clinical detail
    (coded values, proper reference resolution) is deferred to W2-5.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch meals
    meal_result = await db.execute(
        select(Meal)
        .where(and_(Meal.user_id == user.id, Meal.timestamp >= cutoff))
        .options(selectinload(Meal.items))
        .order_by(Meal.timestamp.desc())
        .limit(100)
    )
    meals = list(meal_result.scalars().unique().all())

    # Fetch symptoms
    symptom_result = await db.execute(
        select(SymptomScore)
        .where(and_(SymptomScore.user_id == user.id, SymptomScore.timestamp >= cutoff))
        .order_by(SymptomScore.timestamp.desc())
        .limit(100)
    )
    symptoms = list(symptom_result.scalars().all())

    # Build FHIR R4 Bundle
    entries = []

    # NutritionIntake resources (one per meal)
    for meal in meals:
        food_items = [
            {"itemCodeableConcept": {"text": item.name}} for item in meal.items
        ]
        resource = {
            "resourceType": "NutritionIntake",
            "id": str(meal.id),
            "status": "completed",
            "subject": {"reference": f"Patient/{user.id}"},
            "occurrenceDateTime": meal.timestamp.isoformat(),
            "consumedItem": food_items if food_items else [{"itemCodeableConcept": {"text": meal.raw_description or "Unknown"}}],
            "note": [{"text": meal.raw_description}] if meal.raw_description else [],
        }
        entries.append({
            "fullUrl": f"urn:uuid:{meal.id}",
            "resource": resource,
        })

    # Observation resources (one per symptom log)
    for symptom in symptoms:
        resource = {
            "resourceType": "Observation",
            "id": str(symptom.id),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://foodai.app/fhir/symptom-type",
                        "code": str(symptom.symptom_type),
                        "display": str(symptom.symptom_type).replace("_", " ").title(),
                    }
                ],
                "text": str(symptom.symptom_type).replace("_", " ").title(),
            },
            "subject": {"reference": f"Patient/{user.id}"},
            "effectiveDateTime": symptom.timestamp.isoformat(),
            "valueQuantity": {
                "value": int(symptom.vas_score),
                "unit": "VAS score",
                "system": "http://foodai.app/fhir/vas",
                "code": "vas-0-100",
            },
        }
        if symptom.notes:
            resource["note"] = [{"text": symptom.notes}]

        entries.append({
            "fullUrl": f"urn:uuid:{symptom.id}",
            "resource": resource,
        })

    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "entry": entries,
    }

    return JSONResponse(
        content=bundle,
        media_type="application/fhir+json",
    )
