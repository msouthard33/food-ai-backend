"""Report generation service."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchem import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meal import Meal,
from app.models.trigger import TriggerPrediction
from app.models.report import InsightReport
from app.schemas.report import InsightReportOut

from app.services.trigger_service import get_user_triggers
