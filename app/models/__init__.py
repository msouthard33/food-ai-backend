"""ORM models — import all modules so Base.metadata registers every table."""

from app.models.enums import (  # noqa: F401
    ComponentSource,
    ComponentType,
    ConditionType,
    MealType,
    ProcessingStatus,
    ReportType,
    Severity,
    ShareMethod,
    SymptomType,
    TriggerStatus,
)
from app.models.food import ComponentDefinition, FoodComponentDetail, FoodEntry  # noqa: F401
from app.models.meal import AIConversation, Meal, MealItem, MealItemComponent  # noqa: F401
from app.models.report import InsightReport, ReportShare  # noqa: F401
from app.models.symptom import DailyCheckin, SymptomScore  # noqa: F401
from app.models.trigger import CorrelationEvent, TriggerPrediction  # noqa: F401
from app.models.user import User, UserCondition, UserKnownAllergen, UserSettings  # noqa: F401
