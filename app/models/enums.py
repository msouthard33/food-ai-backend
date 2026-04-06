"""Database enum types mirroring the PostgreSQL schema."""

import enum


class MealType(str, enum.Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    BEVERAGE = "beverage"


class SymptomType(str, enum.Enum):
    BLOATING = "bloating"
    PAIN = "pain"
    NAUSEA = "nausea"
    BRAIN_FOG = "brain_fog"
    FATIGUE = "fatigue"
    SKIN_REACTION = "skin_reaction"
    BOWEL_CHANGES = "bowel_changes"
    HEARTBURN = "heartburn"
    HEADACHE = "headache"
    JOINT_PAIN = "joint_pain"
    RESPIRATORY = "respiratory"
    OTHER = "other"


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ConditionType(str, enum.Enum):
    IBS = "ibs"
    SIBO = "sibo"
    HISTAMINE_INTOLERANCE = "histamine_intolerance"
    CELIAC = "celiac"
    MCAS = "mcas"
    NICKEL_ALLERGY = "nickel_allergy"
    FODMAP_SENSITIVITY = "fodmap_sensitivity"
    LEAKY_GUT = "leaky_gut"
    DYSBIOSIS = "dysbiosis"
    CROHNS = "crohns"
    ULCERATIVE_COLITIS = "ulcerative_colitis"
    OTHER = "other"


class ComponentType(str, enum.Enum):
    GLUTEN = "gluten"
    MILK_DAIRY = "milk_dairy"
    SOY = "soy"
    EGGS = "eggs"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    FISH = "fish"
    SHELLFISH = "shellfish"
    HISTAMINES = "histamines"
    SALICYLATES = "salicylates"
    OXALATES = "oxalates"
    AMINES = "amines"
    SULFITES = "sulfites"
    ADDITIVES = "additives"
    FODMAP = "fodmap"
    LACTOSE = "lactose"
    FRUCTOSE = "fructose"
    LECTINS = "lectins"
    WHEAT = "wheat"
    SESAME = "sesame"
    NIGHTSHADES = "nightshades"
    ALCOHOL = "alcohol"
    OTHER = "other"


class ComponentSource(str, enum.Enum):
    # Values match the DB component_source_enum
    DATABASE = "database"
    AI_INFERRED = "ai_inferred"
    USER_REPORTED = "user_reported"
    RESEARCH_BASED = "research_based"


class ReportType(str, enum.Enum):
    # Values match the DB report_type_enum
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"
    CLINICIAN = "clinician"


class Severity(str, enum.Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ShareMethod(str, enum.Enum):
    # Values match the DB share_method_enum
    LINK = "link"
    PDF = "pdf"
    IN_APP = "in_app"


class TriggerStatus(str, enum.Enum):
    # Values match the DB trigger_status_enum
    SUSPECT = "suspect"
    PROBABLE = "probable"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
