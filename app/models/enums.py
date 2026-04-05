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
