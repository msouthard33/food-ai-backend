"""Medication co-log correlation helpers — MCAS covariate for trigger analysis.

Medication timing is treated as a **covariate / confounder** in correlation scoring,
not as an independent trigger. Clinical rationale: when a symptom episode was
medicated (e.g. an antihistamine was taken in the surrounding window), the observed
severity is pharmacologically blunted, so that episode is *weaker* evidence that a
preceding food is a clean trigger. We surface this confound and modestly discount
the score rather than hiding it — an honest MCAS differentiator, not a causal model.
"""

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.medication import MedicationLog

# A medicated symptom episode counts as half-strength evidence for a food trigger.
# combined_score = trigger_score * (1 - MEDICATION_CONFOUND_WEIGHT * medicated_fraction)
MEDICATION_CONFOUND_WEIGHT = 0.5


async def get_medicated_symptom_map(
    db: AsyncSession,
    user_id: uuid.UUID,
    symptom_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, list[MedicationLog]]:
    """Return ``{symptom_log_id: [MedicationLog, ...]}`` for episodes that were medicated.

    Symptom ids with no co-logged medication are omitted from the map, so a simple
    ``symptom_id in mapping`` test answers "was this episode medicated?".
    """
    ids = [sid for sid in symptom_ids]
    if not ids:
        return {}

    result = await db.execute(
        select(MedicationLog).where(
            MedicationLog.user_id == user_id,
            MedicationLog.symptom_log_id.in_(ids),
        )
    )
    mapping: dict[uuid.UUID, list[MedicationLog]] = {}
    for med in result.scalars().all():
        mapping.setdefault(med.symptom_log_id, []).append(med)
    return mapping


def medication_adjusted_score(
    raw_score: float,
    n_episodes: int,
    n_medicated_episodes: int,
) -> float:
    """Discount a raw correlation score by the share of its episodes that were medicated.

    Returns the adjusted score on the same 0–100 scale. With zero medicated episodes
    the score is unchanged; if every supporting episode was medicated the score is
    reduced by ``MEDICATION_CONFOUND_WEIGHT`` (default 50%).
    """
    if n_episodes <= 0 or n_medicated_episodes <= 0:
        return raw_score
    medicated_fraction = min(n_medicated_episodes / n_episodes, 1.0)
    return raw_score * (1.0 - MEDICATION_CONFOUND_WEIGHT * medicated_fraction)
