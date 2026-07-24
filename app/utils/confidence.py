"""D9 confidence tier mapping — canonical vocabulary for all AI output."""


def confidence_to_tier_label(confidence: float) -> str:
    """Map a 0.0–1.0 confidence score to the D9 canonical tier label.

    D9 mapping:
      >= 0.85 → "Well-established"
      0.55–0.84 → "Some evidence"
      < 0.55 → "AI estimate"
    """
    if confidence >= 0.85:
        return "Well-established"
    if confidence >= 0.55:
        return "Some evidence"
    return "AI estimate"
