"""Capture-side timestamp validation helpers (Wave 2, Pillar 2).

Shared by the meal- and symptom-create schemas so that every user-supplied
"occurred-at" / "onset-at" timestamp is:

* timezone-aware (naive datetimes are rejected — we cannot correlate an event
  whose wall-clock has no zone), and
* not in the future (beyond a small clock-skew tolerance).

Stored values are normalised to UTC; the client's original zone/offset is kept
separately on the row (``client_timezone``) so the future exposure/lag rewire
can reconstruct local wall-clock time. These helpers are additive and do not
touch the exposure schema or correlation queries.
"""

from datetime import datetime, timedelta, timezone

# Tolerance for benign client/server clock skew when rejecting future times.
FUTURE_TOLERANCE = timedelta(minutes=5)

# Allowed values for the additive ``time_precision`` capture flag.
TIME_PRECISION_VALUES = ("exact", "approximate")


def require_timezone_aware(value: datetime, field_name: str = "timestamp") -> datetime:
    """Reject naive datetimes; normalise aware datetimes to UTC.

    Raises ``ValueError`` (surfaced by Pydantic as a 422) when ``value`` has no
    ``tzinfo``. A timezone-aware value is converted to UTC for storage.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field_name} must be timezone-aware (include a UTC offset or 'Z' suffix)"
        )
    return value.astimezone(timezone.utc)


def reject_future(value: datetime, field_name: str = "timestamp") -> datetime:
    """Reject a timestamp that is in the future beyond the skew tolerance."""
    now = datetime.now(timezone.utc)
    if value > now + FUTURE_TOLERANCE:
        raise ValueError(f"{field_name} cannot be in the future")
    return value


def validate_occurred_at(value: datetime | None, field_name: str = "timestamp") -> datetime | None:
    """Full capture-side validation for an occurred-at / onset-at timestamp.

    Returns ``None`` unchanged (the caller applies a default) so this stays
    backward-compatible with clients that omit the field.
    """
    if value is None:
        return None
    value = require_timezone_aware(value, field_name)
    return reject_future(value, field_name)
