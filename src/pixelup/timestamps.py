from __future__ import annotations

from datetime import UTC, datetime


def to_utc_iso_ms(moment: datetime) -> str:
    """Format a datetime as PixelUp's internal timestamp.

    The internal/serialized form is UTC, ISO-8601, with millisecond precision and
    a trailing ``Z`` instead of ``+00:00``. Aware inputs in other time zones are
    converted to UTC; naive inputs are interpreted as UTC (PixelUp's internal
    convention) rather than local time.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now_iso_ms() -> str:
    """Current time in the internal UTC ISO-8601 millisecond format."""
    return to_utc_iso_ms(datetime.now(UTC))
