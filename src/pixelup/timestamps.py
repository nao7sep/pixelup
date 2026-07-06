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


def utc_stamp_ms(moment: datetime) -> str:
    """Format a datetime as PixelUp's compact filename-safe UTC stamp.

    Shape: ``yyyymmdd-hhmmss-fff-utc`` (e.g. ``20260610-031542-123-utc``) — no
    colons, millisecond precision, and an explicit ``utc`` tag, so the stamp is
    safe to embed in a filename and sorts lexicographically in chronological
    order. Aware inputs in other time zones are converted to UTC; naive inputs
    are interpreted as UTC (PixelUp's internal convention) rather than local
    time. Used for session-log filenames and the ``.invalid`` quarantine stamp.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    d = moment.astimezone(UTC)
    return d.strftime("%Y%m%d-%H%M%S") + f"-{d.microsecond // 1000:03d}-utc"


def utc_now_stamp_ms() -> str:
    """Current time in the compact filename-safe UTC stamp format (see ``utc_stamp_ms``)."""
    return utc_stamp_ms(datetime.now(UTC))
