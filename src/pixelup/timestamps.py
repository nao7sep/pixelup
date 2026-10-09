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


def utc_stamp(moment: datetime) -> str:
    """Format a datetime as PixelUp's compact filename-safe UTC stamp.

    Shape: ``yyyymmdd-hhmmss-utc`` (e.g. ``20260610-031542-utc``) — no colons, an
    explicit ``utc`` tag, so the stamp is safe to embed in a filename and sorts
    lexicographically in chronological order. Seconds are enough: it names the
    records fallback log and a quarantined ``.invalid`` config, each created at most
    once per launch by the one supported PixelUp process (timestamp-conventions).
    Aware inputs in other time zones are converted to UTC; naive inputs are
    interpreted as UTC (PixelUp's internal convention) rather than local time.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y%m%d-%H%M%S") + "-utc"


def utc_now_stamp() -> str:
    """Current time in the compact filename-safe UTC stamp format (see ``utc_stamp``)."""
    return utc_stamp(datetime.now(UTC))
