"""The display zone: where the times PixelUp shows are read (timestamp-conventions).

PixelUp stores every instant in UTC and reads no times without a zone, so it has
one zone setting. ``system`` follows the computer's zone at each use and falls back
to UTC when that is unavailable; any other value is an IANA zone the platform lists.
"""

from __future__ import annotations

from functools import cache

from PySide6.QtCore import QByteArray, QTimeZone

SYSTEM_ZONE = "system"

_display_zone = SYSTEM_ZONE


@cache
def zone_ids() -> tuple[str, ...]:
    """The IANA zones this platform offers, in order."""
    zones = QTimeZone.availableTimeZoneIds()
    return tuple(sorted(bytes(zone.data()).decode("ascii") for zone in zones))


def is_zone_preference(value: object) -> bool:
    return value == SYSTEM_ZONE or (isinstance(value, str) and value in zone_ids())


def system_zone_id() -> str:
    """The computer's zone now, or UTC when it is unavailable."""
    zone = QTimeZone.systemTimeZone()
    if not zone.isValid():
        return "UTC"
    return bytes(zone.id().data()).decode("ascii") or "UTC"


def resolve_zone(preference: str) -> QTimeZone:
    zone_id = system_zone_id() if preference == SYSTEM_ZONE else preference
    zone = QTimeZone(QByteArray(zone_id.encode("ascii")))
    return zone if zone.isValid() else QTimeZone(QTimeZone.Initialization.UTC)


def set_display_zone(preference: str) -> None:
    global _display_zone
    _display_zone = preference


def display_zone() -> QTimeZone:
    """The zone the times on screen are shown in, resolved now."""
    return resolve_zone(_display_zone)
