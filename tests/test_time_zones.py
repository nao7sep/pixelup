from __future__ import annotations

import pytest
from PySide6.QtCore import QDateTime, QLocale, QTimeZone

from pixelup import time_zones
from pixelup.records_window import format_record_time
from pixelup.time_zones import (
    SYSTEM_ZONE,
    display_zone,
    is_zone_preference,
    resolve_zone,
    set_display_zone,
    system_zone_id,
    zone_ids,
)

STORED = "2026-01-01T00:00:00.000Z"


def test_the_platforms_zones_are_offered_in_order() -> None:
    zones = zone_ids()
    assert "Asia/Tokyo" in zones
    assert list(zones) == sorted(zones)
    assert is_zone_preference(SYSTEM_ZONE)
    assert is_zone_preference("Asia/Tokyo")
    assert not is_zone_preference("Mars/Olympus_Mons")
    assert not is_zone_preference(9)


def test_system_follows_the_computer_and_falls_back_to_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_zone(SYSTEM_ZONE).id() == QTimeZone.systemTimeZone().id()
    monkeypatch.setattr(time_zones.QTimeZone, "systemTimeZone", lambda: QTimeZone())
    assert system_zone_id() == "UTC"
    assert resolve_zone(SYSTEM_ZONE).offsetFromUtc(QDateTime.currentDateTimeUtc()) == 0


def test_an_unknown_zone_reads_as_utc() -> None:
    assert resolve_zone("Mars/Olympus_Mons").offsetFromUtc(QDateTime.currentDateTimeUtc()) == 0


@pytest.mark.parametrize(
    ("zone", "shown"),
    [("Asia/Tokyo", "1/1/26 9:00:00\u202fAM"), ("America/New_York", "12/31/25 7:00:00\u202fPM")],
)
def test_records_show_times_in_the_display_zone(zone: str, shown: str) -> None:
    set_display_zone(zone)
    assert bytes(display_zone().id().data()).decode() == zone
    assert format_record_time(STORED, QLocale("en_US")) == shown
