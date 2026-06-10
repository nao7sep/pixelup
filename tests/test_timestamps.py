from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

from pixelup.timestamps import to_utc_iso_ms, utc_now_iso_ms

CANONICAL = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def test_to_utc_iso_ms_truncates_microseconds_and_uses_z_suffix() -> None:
    moment = datetime(2026, 5, 7, 5, 43, 21, 123456, tzinfo=UTC)
    assert to_utc_iso_ms(moment) == "2026-05-07T05:43:21.123Z"


def test_to_utc_iso_ms_pads_milliseconds_to_three_digits() -> None:
    moment = datetime(2026, 5, 7, 5, 43, 21, tzinfo=UTC)
    assert to_utc_iso_ms(moment) == "2026-05-07T05:43:21.000Z"


def test_to_utc_iso_ms_converts_other_zones_to_utc() -> None:
    jst = timezone(timedelta(hours=9))
    moment = datetime(2026, 5, 7, 5, 43, 21, tzinfo=jst)
    # 05:43:21 JST is the previous day at 20:43:21 UTC.
    assert to_utc_iso_ms(moment) == "2026-05-06T20:43:21.000Z"


def test_to_utc_iso_ms_treats_naive_as_utc() -> None:
    # A naive datetime must be read as UTC, not shifted by the local offset.
    moment = datetime(2026, 5, 7, 5, 43, 21)
    assert to_utc_iso_ms(moment) == "2026-05-07T05:43:21.000Z"


def test_utc_now_iso_ms_matches_canonical_format() -> None:
    assert CANONICAL.fullmatch(utc_now_iso_ms())
