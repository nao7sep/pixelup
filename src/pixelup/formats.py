"""The format version of every store PixelUp writes, one integer per format
(store-recovery-conventions).

Every format is at 1 and no durable-data constant exists: PixelUp's data is not
durable yet, so a format change edits its format in place.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

CONFIG_FORMAT_VERSION = 1  # config.json
WINDOW_STATE_FORMAT_VERSION = 1  # window.ini
RECORDS_FORMAT_VERSION = 1  # records.sqlite3
BACKUPS_FORMAT_VERSION = 1  # backups.sqlite3
SIDECAR_FORMAT_VERSION = 1  # the .json sidecar written beside an output image


class NewerFormatError(Exception):
    """A store written by a newer PixelUp: intact data this build cannot read,
    which is left exactly in place."""

    def __init__(self, path: Path, version: int) -> None:
        super().__init__(f"{path} has format version {version}, newer than this build reads")
        self.path = path
        self.version = version


def format_version(value: object) -> int:
    """A stored marker as a version; an absent marker reads as 1.

    Raises ``ValueError`` for a marker that is not a positive integer, which makes
    the store's shape unreadable.
    """
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("the format version is not a positive integer")
    return value


def check_sqlite_format(connection: sqlite3.Connection, path: Path, supported: int) -> bool:
    """Raise ``NewerFormatError`` when the database is newer than ``supported``.

    Reads only, so it runs before anything writes to the file. Returns whether the
    database still lacks its marker: SQLite's ``user_version`` starts at 0, which
    reads as 1.
    """
    stored = connection.execute("PRAGMA user_version").fetchone()[0]
    if stored > supported:
        raise NewerFormatError(path, stored)
    return stored == 0


def mark_sqlite_format(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(f"PRAGMA user_version = {int(version)}")
