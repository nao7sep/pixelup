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
    """A stored marker as a version.

    Raises ``ValueError`` for a missing marker or one that is not a positive
    integer: either makes the store unreadable, and nothing infers a version from
    the store's shape.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("the format version is missing or not a positive integer")
    return value


def check_sqlite_format(connection: sqlite3.Connection, path: Path, supported: int) -> bool:
    """Check a database's version before anything writes to it; return whether it
    is brand new, to be stamped at creation.

    SQLite's ``user_version`` starts at 0, so 0 marks a new database only while it
    holds no schema; on a database with tables it is a missing marker, and the
    database is unreadable (``ValueError``). A newer version raises
    ``NewerFormatError``.
    """
    stored = connection.execute("PRAGMA user_version").fetchone()[0]
    if stored == 0:
        if connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is not None:
            raise ValueError(f"{path} has no format version")
        return True
    if stored > supported:
        raise NewerFormatError(path, stored)
    return False


def mark_sqlite_format(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(f"PRAGMA user_version = {int(version)}")
