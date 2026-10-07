"""The format version of every store PixelUp writes, one integer per format
(store-recovery-conventions).

Every format is at 1 and no durable-data constant exists: PixelUp's data is not
durable yet, so a format change edits its format in place.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from filelock import FileLock

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


def check_sqlite_format(connection: sqlite3.Connection, path: Path, supported: int) -> None:
    """Admit an existing database using its explicit positive version."""
    stored = connection.execute("PRAGMA user_version").fetchone()[0]
    if stored < 1:
        raise ValueError(f"{path} has no format version")
    if stored > supported:
        raise NewerFormatError(path, stored)


def open_sqlite_store(path: Path, supported: int, schema: str) -> sqlite3.Connection:
    """Create schema and marker together, using exclusive creation as fresh provenance.

    The initialization lock precedes SQLite's writer lock. Ordinary operations take only
    the SQLite transaction; none acquires the initialization lock while holding one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_name(f"{path.stem}-initialize.lock")), timeout=5):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            fresh = False
        else:
            os.close(descriptor)
            fresh = True
        connection = sqlite3.connect(path, timeout=5, isolation_level=None, check_same_thread=False)
        try:
            if fresh:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            if fresh:
                stored = connection.execute("PRAGMA user_version").fetchone()[0]
                if stored != 0:
                    check_sqlite_format(connection, path, supported)
                else:
                    for statement in schema.split(";"):
                        if statement.strip():
                            connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {int(supported)}")
            else:
                check_sqlite_format(connection, path, supported)
            connection.commit()
            connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except BaseException as exc:
            rollback_sqlite(connection, exc)
            try:
                connection.close()
            except Exception as cleanup:
                exc.add_note(f"SQLite close failed: {cleanup!r}")
            raise


def rollback_sqlite(connection: sqlite3.Connection, primary: BaseException) -> None:
    """Attempt cleanup without replacing the operation's diagnostic failure."""
    try:
        connection.rollback()
    except Exception as cleanup:
        primary.add_note(f"SQLite rollback failed: {cleanup!r}")
