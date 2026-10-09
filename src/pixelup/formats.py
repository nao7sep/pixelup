"""The format version of every store PixelUp versions, one integer per format
(store-recovery-conventions).

Saved settings (``config.json``) and the records history (``records.sqlite3``) are
durable: they survive PixelUp updates (developer decision), so a change to either
format converts existing data with a small targeted conversion rather than
resetting it. The output sidecar is a published format. ``window.ini`` holds only
disposable window state and carries no marker.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

CONFIG_FORMAT_VERSION = 1  # config.json
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
    """Open a store, creating its schema and marker together when it is new.

    The format is checked once here, at the boundary that owns the store: no
    supported scenario changes it while a connection is open, so later operations
    do not check it again (store-recovery-conventions). A database with no marker
    and no schema objects is new, whether SQLite just created the file or an
    interrupted creation left it empty; anything else is admitted by its marker, so
    a store without one takes its unreadable branch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5, isolation_level=None, check_same_thread=False)
    try:
        connection.execute("BEGIN IMMEDIATE")
        stored = connection.execute("PRAGMA user_version").fetchone()[0]
        created = (
            stored == 0
            and connection.execute("SELECT COUNT(*) FROM sqlite_schema").fetchone()[0] == 0
        )
        if created:
            for statement in schema.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {int(supported)}")
        else:
            check_sqlite_format(connection, path, supported)
        connection.commit()
        if created:
            # Set once, outside the transaction; an existing store keeps its own mode
            # and is never changed before its marker admits it.
            connection.execute("PRAGMA journal_mode = WAL")
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
