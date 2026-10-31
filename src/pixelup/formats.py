"""The identity header of every file PixelUp writes in its own format
(file-identity-conventions), and the check every reader makes of it.

Each file says what it is in ``format`` (owner ``pixelup``, a kind, a version)
and what wrote it in ``writer`` (this package and its release). Readers accept or
refuse a file on ``format`` alone and never read ``writer``.

App-owned data is disposable during development (developer decision), so every
format reads only its current version: a format change uses the new version
directly, with no conversion of earlier data (store-recovery-conventions). The
backup history leaves any other store in place and recording stays off for that
session. The output sidecar is a published format. ``window.ini`` is Qt's own
window-geometry file and carries no header.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup import __version__

OWNER = "pixelup"
WRITER_NAME = "pixelup"
HEADER_KEYS = ("format", "writer")


@dataclass(frozen=True, slots=True)
class Format:
    """One of PixelUp's formats: its kind and the version this build reads and writes."""

    kind: str
    version: int


CONFIG_FORMAT = Format("config", 1)  # config.json
PARAMETERS_FORMAT = Format("parameters", 1)  # parameters.json
RECORDS_FORMAT = Format("records", 1)  # records.sqlite3
BACKUPS_FORMAT = Format("backups", 1)  # backups.sqlite3
SIDECAR_FORMAT = Format("sidecar", 1)  # the .json sidecar written beside an output image


class NewerFormatError(Exception):
    """A file written by a newer PixelUp: intact data this build cannot read,
    which is left exactly in place."""

    def __init__(self, path: Path, version: int) -> None:
        super().__init__(f"{path} has format version {version}, newer than this build reads")
        self.path = path
        self.version = version


@dataclass(frozen=True, slots=True)
class RefusedFile:
    """A file this build will not use: written by a newer PixelUp, or unreadable."""

    path: Path
    newer: bool


class RefusedFilesError(Exception):
    """Files that stop the launch (store-recovery-conventions, What a refused file
    does). Every refused file found is named, and none of them has been changed."""

    def __init__(self, files: tuple[RefusedFile, ...]) -> None:
        super().__init__(", ".join(str(file.path) for file in files))
        self.files = files


def _writer() -> dict[str, str]:
    return {"name": WRITER_NAME, "version": __version__}


def json_text(expected: Format, content: Mapping[str, Any]) -> str:
    """A JSON file of ``expected``: the header as its first two keys, then the
    content with every object's keys sorted."""
    assert not set(HEADER_KEYS) & content.keys(), "content never uses a header key"
    header = {
        "format": {"owner": OWNER, "kind": expected.kind, "version": expected.version},
        "writer": _writer(),
    }
    return json.dumps({**header, **_sorted(dict(content))}, indent=2) + "\n"


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted(item) for item in value]
    return value


def json_content(data: object, expected: Format, path: Path) -> dict[str, Any]:
    """The content of a parsed JSON file of ``expected``, without its header.

    Raises ``ValueError`` when the file is not an object or its ``format`` is
    missing, malformed or names another owner or kind: the file is unreadable, and
    nothing infers its identity from its shape. Raises ``NewerFormatError`` for a
    version above ``expected``. ``writer`` is never read.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    version = _check(data.get("format"), expected, path)
    if version > expected.version:
        raise NewerFormatError(path, version)
    return {key: value for key, value in data.items() if key not in HEADER_KEYS}


def _check(stored: object, expected: Format, path: Path) -> int:
    """The version of a ``format`` naming ``expected``'s owner and kind."""
    if not isinstance(stored, dict):
        raise ValueError(f"{path} has no format header")
    owner, kind, version = stored.get("owner"), stored.get("kind"), stored.get("version")
    if not isinstance(owner, str) or not isinstance(kind, str):
        raise ValueError(f"{path} has a malformed format header")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"{path} has no format version")
    if owner != OWNER or kind != expected.kind:
        raise ValueError(f"{path} is a {owner} {kind} file, not a {OWNER} {expected.kind} file")
    return version


_HEADER_TABLE = "CREATE TABLE file_header (key TEXT PRIMARY KEY, value TEXT NOT NULL)"


def check_sqlite_format(connection: sqlite3.Connection, path: Path, expected: Format) -> None:
    """Admit an existing database by its header: ``PRAGMA user_version`` and the
    ``file_header`` rows ``format.owner`` and ``format.kind``.

    Raises ``ValueError`` when either is missing or names another format, and
    ``NewerFormatError`` for a version above ``expected``. The writer rows are
    never read.
    """
    stored = connection.execute("PRAGMA user_version").fetchone()[0]
    has_table = connection.execute(
        "SELECT COUNT(*) FROM sqlite_schema WHERE type = 'table' AND name = 'file_header'"
    ).fetchone()[0]
    if not has_table:
        raise ValueError(f"{path} has no format header")
    rows = dict(
        connection.execute(
            "SELECT key, value FROM file_header WHERE key IN ('format.owner', 'format.kind')"
        ).fetchall()
    )
    _check(
        {"owner": rows.get("format.owner"), "kind": rows.get("format.kind"), "version": stored},
        expected,
        path,
    )
    if stored > expected.version:
        raise NewerFormatError(path, stored)


def probe_sqlite_store(path: Path, expected: Format) -> None:
    """Check an existing store's header without writing to it.

    A missing or empty file, or a database with no version and no schema, is a
    store not yet created and passes. Otherwise raises as ``check_sqlite_format``
    does, or ``sqlite3.DatabaseError`` for a file that is not a database. Only
    reads are run, so a refused store stays byte-identical.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    connection = sqlite3.connect(path)
    try:
        stored = connection.execute("PRAGMA user_version").fetchone()[0]
        empty = connection.execute("SELECT COUNT(*) FROM sqlite_schema").fetchone()[0] == 0
        if not (stored == 0 and empty):
            check_sqlite_format(connection, path, expected)
    finally:
        connection.close()


def _write_writer_rows(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM file_header WHERE key LIKE 'writer.%'")
    connection.executemany(
        "INSERT INTO file_header (key, value) VALUES (?, ?)",
        [(f"writer.{key}", value) for key, value in _writer().items()],
    )


def open_sqlite_store(path: Path, expected: Format, schema: str) -> sqlite3.Connection:
    """Open a store for writing, creating its schema and header together when it is new.

    The header is checked once here, at the boundary that owns the store: no
    supported scenario changes it while a connection is open, so later operations
    do not check it again (store-recovery-conventions). A database with no version
    and no schema objects is new, whether SQLite just created the file or an
    interrupted creation left it empty; anything else is admitted by its header, so
    a store without one takes its unreadable branch. An admitted store gets its
    writer rows rewritten in the same transaction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    try:
        connection.execute("BEGIN IMMEDIATE")
        stored = connection.execute("PRAGMA user_version").fetchone()[0]
        created = (
            stored == 0
            and connection.execute("SELECT COUNT(*) FROM sqlite_schema").fetchone()[0] == 0
        )
        if created:
            for statement in (_HEADER_TABLE, *schema.split(";")):
                if statement.strip():
                    connection.execute(statement)
            connection.executemany(
                "INSERT INTO file_header (key, value) VALUES (?, ?)",
                [("format.owner", OWNER), ("format.kind", expected.kind)],
            )
            connection.execute(f"PRAGMA user_version = {int(expected.version)}")
        else:
            check_sqlite_format(connection, path, expected)
        _write_writer_rows(connection)
        connection.commit()
        if created:
            # Set once, outside the transaction; an existing store keeps its own mode
            # and is never changed before its header admits it.
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
