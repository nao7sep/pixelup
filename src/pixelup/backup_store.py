"""The write-through data-backup store (data-backup-conventions).

It owns one add-only SQLite file, ``backups.sqlite3``, directly under PixelUp's
storage root (``PIXELUP_HOME`` or ``~/.pixelup``, resolved in one place by
:func:`resolve_state_dir` — never a hardcoded path). Every managed *text* save
records the exact bytes it just wrote here, strictly AFTER its atomic rename
lands, so the history is always as current as the last save. There is no startup
scan, no periodic pass, no restore path.

SQLite binding: Python's built-in ``sqlite3`` module — no native rebuild, no
extra dependency, and synchronous exactly like the record-after-rename hook
wants. A ``BLOB`` round-trips through ``sqlite3.Binary``/``bytes`` byte-for-byte,
so CR/LF, a BOM, and non-UTF-8 bytes are stored verbatim.

Two absolute musts drive every line below (they are not best-effort aspirations):

- It never breaks a save and never crashes the app. The save has already
  succeeded — the file is on disk before :func:`record` is called — so any
  failure here (the DB is locked, the disk is full, an insert throws) is caught,
  logged once at ``warn``, and swallowed. A lost record self-heals on the next
  save of that file, whose content will differ from the last recorded row.
- It logs only failures. A successful record logs NOTHING; a line per save would
  flood the log.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from pixelup.config import resolve_state_dir
from pixelup.session_log import log
from pixelup.timestamps import utc_now_iso_ms

STORE_FILE_NAME = "backups.sqlite3"

# The one add-only table. `content` is a BLOB of the exact bytes written — never
# decoded text, so CR/LF, a BOM, and non-UTF-8 bytes are stored byte-identically.
# `written_at_utc` is the serialized ISO-8601-ms form (2026-07-06T04:05:12.345Z),
# a data value — NEVER the yyyymmdd-hhmmss-fff-utc filename stamp. The (path, id)
# index serves the latest-row-per-path dedup lookup.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS backups (
  id             INTEGER PRIMARY KEY,
  path           TEXT NOT NULL,
  content        BLOB NOT NULL,
  content_sha256 TEXT NOT NULL,
  byte_size      INTEGER NOT NULL,
  written_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backups_path_id ON backups (path, id);
"""

# Module-level singleton, resolved once. A `None` connection means recording is
# disabled for this session because the store could not be opened — a single warn
# was already logged; every later `record` becomes a no-op rather than retrying
# (and re-logging) a broken open on every save.
_connection: sqlite3.Connection | None = None
_initialized = False


def _store_file() -> Path:
    """The store file under the resolved storage root. Computed lazily (not frozen
    into a module constant at import time) so ``PIXELUP_HOME`` is read after the
    environment is set, per the storage-path convention's caution against
    import-time resolution."""
    return resolve_state_dir() / STORE_FILE_NAME


def _ensure_open() -> sqlite3.Connection | None:
    """Open and initialize the store once (create the table if absent, switch on
    WAL). Best-effort: on any failure it logs ONE warn, leaves recording disabled
    for the session, and never raises. WAL is what lets the tolerated two-instance
    case (two PixelUp windows writing at once) serialize safely without a
    cross-process lock.
    """
    global _connection, _initialized
    if _initialized:
        return _connection
    _initialized = True
    file = _store_file()
    try:
        # not recorded: backups.sqlite3 is the store itself — binary, and written
        # by this backup layer, not through the managed-text atomic-write path — so
        # it never records itself. No recursion, no special case (data-backup
        # conventions: "A binary store, excluded from itself").
        # resolve_state_dir() already created the root; be defensive anyway in case
        # the store is the first thing written on a fresh root.
        file.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the record hook may fire from a worker thread
        # (a job's save), and the connection is only ever touched from record(),
        # which serializes its own access. WAL + busy_timeout handle cross-process.
        opened = sqlite3.connect(file, check_same_thread=False)
        opened.execute("PRAGMA journal_mode = WAL")
        # busy_timeout: under the tolerated two-instance case, a contended write
        # waits up to this long for SQLite's write lock instead of immediately
        # failing with SQLITE_BUSY and dropping that record.
        opened.execute("PRAGMA busy_timeout = 5000")
        opened.executescript(_SCHEMA)
        opened.commit()
        _connection = opened
    except Exception as exc:  # noqa: BLE001 - best-effort: log once and disable, never crash.
        log.warning(
            "backup_store.open_failed",
            file=str(file),
            reason=str(exc),
        )
        _connection = None
    return _connection


def _sha256(data: bytes) -> str:
    """SHA-256 of the exact bytes, lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def record(absolute_path: Path, data: bytes) -> None:
    """Record one managed-text write: ``absolute_path`` is the FULL absolute path of
    the file as written; ``data`` is the exact raw bytes just written (the caller
    already holds them — never re-read the file).

    Dedup by content hash per path: the new content's SHA-256 is compared against
    the latest row for the same ``path``, and the insert is SKIPPED when they are
    equal. This collapses consecutive identical saves (an autosave with no real
    change writes no row) while still recording every genuinely distinct version —
    including a revert, whose content differs from the immediately preceding row.

    Best-effort and silent on success; any failure is caught, logged once at
    ``warn`` (file + reason), and swallowed. It never raises, never crashes the
    app, and never breaks the save.
    """
    store = _ensure_open()
    if store is None:
        return  # open failed earlier; disabled for the session (already warned once)
    path_text = str(absolute_path)
    try:
        digest = _sha256(data)
        row = store.execute(
            "SELECT content_sha256 FROM backups WHERE path = ? ORDER BY id DESC LIMIT 1",
            (path_text,),
        ).fetchone()
        if row is not None and row[0] == digest:
            return  # unchanged since the last recorded version — dedup skip

        store.execute(
            "INSERT INTO backups (path, content, content_sha256, byte_size, written_at_utc)"
            " VALUES (?, ?, ?, ?, ?)",
            (path_text, sqlite3.Binary(data), digest, len(data), utc_now_iso_ms()),
        )
        store.commit()
    except Exception as exc:  # noqa: BLE001 - best-effort: log once and swallow, never crash.
        log.warning(
            "backup_store.record_failed",
            file=path_text,
            reason=str(exc),
        )


def close_backup_store() -> None:
    """Close the store (best-effort). For tests that need to release the file handle
    between throwaway roots; the app itself lets the process exit close it. Resets
    the singleton so the next :func:`record` re-opens against the current
    ``PIXELUP_HOME``.
    """
    global _connection, _initialized
    try:
        if _connection is not None:
            _connection.close()
    except Exception:  # noqa: BLE001 - best-effort: a close failure on teardown is harmless.
        pass
    _connection = None
    _initialized = False
