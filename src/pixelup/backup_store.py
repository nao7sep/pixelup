"""The data-backup history (data-backup-conventions).

It owns one SQLite file, ``backups.sqlite3``, directly under PixelUp's storage root
(``PIXELUP_DATA_DIR`` or ``~/.pixelup``, resolved in one place by
:func:`resolve_state_dir`). It keeps the last version of each protected file saved
in each session (one process launch). PixelUp protects only ``config.json``, its one
file the user creates through the app: models are derived and downloaded again,
upscaled images and their sidecars are output the user owns, ``window.ini`` is
state, and records are records. There is no startup scan, no capture at exit beyond
draining pending writes, and no restore path; recovery is manual.

SQLite binding: Python's built-in ``sqlite3`` module. A ``BLOB`` round-trips through
``sqlite3.Binary``/``bytes`` byte-for-byte, so CR/LF, a BOM and non-UTF-8 bytes are
stored verbatim. Each file is one value; a file beyond SQLite's value limit fails
to record with a diagnostic.

Two absolute musts drive every line below:

- It never delays, breaks or fails a save. :func:`record` only hands the exact bytes
  just written to one serial recorder thread and returns; the thread applies writes
  in save order, keeping only the newest pending write per path. Any failure there
  (the DB is locked, the disk is full, an insert throws) is logged once at ``warn``
  and recording stays disabled for this session.
- It logs only failures. A successful record logs nothing.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from pixelup.config import resolve_state_dir
from pixelup.formats import BACKUPS_FORMAT_VERSION, open_sqlite_store
from pixelup.session_log import current_session, log
from pixelup.timestamps import utc_now_iso_ms

STORE_FILE_NAME = "backups.sqlite3"

# The one table. `content` is a BLOB of the exact bytes written — never decoded
# text. `session_id` is the launch's records session. `written_at_utc` is the
# serialized ISO-8601-ms form (2026-07-06T04:05:12.345Z) of the row's latest save,
# a data value — NEVER the yyyymmdd-hhmmss-utc filename stamp. The unique
# (path, session_id) key owns the per-session row; the (path, id) index serves the
# latest-row lookup.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS backups (
  id             INTEGER PRIMARY KEY,
  path           TEXT NOT NULL,
  content        BLOB NOT NULL,
  content_sha256 TEXT NOT NULL,
  byte_size      INTEGER NOT NULL,
  written_at_utc TEXT NOT NULL,
  session_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_backups_path_id ON backups (path, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_backups_path_session ON backups (path, session_id);
"""

def _store_file() -> Path:
    """The store file under the resolved storage root. Computed lazily (not frozen
    into a module constant at import time) so ``PIXELUP_DATA_DIR`` is read after the
    environment is set, per the storage-path convention's caution against
    import-time resolution."""
    return resolve_state_dir() / STORE_FILE_NAME


def _open_store() -> sqlite3.Connection:
    # not recorded: backups.sqlite3 is the store itself, written by this layer and
    # not through the managed-text path, so it never records itself.
    return open_sqlite_store(_store_file(), BACKUPS_FORMAT_VERSION, _SCHEMA)


def _sha256(data: bytes) -> str:
    """SHA-256 of the exact bytes, lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def _write(store: sqlite3.Connection, session: str, path: str, data: bytes) -> None:
    """Keep ``data`` as this session's version of ``path``.

    The session's first save of a path inserts its row unless the content equals the
    latest row of an earlier session; later saves in the session replace the row.
    """
    digest = _sha256(data)
    store.execute("BEGIN IMMEDIATE")
    try:
        current = store.execute(
            "SELECT content_sha256 FROM backups WHERE path = ? AND session_id = ?",
            (path, session),
        ).fetchone()
        if current is None:
            latest = store.execute(
                "SELECT content_sha256 FROM backups WHERE path = ? ORDER BY id DESC LIMIT 1",
                (path,),
            ).fetchone()
            if latest is None or latest[0] != digest:
                store.execute(
                    "INSERT INTO backups (session_id, path, content, content_sha256,"
                    " byte_size, written_at_utc) VALUES (?, ?, ?, ?, ?, ?)",
                    (session, path, sqlite3.Binary(data), digest, len(data), utc_now_iso_ms()),
                )
        elif current[0] != digest:
            store.execute(
                "UPDATE backups SET content = ?, content_sha256 = ?, byte_size = ?,"
                " written_at_utc = ? WHERE path = ? AND session_id = ?",
                (sqlite3.Binary(data), digest, len(data), utc_now_iso_ms(), path, session),
            )
        store.commit()
    except BaseException:
        store.rollback()
        raise


class _Recorder:
    """The one serial owner of backup writes, on its own daemon thread.

    Pending writes wait in save order with only the newest per path, so an earlier
    save never replaces a later one and the backlog stays bounded by the number of
    protected paths. The thread is a daemon: an OS exit never waits for it.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: dict[str, bytes] = {}
        self._writing = False
        self._thread: threading.Thread | None = None
        self._connection: sqlite3.Connection | None = None
        self._session: str | None = None
        self._disabled = False

    def submit(self, path: str, data: bytes) -> None:
        failure: Exception | None = None
        with self._condition:
            if self._disabled:
                return
            try:
                self._pending.pop(path, None)
                self._pending[path] = data
                if self._thread is None:
                    thread = threading.Thread(target=self._run, name="pixelup-backups", daemon=True)
                    thread.start()
                    # Only a started thread is owned and eligible for close() to join.
                    self._thread = thread
            except Exception as exc:  # noqa: BLE001 - optional work cannot deny a landed save.
                failure = exc
                self._disabled = True
                self._pending.clear()
            self._condition.notify_all()
        if failure is not None:
            # Diagnose outside the condition, after abandoning unstarted work.
            # Diagnostic admission is best-effort too: it cannot undo primary success.
            try:
                log.warning(
                    "backup_store.submit_failed",
                    file=path,
                    exc_info=(type(failure), failure, failure.__traceback__),
                )
            except Exception:  # noqa: BLE001 - the warning was attempted; keep the save successful.
                pass

    def drain(self) -> None:
        """Wait for pending writes to land; the quit's bound limits the wait."""
        with self._condition:
            self._condition.wait_for(lambda: not self._pending and not self._writing)

    def close(self) -> None:
        """Drain, close the store and end the thread, leaving the recorder ready to
        open the store again at its next write (tests and teardown)."""
        self.drain()
        with self._condition:
            self._disabled = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - a close failure on teardown is harmless.
                pass
        with self._condition:
            self._thread = None
            self._session = None
            self._disabled = False

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._disabled)
                if self._disabled:
                    self._pending.clear()
                    self._condition.notify_all()
                    return
                path = next(iter(self._pending))
                data = self._pending.pop(path)
                self._writing = True
            try:
                self._apply(path, data)
            finally:
                with self._condition:
                    self._writing = False
                    self._condition.notify_all()

    def _apply(self, path: str, data: bytes) -> None:
        if self._connection is None:
            try:
                self._connection = _open_store()
                self._session = current_session() or utc_now_iso_ms()
            except Exception as exc:  # noqa: BLE001 - best-effort: log once and disable.
                self._disable()
                # The failed resolver is part of this exception. Reuse the saved
                # file's accepted identity rather than resolving the root again.
                try:
                    log.warning(
                        "backup_store.open_failed",
                        file=path,
                        store=STORE_FILE_NAME,
                        reason=str(exc),
                        exc_info=True,
                    )
                except Exception:  # noqa: BLE001 - the diagnostic was attempted; recording is off.
                    pass
                return
        try:
            _write(self._connection, self._session, path, data)
        except Exception as exc:  # noqa: BLE001 - best-effort: log once and disable.
            self._disable()
            connection, self._connection = self._connection, None
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - closing a failed store is best-effort.
                pass
            try:
                log.warning("backup_store.record_failed", file=path, reason=str(exc), exc_info=True)
            except Exception:  # noqa: BLE001 - the diagnostic was attempted; recording is off.
                pass

    def _disable(self) -> None:
        with self._condition:
            self._disabled = True
            self._pending.clear()
            self._condition.notify_all()


_recorder = _Recorder()


def record(absolute_path: Path, data: bytes) -> None:
    """Hand one managed-text write to the recorder and return at once.

    ``absolute_path`` is the FULL absolute path of the file as written; ``data`` is
    the exact raw bytes just written (the caller already holds them — never re-read
    the file). It never raises and never waits on the store.
    """
    _recorder.submit(str(absolute_path), data)


def drain_backups() -> None:
    """Wait for pending writes, for an ordinary quit; an OS session end skips this."""
    _recorder.drain()


def close_backup_store() -> None:
    """Drain and close the store (best-effort). For tests that need to release the
    file handle between throwaway roots; the app itself lets the process exit close
    it. The next :func:`record` re-opens the store against the current
    ``PIXELUP_DATA_DIR`` with a fresh session.
    """
    _recorder.close()
