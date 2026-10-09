from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixelup.config import resolve_state_dir
from pixelup.formats import (
    RECORDS_FORMAT_VERSION,
    open_sqlite_store,
    rollback_sqlite,
)
from pixelup.timestamps import to_utc_iso_ms, utc_stamp

LOGGER_NAME = "pixelup"
DEBUG_ENV = "PIXELUP_DEBUG"
RECORDS_FILE_NAME = "records.sqlite3"

# One row per log line. `session` is the launch's start time; `job_id` (a queue
# job) and `operation_id` (a managed-model install) are numbered within their
# session; `fields` is the JSON object of everything else the line carries.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
  id           INTEGER PRIMARY KEY,
  session      TEXT NOT NULL,
  time         TEXT NOT NULL,
  level        TEXT NOT NULL,
  message      TEXT NOT NULL,
  job_id       INTEGER,
  operation_id INTEGER,
  fields       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_session ON logs (session, id);
CREATE INDEX IF NOT EXISTS idx_logs_time ON logs (time, id);
"""
_COLUMNS = frozenset({"time", "level", "message", "job_id", "operation_id"})

# Keys a caller's structured field must never overwrite: the three-part envelope,
# `error`, which is reserved for the attached-exception payload, and the two keys
# a fallback line adds.
_RESERVED_FIELDS = frozenset({"time", "level", "message", "error", "session", "records_error"})

# stdlib level -> the four convention level names (WARNING renders as "warn").
_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}


def _error_object(exc_info: Any) -> dict[str, Any]:
    """Full-fidelity error payload: type, message, and the formatted traceback.

    ``traceback.format_exception`` already walks the ``__cause__`` / ``__context__``
    chain, so the rendered traceback carries the cause chain the convention asks for.
    """
    if not isinstance(exc_info, tuple) or exc_info[1] is None:
        return {}
    exc_type, exc, tb = exc_info
    return {
        "type": (exc_type or type(exc)).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc_type, exc, tb)).rstrip(),
    }


def _dumps(entry: dict[str, Any]) -> str:
    """Serialize a log entry to one JSON line that can never fail.

    The happy path leans on ``default=str`` to coerce stray non-serializable
    values. If anything still defeats serialization — most often a nested dict
    with non-string keys — the entry is coerced field-by-field into a
    guaranteed-serializable form rather than letting the exception bubble up and
    drop the line. A log line must never be lost.
    """
    try:
        return json.dumps(entry, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(_json_safe(entry), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    """Total coercion to a JSON-serializable shape: objects become string-keyed,
    scalars pass through, everything else is stringified. Never raises."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return f"<unrenderable {type(value).__name__}>"


def _entry(record: logging.LogRecord) -> dict[str, Any]:
    """The convention envelope (``time``, ``level``, ``message``) plus the caller's
    structured fields, plus an ``error`` object carrying full exception fidelity
    when one is attached."""
    entry: dict[str, Any] = {
        "time": to_utc_iso_ms(datetime.fromtimestamp(record.created, UTC)),
        "level": _LEVEL_NAMES.get(record.levelno, record.levelname.lower()),
        "message": record.getMessage(),
    }
    fields = getattr(record, "fields", None)
    if isinstance(fields, Mapping):
        for key, value in fields.items():
            if key not in _RESERVED_FIELDS:
                entry[key] = value
    if record.exc_info:
        error = _error_object(record.exc_info)
        if error:
            entry["error"] = error
    return entry


def _open_records(database: Path) -> sqlite3.Connection:
    return open_sqlite_store(database, RECORDS_FORMAT_VERSION, _SCHEMA)


# How many lines may wait for the writer. A logging burst beyond this while the store
# stalls is counted, not queued, so memory stays bounded; the count is written as one
# `log.dropped` line once the writer catches up.
_PENDING_LIMIT = 10_000


class RecordsHandler(logging.Handler):
    """Writes each log line as one row of this launch's session in the records
    database (logging-conventions, data-lifecycle-conventions).

    The thread that logs only builds the line and queues it; one writer thread owns
    the database, the fallback file and the stored listener, so a stalled store
    never holds up the window or a worker (logging-conventions, "Execution and
    durability"). Lines keep their order.
    """

    def __init__(self, database: Path, *, session: str, fallback: Path) -> None:
        super().__init__()
        self._database = database
        self.session = session
        self._fallback = fallback
        self._connection: sqlite3.Connection | None = None
        # Called after each line the database stored, on the writer thread; a line
        # that went to the fallback file is not in the database, so it calls
        # nothing. The Records window reads the newest records on it.
        self.stored_listener: Callable[[], None] | None = None
        self._condition = threading.Condition()
        self._pending: deque[dict[str, Any]] = deque()
        self._dropped = 0
        self._writing = False
        self._closed = False
        self._writer = threading.Thread(target=self._run, name="pixelup-records", daemon=True)
        self._writer.start()

    def handle(self, record: logging.LogRecord) -> bool:
        # Overridden so the caller never takes the handler lock that would wait on
        # the writer; building the entry here keeps the caller's exception context.
        if not self.filter(record):
            return False
        # A JSON round trip freezes the caller's field values as they are now, so a
        # later change to a dict the caller still holds never reaches the record.
        entry = json.loads(_dumps(_entry(record)))
        with self._condition:
            if self._closed:
                return True
            if len(self._pending) >= _PENDING_LIMIT:
                self._dropped += 1
                return True
            self._pending.append(entry)
            self._condition.notify_all()
        return True

    def emit(self, record: logging.LogRecord) -> None:
        self.handle(record)

    def flush_pending(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for queued lines; True when none remain."""
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._pending and not self._writing, timeout
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending:
                    return
                entry = self._pending.popleft()
                # Lines are dropped only while the queue is full, so the count
                # follows the last line that was queued before them.
                dropped = 0
                if not self._pending:
                    dropped, self._dropped = self._dropped, 0
                self._writing = True
            try:
                self._write(entry)
                if dropped:
                    self._write(_dropped_entry(dropped))
            finally:
                with self._condition:
                    self._writing = False
                    self._condition.notify_all()

    def _write(self, entry: dict[str, Any]) -> None:
        try:
            self._insert(entry)
        except Exception as exc:  # noqa: BLE001 - logging never crashes the app.
            self._fall_back(entry, exc)
            return
        listener = self.stored_listener
        if listener is None:
            return
        try:
            listener()
        except Exception as exc:  # noqa: BLE001 - logging never crashes the app.
            sys.stderr.write(f"[pixelup:records] stored listener failed: {exc!r}\n")
            sys.stderr.flush()

    def _insert(self, entry: dict[str, Any]) -> None:
        # not recorded: records.sqlite3 is a binary store written through SQLite,
        # never the managed-text atomic-write path (data-backup-conventions).
        if self._connection is None:
            self._connection = _open_records(self._database)
        fields = {key: value for key, value in entry.items() if key not in _COLUMNS}
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                "INSERT INTO logs (session, time, level, message, job_id, operation_id, fields)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session,
                    entry["time"],
                    entry["level"],
                    entry["message"],
                    entry.get("job_id"),
                    entry.get("operation_id"),
                    _dumps(fields),
                ),
            )
            self._connection.commit()
        except BaseException as exc:
            rollback_sqlite(self._connection, exc)
            raise

    def _fall_back(self, entry: dict[str, Any], exc: Exception) -> None:
        line = (
            _dumps(
                {
                    **entry,
                    "session": self.session,
                    "records_error": repr(exc),
                    "records_error_details": _error_object((type(exc), exc, exc.__traceback__)),
                }
            )
            + "\n"
        )
        # not recorded: the fallback file is append-mode, never the managed-text
        # atomic-write path (data-backup-conventions).
        try:
            self._fallback.parent.mkdir(parents=True, exist_ok=True)
            with self._fallback.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            sys.stderr.write(line)
            sys.stderr.flush()

    def close(self) -> None:
        """Give queued lines a short bound, then stop the writer. A writer still stuck
        in the store keeps its connection; process exit ends both."""
        self.flush_pending(_CLOSE_FLUSH_SECONDS)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._writer.join(_CLOSE_FLUSH_SECONDS)
        if not self._writer.is_alive() and self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None
        super().close()


# Shutdown's bounded best-effort flush; the quit watchdog still owns the hard exit.
_CLOSE_FLUSH_SECONDS = 1.0


def _dropped_entry(count: int) -> dict[str, Any]:
    return {
        "time": to_utc_iso_ms(datetime.now(UTC)),
        "level": "warn",
        "message": "log.dropped",
        "count": count,
    }


def flush_records(timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for this launch's queued lines to be written."""
    handler = _records_handler()
    return True if handler is None else handler.flush_pending(timeout)


class SessionLog:
    """The app-wide structured logger. Callers pass a short, stable ``message``
    plus keyword fields; the handler writes them as one record. ``message``
    is the greppable event identity, fields carry the data — never build a
    pre-formatted string."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, message, fields)

    def error(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._emit(logging.ERROR, message, fields, exc_info=exc_info)

    def exception(self, message: str, **fields: Any) -> None:
        """Log at error level with the exception currently being handled; call
        from inside an ``except`` block."""
        self._emit(logging.ERROR, message, fields, exc_info=True)

    def _emit(
        self,
        level: int,
        message: str,
        fields: Mapping[str, Any],
        *,
        exc_info: Any = None,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        self._logger.log(level, message, extra={"fields": dict(fields)}, exc_info=exc_info)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# Process-wide singleton. Import this and call log.info(...) etc.; before
# configure_session_logging() runs it falls back to stdlib's last-resort handler.
log = SessionLog(get_logger())


def _records_handler() -> RecordsHandler | None:
    return next(
        (handler for handler in get_logger().handlers if isinstance(handler, RecordsHandler)),
        None,
    )


def current_session() -> str | None:
    """This launch's session, as every record it writes carries, once logging is set up."""
    handler = _records_handler()
    return None if handler is None else handler.session


def set_stored_listener(listener: Callable[[], None] | None) -> None:
    """Call ``listener`` after each line the records database stores (any thread)."""
    handler = _records_handler()
    if handler is not None:
        handler.stored_listener = listener


def debug_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether developer ``debug`` logging is on. Off by default — debug is for a
    developer diagnosing the app and must never be emitted on end-user machines.
    Enable it only by setting ``PIXELUP_DEBUG`` to a truthy value."""
    source = env if env is not None else os.environ
    return source.get(DEBUG_ENV, "").strip().lower() not in {"", "0", "false", "no", "off"}


def configure_session_logging() -> Path:
    """Start this launch's session and route the app logger to the records database.

    Debug is enabled only when PIXELUP_DEBUG is set. Returns the database path.
    """
    started = datetime.now(UTC)
    root = resolve_state_dir()
    database = root / RECORDS_FILE_NAME

    logger = get_logger()
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG if debug_enabled() else logging.INFO)
    logger.propagate = False
    logger.addHandler(
        RecordsHandler(
            database,
            session=to_utc_iso_ms(started),
            fallback=root / "logs" / f"{utc_stamp(started)}.log",
        )
    )

    _install_excepthook()
    log.info("log.session_started", records=str(database), debug=debug_enabled())
    return database


# The interpreter's excepthook from before PixelUp first wrapped it. Captured
# once so repeated configure_session_logging() calls re-install our hook without
# chaining onto a previous copy of itself — which would log a single crash once
# per configuration.
_BASE_EXCEPTHOOK: Any = None


def _install_excepthook() -> None:
    global _BASE_EXCEPTHOOK
    if _BASE_EXCEPTHOOK is None:
        _BASE_EXCEPTHOOK = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        log.error("unhandled.exception", exc_info=(exc_type, exc, tb))
        _BASE_EXCEPTHOOK(exc_type, exc, tb)

    sys.excepthook = hook
