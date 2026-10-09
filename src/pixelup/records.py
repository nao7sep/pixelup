"""What the Records window reads from records.sqlite3: a filtered page of
summaries, newest first, one record whole, and the launches that have records.

Reads only: nothing here writes to the database or logs, because every stored
record signals the Records window to read again (session_log's stored
listener), so a read that logged would start the next read forever. Each read
is bounded (PLAYBOOK, Bound every external wait): a lock waits at most
LOCK_TIMEOUT_SECONDS and the query itself is interrupted after
READ_TIMEOUT_SECONDS, or as soon as the caller cancels it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pixelup.formats import RECORDS_FORMAT_VERSION, check_sqlite_format, rollback_sqlite

PAGE_SIZE = 100
LOCK_TIMEOUT_SECONDS = 2.0
READ_TIMEOUT_SECONDS = 10.0
# How many SQLite virtual-machine steps run between two looks at the deadline.
_PROGRESS_STEPS = 1000
# The list shows the start of a record's fields; the detail reads them whole.
_SUMMARY_FIELDS_LENGTH = 400

type RecordLevel = Literal["error", "warn", "info", "debug"]
# What the level filter offers: a record's own level, or `attention`, every
# record at `warn` or `error`.
type RecordLevelFilter = Literal["attention", "error", "warn", "info", "debug"]

LEVELS: tuple[RecordLevel, ...] = ("error", "warn", "info", "debug")
LEVEL_FILTERS: tuple[RecordLevelFilter, ...] = ("attention", *LEVELS)


@dataclass(frozen=True, slots=True)
class RecordsQuery:
    # A launch, named by its session.
    session: str | None = None
    level: RecordLevelFilter | None = None
    search: str = ""


@dataclass(frozen=True, slots=True)
class RecordCursor:
    """Where the next page starts: the last summary of the page before it."""

    time: str
    id: int


@dataclass(frozen=True, slots=True)
class RecordSummary:
    id: int
    session: str
    time: str
    level: str
    message: str
    # The start of the stored fields, as JSON text.
    fields: str


@dataclass(frozen=True, slots=True)
class RecordsPage:
    records: tuple[RecordSummary, ...]
    more: bool


@dataclass(frozen=True, slots=True)
class RecordDetail:
    id: int
    session: str
    time: str
    level: str
    message: str
    job_id: int | None
    operation_id: int | None
    # Everything else the line carries, as the JSON text the database holds.
    fields: str


def _like_pattern(search: str) -> str | None:
    trimmed = search.strip()
    if not trimmed:
        return None
    escaped = trimmed.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def read_page(
    connection: sqlite3.Connection, query: RecordsQuery, after: RecordCursor | None
) -> RecordsPage:
    where: list[str] = []
    params: list[object] = []
    if query.session is not None:
        where.append("session = ?")
        params.append(query.session)
    if query.level == "attention":
        where.append("level IN ('warn', 'error')")
    elif query.level is not None:
        where.append("level = ?")
        params.append(query.level)
    pattern = _like_pattern(query.search)
    if pattern is not None:
        where.append("(message LIKE ? ESCAPE '\\' OR fields LIKE ? ESCAPE '\\')")
        params += [pattern, pattern]
    if after is not None:
        where.append("(time < ? OR (time = ? AND id < ?))")
        params += [after.time, after.time, after.id]
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(PAGE_SIZE + 1)
    rows = connection.execute(
        f"SELECT id, session, time, level, message, substr(fields, 1, {_SUMMARY_FIELDS_LENGTH})"
        f" FROM logs {clause} ORDER BY time DESC, id DESC LIMIT ?",
        params,
    ).fetchall()
    records = tuple(RecordSummary(*row) for row in rows[:PAGE_SIZE])
    return RecordsPage(records, len(rows) > PAGE_SIZE)


def read_detail(connection: sqlite3.Connection, record_id: int) -> RecordDetail | None:
    row = connection.execute(
        "SELECT id, session, time, level, message, job_id, operation_id, fields"
        " FROM logs WHERE id = ?",
        (record_id,),
    ).fetchone()
    return None if row is None else RecordDetail(*row)


def read_sessions(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Every launch that has records, newest first."""
    rows = connection.execute("SELECT DISTINCT session FROM logs ORDER BY session DESC")
    return tuple(row[0] for row in rows)


class RecordsReader:
    """One read-only connection to the records database, used from one thread.

    The connection opens on the first read, read-only, so a read never creates
    or changes the file; a read that fails to open tries again on the next one.
    ``cancel`` may be called from any thread: it interrupts the read in flight,
    and every read after it fails at once.
    """

    def __init__(self, database: Path) -> None:
        self._database = database
        self._cancelled = threading.Event()
        self._connection: sqlite3.Connection | None = None
        self._deadline: float | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def page(self, query: RecordsQuery, after: RecordCursor | None) -> RecordsPage:
        return self._bounded(lambda connection: read_page(connection, query, after))

    def detail(self, record_id: int) -> RecordDetail | None:
        return self._bounded(lambda connection: read_detail(connection, record_id))

    def sessions(self) -> tuple[str, ...]:
        return self._bounded(read_sessions)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _bounded[T](self, read: Callable[[sqlite3.Connection], T]) -> T:
        self._deadline = time.monotonic() + READ_TIMEOUT_SECONDS
        connection = None
        try:
            connection = self._open()
            connection.execute("BEGIN")
            result = read(connection)
        except BaseException as exc:
            if connection is not None:
                rollback_sqlite(connection, exc)
            raise
        else:
            connection.rollback()
            return result
        finally:
            self._deadline = None

    def _open(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(
                f"{self._database.absolute().as_uri()}?mode=ro",
                uri=True,
                timeout=LOCK_TIMEOUT_SECONDS,
            )
            connection.set_progress_handler(self._interrupted, _PROGRESS_STEPS)
            try:
                check_sqlite_format(connection, self._database, RECORDS_FORMAT_VERSION)
            except BaseException:
                connection.close()
                raise
            self._connection = connection
        return self._connection

    def _interrupted(self) -> int:
        # A non-zero answer makes SQLite abandon the statement with an
        # OperationalError, which the caller reports like any other failed read.
        deadline = self._deadline
        late = deadline is not None and time.monotonic() > deadline
        return 1 if late or self._cancelled.is_set() else 0


def cursor_after(records: Sequence[RecordSummary]) -> RecordCursor | None:
    """The page after the last record shown."""
    if not records:
        return None
    last = records[-1]
    return RecordCursor(last.time, last.id)


def _order(record: RecordSummary) -> tuple[str, int]:
    # The order the list shows records in, newest first, as the database pages them.
    return (record.time, record.id)


def merge_newest_page(
    shown: Sequence[RecordSummary], shown_more: bool, page: RecordsPage
) -> tuple[list[RecordSummary], bool]:
    """The newest page read again, joined with the rows already shown.

    A row in both takes the page's copy, and the rows shown beyond the page stay,
    so the pages already read are kept and a page read out of order loses nothing.
    """
    by_id = {record.id: record for record in shown}
    for record in page.records:
        by_id[record.id] = record
    records = sorted(by_id.values(), key=_order, reverse=True)
    beyond = bool(page.records) and any(
        _order(record) < _order(page.records[-1]) for record in shown
    )
    return records, shown_more if beyond else page.more


def pretty_json(text: str) -> str:
    """Stored JSON, indented for reading; text that is not JSON is shown as it is."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text


def split_fields(fields: str) -> tuple[str | None, str | None]:
    """A record's fields as its details and its error, each as indented JSON.

    The logger keeps an attached exception under ``error`` beside the caller's
    fields (session_log), so the window shows it as a block of its own, as
    Mumbler shows a log line's error apart from its details.
    """
    try:
        value = json.loads(fields)
    except ValueError:
        return fields, None
    if not isinstance(value, dict):
        return pretty_json(fields), None
    error = value.pop("error", None)
    details = json.dumps(value, indent=2, ensure_ascii=False) if value else None
    shown_error = None if error is None else json.dumps(error, indent=2, ensure_ascii=False)
    return details, shown_error
