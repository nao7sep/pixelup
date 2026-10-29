"""Reading records.sqlite3 for the Records window: pages, filters, a record
whole and the launches, read on the records writer's thread and connection,
never logging."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from pixelup import records
from pixelup.formats import NewerFormatError
from pixelup.records import (
    PAGE_SIZE,
    RecordCursor,
    RecordDetail,
    RecordsPage,
    RecordsQuery,
    RecordSummary,
    cursor_after,
    merge_newest_page,
    read_detail,
    read_page,
    read_sessions,
    split_fields,
)
from pixelup.session_log import (
    RecordsHandler,
    _open_records,
    configure_session_logging,
    flush_records,
    log,
    records_store,
    set_stored_listener,
)

OLD = "2026-09-30T08:00:00.000Z"
NEW = "2026-10-01T09:00:00.000Z"


def _seed(database: Path, rows: list[dict]) -> None:
    connection = _open_records(database)
    try:
        for row in rows:
            connection.execute(
                "INSERT INTO logs (session, time, level, message, job_id, operation_id, fields)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("session", NEW),
                    row["time"],
                    row.get("level", "info"),
                    row.get("message", "image.added"),
                    row.get("job_id"),
                    row.get("operation_id"),
                    json.dumps(row.get("fields", {})),
                ),
            )
    finally:
        connection.close()


def _time(second: int) -> str:
    return f"2026-10-01T09:{second // 60:02d}:{second % 60:02d}.000Z"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "records.sqlite3"


@pytest.fixture
def connection(database: Path):
    connection = _open_records(database)
    yield connection
    connection.close()


def _read_through(handler: RecordsHandler, run):  # type: ignore[no-untyped-def]
    """A read the handler's writer runs, waited for: its result, or its failure."""
    settled = threading.Event()
    outcome: list[object] = []

    def done(result: object, error: Exception | None) -> None:
        outcome.append(result if error is None else error)
        settled.set()

    handler.read(run, done)
    assert settled.wait()
    return outcome[0]


def _handler(database: Path) -> RecordsHandler:
    return RecordsHandler(database, session=NEW, fallback=database.with_suffix(".log"))


def _messages(page: RecordsPage) -> list[str]:
    return [record.message for record in page.records]


def test_pages_every_record_newest_first_and_continues_from_the_last_one(
    database: Path, connection: sqlite3.Connection
) -> None:
    total = PAGE_SIZE * 2 + 5
    _seed(database, [{"time": _time(i), "message": f"m{i}"} for i in range(total)])

    first = read_page(connection, RecordsQuery(), None)
    assert len(first.records) == PAGE_SIZE and first.more
    assert first.records[0].message == f"m{total - 1}"
    seen = list(first.records)
    page = first
    while page.more:
        page = read_page(connection, RecordsQuery(), cursor_after(page.records))
        seen += page.records
    assert [record.message for record in seen] == [f"m{i}" for i in reversed(range(total))]


def test_records_of_one_instant_page_by_their_id(
    database: Path, connection: sqlite3.Connection
) -> None:
    _seed(database, [{"time": NEW, "message": f"m{i}"} for i in range(PAGE_SIZE + 3)])

    first = read_page(connection, RecordsQuery(), None)
    rest = read_page(connection, RecordsQuery(), cursor_after(first.records))

    assert not rest.more
    assert len({record.id for record in (*first.records, *rest.records)}) == PAGE_SIZE + 3


def test_filters_by_launch_level_and_search(database: Path, connection: sqlite3.Connection) -> None:
    _seed(
        database,
        [
            {"time": _time(1), "session": OLD, "message": "app.started"},
            {"time": _time(2), "level": "warn", "message": "open.ignored_directory"},
            {
                "time": _time(3),
                "level": "error",
                "message": "job.failed",
                "fields": {"input": "a.png"},
            },
            {"time": _time(4), "level": "debug", "message": "job.progress"},
            {"time": _time(5), "message": "image.added", "fields": {"input": "b.png"}},
        ],
    )

    assert _messages(read_page(connection, RecordsQuery(session=OLD), None)) == ["app.started"]
    assert _messages(read_page(connection, RecordsQuery(level="attention"), None)) == [
        "job.failed",
        "open.ignored_directory",
    ]
    assert _messages(read_page(connection, RecordsQuery(level="debug"), None)) == ["job.progress"]
    # Search reads the message and every stored field.
    assert _messages(read_page(connection, RecordsQuery(search=" ignored "), None)) == [
        "open.ignored_directory"
    ]
    assert _messages(read_page(connection, RecordsQuery(search="b.png"), None)) == ["image.added"]
    assert _messages(read_page(connection, RecordsQuery(level="error", search="b.png"), None)) == []


def test_search_takes_percent_and_underscore_literally(
    database: Path, connection: sqlite3.Connection
) -> None:
    _seed(
        database,
        [
            {"time": _time(1), "message": "a_b", "fields": {"progress": "50%"}},
            {"time": _time(2), "message": "axb", "fields": {"progress": "50x"}},
        ],
    )

    assert _messages(read_page(connection, RecordsQuery(search="a_b"), None)) == ["a_b"]
    assert _messages(read_page(connection, RecordsQuery(search="50%"), None)) == ["a_b"]


def test_reads_a_record_whole_and_the_launches_newest_first(
    database: Path, connection: sqlite3.Connection
) -> None:
    fields = {"input": "a.png", "error": {"type": "ValueError", "message": "boom"}}
    _seed(
        database,
        [
            {"time": _time(1), "session": OLD},
            {"time": _time(2), "job_id": 3, "operation_id": 2, "level": "error", "fields": fields},
        ],
    )
    newest = read_page(connection, RecordsQuery(), None).records[0]

    detail = read_detail(connection, newest.id)

    assert detail == RecordDetail(
        newest.id, NEW, _time(2), "error", "image.added", 3, 2, json.dumps(fields)
    )
    assert read_detail(connection, newest.id + 100) is None
    assert read_sessions(connection) == (NEW, OLD)


def test_a_read_never_stores_a_record() -> None:
    # Every stored record tells the Records window to read again, so a read
    # that logged would read forever.
    configure_session_logging()
    flush_records()
    stored: list[None] = []
    set_stored_listener(lambda: stored.append(None))
    handler = records_store()
    assert handler is not None

    def read_all(connection: sqlite3.Connection) -> None:
        page = read_page(connection, RecordsQuery(), None)
        read_detail(connection, page.records[0].id)
        read_sessions(connection)

    assert _read_through(handler, read_all) is None
    flush_records()
    assert stored == []

def _summary(record_id: int, time: str) -> RecordSummary:
    return RecordSummary(record_id, NEW, time, "info", f"m{record_id}", "{}")


def test_merge_puts_new_records_ahead_of_the_rows_shown_and_keeps_the_pages_read() -> None:
    shown = [_summary(3, _time(3)), _summary(2, _time(2)), _summary(1, _time(1))]
    page = RecordsPage((_summary(5, _time(5)), _summary(4, _time(4)), _summary(3, _time(3))), True)

    records, more = merge_newest_page(shown, False, page)

    assert [record.id for record in records] == [5, 4, 3, 2, 1]
    # Rows shown beyond the page keep the list's own word on what follows.
    assert more is False


def test_merge_takes_the_pages_word_when_it_reaches_past_every_row_shown() -> None:
    shown = [_summary(2, _time(2))]
    page = RecordsPage((_summary(3, _time(3)), _summary(2, _time(2)), _summary(1, _time(1))), True)

    assert merge_newest_page(shown, False, page)[1] is True


def test_merge_loses_nothing_to_an_older_page_arriving_after_a_newer_one() -> None:
    shown = [_summary(4, _time(4)), _summary(3, _time(3))]
    page = RecordsPage((_summary(3, _time(3)),), False)

    records, more = merge_newest_page(shown, True, page)

    assert [record.id for record in records] == [4, 3]
    assert more is False


def test_merge_takes_the_pages_copy_of_a_row_it_shares_with_the_list() -> None:
    shown = [_summary(1, _time(1))]
    fresh = RecordSummary(1, NEW, _time(1), "info", "m1", '{"a": 1}')

    records, _more = merge_newest_page(shown, False, RecordsPage((fresh,), False))

    assert records == [fresh]


def test_cursor_after_names_the_last_record_shown() -> None:
    assert cursor_after([]) is None
    assert cursor_after([_summary(2, _time(2)), _summary(1, _time(1))]) == RecordCursor(_time(1), 1)


def test_a_records_error_is_shown_apart_from_its_details() -> None:
    details, error = split_fields(
        json.dumps({"input": "a.png", "error": {"type": "ValueError"}}, ensure_ascii=False)
    )

    assert details == '{\n  "input": "a.png"\n}'
    assert error == '{\n  "type": "ValueError"\n}'
    assert split_fields("{}") == (None, None)
    assert split_fields("not json") == ("not json", None)


def test_the_list_reads_a_bounded_start_of_each_records_fields(
    database: Path, connection: sqlite3.Connection
) -> None:
    _seed(database, [{"time": _time(1), "fields": {"traceback": "x" * 5000}}])

    summary = read_page(connection, RecordsQuery(), None).records[0]

    assert len(summary.fields) == records._SUMMARY_FIELDS_LENGTH
    assert len(read_detail(connection, summary.id).fields) > 5000


def test_a_log_line_is_read_back_as_it_was_written() -> None:
    configure_session_logging()
    log.info("image.added", input="a.png", job_id=4)
    handler = records_store()
    assert handler is not None

    # Queued behind the line, so it reads the line without waiting for a flush.
    def newest(connection: sqlite3.Connection) -> RecordDetail | None:
        return read_detail(connection, read_page(connection, RecordsQuery(), None).records[0].id)

    detail = _read_through(handler, newest)

    assert isinstance(detail, RecordDetail) and detail.message == "image.added"
    assert detail.job_id == 4
    assert json.loads(detail.fields) == {"input": "a.png"}


def test_newer_records_are_named_and_never_read(database: Path) -> None:
    _seed(database, [{"time": OLD}])
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 2")
    connection.close()
    before = database.read_bytes()
    handler = _handler(database)
    try:
        failure = _read_through(handler, lambda c: read_page(c, RecordsQuery(), None))
    finally:
        handler.close()
    assert isinstance(failure, NewerFormatError)
    assert failure.version == 2
    assert failure.path == database
    assert database.read_bytes() == before


def test_records_without_their_format_version_are_unreadable(database: Path) -> None:
    _seed(database, [{"time": OLD}])
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 0")
    connection.close()
    handler = _handler(database)
    try:
        failure = _read_through(handler, lambda c: read_page(c, RecordsQuery(), None))
    finally:
        handler.close()
    assert isinstance(failure, ValueError) and "no format version" in str(failure)
