"""Reading records.sqlite3 for the Records window: pages, filters, a record
whole, the launches, and reads that are bounded and never write."""

from __future__ import annotations

import json
import sqlite3
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
    RecordsReader,
    RecordSummary,
    cursor_after,
    merge_newest_page,
    split_fields,
)
from pixelup.session_log import _open_records, configure_session_logging, log, set_stored_listener

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
def reader(database: Path):
    reader = RecordsReader(database)
    yield reader
    reader.close()


def _messages(page: RecordsPage) -> list[str]:
    return [record.message for record in page.records]


def test_pages_every_record_newest_first_and_continues_from_the_last_one(
    database: Path, reader: RecordsReader
) -> None:
    total = PAGE_SIZE * 2 + 5
    _seed(database, [{"time": _time(i), "message": f"m{i}"} for i in range(total)])

    first = reader.page(RecordsQuery(), None)
    assert len(first.records) == PAGE_SIZE and first.more
    assert first.records[0].message == f"m{total - 1}"
    seen = list(first.records)
    page = first
    while page.more:
        page = reader.page(RecordsQuery(), cursor_after(page.records))
        seen += page.records
    assert [record.message for record in seen] == [f"m{i}" for i in reversed(range(total))]


def test_records_of_one_instant_page_by_their_id(database: Path, reader: RecordsReader) -> None:
    _seed(database, [{"time": NEW, "message": f"m{i}"} for i in range(PAGE_SIZE + 3)])

    first = reader.page(RecordsQuery(), None)
    rest = reader.page(RecordsQuery(), cursor_after(first.records))

    assert not rest.more
    assert len({record.id for record in (*first.records, *rest.records)}) == PAGE_SIZE + 3


def test_filters_by_launch_level_and_search(database: Path, reader: RecordsReader) -> None:
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

    assert _messages(reader.page(RecordsQuery(session=OLD), None)) == ["app.started"]
    assert _messages(reader.page(RecordsQuery(level="attention"), None)) == [
        "job.failed",
        "open.ignored_directory",
    ]
    assert _messages(reader.page(RecordsQuery(level="debug"), None)) == ["job.progress"]
    # Search reads the message and every stored field.
    assert _messages(reader.page(RecordsQuery(search=" ignored "), None)) == [
        "open.ignored_directory"
    ]
    assert _messages(reader.page(RecordsQuery(search="b.png"), None)) == ["image.added"]
    assert _messages(reader.page(RecordsQuery(level="error", search="b.png"), None)) == []


def test_search_takes_percent_and_underscore_literally(
    database: Path, reader: RecordsReader
) -> None:
    _seed(
        database,
        [
            {"time": _time(1), "message": "a_b", "fields": {"progress": "50%"}},
            {"time": _time(2), "message": "axb", "fields": {"progress": "50x"}},
        ],
    )

    assert _messages(reader.page(RecordsQuery(search="a_b"), None)) == ["a_b"]
    assert _messages(reader.page(RecordsQuery(search="50%"), None)) == ["a_b"]


def test_reads_a_record_whole_and_the_launches_newest_first(
    database: Path, reader: RecordsReader
) -> None:
    fields = {"input": "a.png", "error": {"type": "ValueError", "message": "boom"}}
    _seed(
        database,
        [
            {"time": _time(1), "session": OLD},
            {"time": _time(2), "job_id": 3, "operation_id": 2, "level": "error", "fields": fields},
        ],
    )
    newest = reader.page(RecordsQuery(), None).records[0]

    detail = reader.detail(newest.id)

    assert detail == RecordDetail(
        newest.id, NEW, _time(2), "error", "image.added", 3, 2, json.dumps(fields)
    )
    assert reader.detail(newest.id + 100) is None
    assert reader.sessions() == (NEW, OLD)


def test_a_read_neither_creates_the_database_nor_writes_to_it(
    database: Path, reader: RecordsReader
) -> None:
    with pytest.raises(sqlite3.OperationalError):
        reader.page(RecordsQuery(), None)
    assert not database.exists()

    _seed(database, [{"time": _time(1)}])
    reader.page(RecordsQuery(), None)
    connection = reader._open()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connection.execute("DELETE FROM logs")


def test_a_read_never_stores_a_record() -> None:
    # Every stored record tells the Records window to read again, so a read
    # that logged would read forever.
    database = configure_session_logging()
    stored: list[None] = []
    set_stored_listener(lambda: stored.append(None))
    reader = RecordsReader(database)
    try:
        page = reader.page(RecordsQuery(), None)
        reader.detail(page.records[0].id)
        reader.sessions()
    finally:
        reader.close()

    assert stored == []


def test_a_read_past_its_deadline_is_interrupted(
    database: Path, reader: RecordsReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(database, [{"time": _time(i % 3600)} for i in range(500)])
    monkeypatch.setattr(records, "_PROGRESS_STEPS", 1)
    monkeypatch.setattr(records, "READ_TIMEOUT_SECONDS", -1.0)

    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        reader.page(RecordsQuery(search="nothing"), None)


def test_a_cancelled_reader_reads_nothing_more(database: Path, reader: RecordsReader) -> None:
    _seed(database, [{"time": _time(i % 3600)} for i in range(500)])
    assert reader.page(RecordsQuery(), None).records
    reader.cancel()

    assert reader.cancelled
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        reader.page(RecordsQuery(search="nothing"), None)


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
    database: Path, reader: RecordsReader
) -> None:
    _seed(database, [{"time": _time(1), "fields": {"traceback": "x" * 5000}}])

    summary = reader.page(RecordsQuery(), None).records[0]

    assert len(summary.fields) == records._SUMMARY_FIELDS_LENGTH
    assert len(reader.detail(summary.id).fields) > 5000


def test_a_log_line_is_read_back_as_it_was_written() -> None:
    database = configure_session_logging()
    log.info("image.added", input="a.png", job_id=4)
    reader = RecordsReader(database)
    try:
        newest = reader.page(RecordsQuery(), None).records[0]
        detail = reader.detail(newest.id)
    finally:
        reader.close()

    assert newest.message == "image.added"
    assert detail is not None and detail.job_id == 4
    assert json.loads(detail.fields) == {"input": "a.png"}


def test_newer_records_are_named_and_never_read(database: Path) -> None:
    _seed(database, [{"time": OLD}])
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 2")
    connection.close()
    before = database.read_bytes()
    reader = RecordsReader(database)
    try:
        with pytest.raises(NewerFormatError) as raised:
            reader.page(RecordsQuery(), None)
    finally:
        reader.close()
    assert raised.value.version == 2
    assert raised.value.path == database
    assert database.read_bytes() == before


def test_records_without_their_format_version_are_unreadable(database: Path) -> None:
    _seed(database, [{"time": OLD}])
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 0")
    connection.close()
    reader = RecordsReader(database)
    try:
        with pytest.raises(ValueError, match="no format version"):
            reader.page(RecordsQuery(), None)
    finally:
        reader.close()


@pytest.mark.parametrize("version", [0, -1, 2])
def test_cached_reader_rechecks_marker_before_later_page(database: Path, version: int) -> None:
    _seed(database, [{"time": OLD}])
    reader = RecordsReader(database)
    try:
        assert reader.page(RecordsQuery(), None).records
        with sqlite3.connect(database) as sibling:
            sibling.execute(f"PRAGMA user_version = {version}")
        with pytest.raises(NewerFormatError if version == 2 else ValueError):
            reader.page(RecordsQuery(), None)
    finally:
        reader.close()


def test_reader_primary_survives_failed_rollback_and_deadline_always_clears(
    database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(database, [{"time": OLD}])
    reader = RecordsReader(database)
    connection = reader._open()
    primary = sqlite3.OperationalError("primary read failure")

    class FailingCleanup:
        def execute(self, sql, *args):
            return connection.execute(sql, *args)

        def rollback(self):
            connection.rollback()
            raise sqlite3.OperationalError("cleanup rollback failure")

    monkeypatch.setattr(reader, "_open", lambda: FailingCleanup())
    try:
        with pytest.raises(sqlite3.OperationalError) as raised:
            reader._bounded(lambda _: (_ for _ in ()).throw(primary))
        assert raised.value is primary
        assert "cleanup rollback failure" in raised.value.__notes__[0]
        assert reader._deadline is None
    finally:
        reader.close()
