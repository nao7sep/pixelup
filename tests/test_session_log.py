import json
import shutil
import sqlite3
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pixelup.session_log import (
    RECORDS_FILE_NAME,
    configure_session_logging,
    current_session,
    debug_enabled,
    flush_records,
    log,
    set_stored_listener,
)


def _records(database: Path) -> list[dict]:
    """Every log row as the line it stands for: columns and fields in one object,
    once the writer has written every queued line (unless called from the writer)."""
    if threading.current_thread().name != "pixelup-records":
        flush_records()
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT session, time, level, message, job_id, operation_id, fields"
            " FROM logs ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    records = []
    for session, moment, level, message, job_id, operation_id, fields in rows:
        record = {"session": session, "time": moment, "level": level, "message": message}
        if job_id is not None:
            record["job_id"] = job_id
        if operation_id is not None:
            record["operation_id"] = operation_id
        records.append({**record, **json.loads(fields)})
    return records


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_configure_writes_records_under_the_storage_root(tmp_path: Path) -> None:
    database = configure_session_logging()
    log.info("image.added", input="a.png", count=3)

    assert database == (tmp_path / RECORDS_FILE_NAME).resolve()
    records = _records(database)
    assert records[0]["message"] == "log.session_started"
    entry = records[-1]
    assert entry["message"] == "image.added"
    assert entry["level"] == "info"
    assert entry["input"] == "a.png"
    assert entry["count"] == 3
    assert entry["time"].endswith("Z")
    parsed = datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_every_record_carries_its_launch_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # A session is named by its start millisecond; two launches one millisecond apart.
    launches = iter(
        (
            datetime(2026, 10, 5, 12, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 10, 5, 12, 0, 0, 1000, tzinfo=UTC),
        )
    )

    class _Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return next(launches)

    monkeypatch.setattr("pixelup.session_log.datetime", _Clock)
    database = configure_session_logging()
    log.info("first.launch")
    configure_session_logging()
    log.info("second.launch")

    records = _records(database)
    assert {entry["session"] for entry in records[:2]} == {"2026-10-05T12:00:00.000Z"}
    assert {entry["session"] for entry in records[2:]} == {"2026-10-05T12:00:00.001Z"}


def test_domain_ids_are_columns() -> None:
    database = configure_session_logging()
    log.info("job.started", job_id=4)
    log.info("models.install_started", operation_id=2)
    flush_records()

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT message, job_id, operation_id, fields FROM logs"
            " WHERE job_id IS NOT NULL OR operation_id IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("job.started", 4, None, "{}"), ("models.install_started", None, 2, "{}")]


def test_warning_level_renders_as_warn() -> None:
    database = configure_session_logging()
    log.warning("open.ignored_non_file", path="x")

    assert _records(database)[-1]["level"] == "warn"


def test_logged_fields_are_kept_as_given() -> None:
    database = configure_session_logging()
    log.info("auth.try", token="supersecret", user="bob")

    entry = _records(database)[-1]
    assert entry["token"] == "supersecret"
    assert entry["user"] == "bob"


def test_reserved_envelope_fields_cannot_be_overwritten() -> None:
    database = configure_session_logging()
    # `level`, `time` and `session` collide with the envelope; the real values are
    # kept rather than a caller's field. (`message` is the positional argument, so
    # it cannot be passed as a field at all.)
    log.info("real.event", level="bogus", time="bogus", session="bogus", detail="kept")

    entry = _records(database)[-1]
    assert entry["message"] == "real.event"
    assert entry["level"] == "info"
    assert entry["time"] != "bogus"
    assert entry["session"] != "bogus"
    assert entry["detail"] == "kept"


def test_error_includes_type_message_and_cause_chain() -> None:
    database = configure_session_logging()
    try:
        try:
            raise ValueError("root cause")
        except ValueError as inner:
            raise RuntimeError("wrapped failure") from inner
    except RuntimeError:
        log.exception("job.failed_unexpectedly", job_id=7)

    entry = _records(database)[-1]
    assert entry["level"] == "error"
    assert entry["job_id"] == 7
    assert entry["error"]["type"] == "RuntimeError"
    assert entry["error"]["message"] == "wrapped failure"
    traceback_text = entry["error"]["traceback"]
    assert "RuntimeError: wrapped failure" in traceback_text
    assert "ValueError: root cause" in traceback_text


@pytest.mark.parametrize("transported", [False, True])
def test_warning_keeps_the_original_exception_and_cause(transported: bool) -> None:
    database = configure_session_logging()

    def fail() -> None:
        try:
            raise ValueError("original diagnostic sentinel")
        except ValueError as cause:
            raise RuntimeError("wrapped warning") from cause

    try:
        fail()
    except RuntimeError as exc:
        if transported:
            failure = exc
        else:
            log.warning("optional.failed", exc_info=True, operation="save")
    if transported:
        # Worker outcomes are delivered after their except block has ended.
        log.warning(
            "optional.failed",
            exc_info=(type(failure), failure, failure.__traceback__),
            operation="save",
        )

    entry = _records(database)[-1]
    assert entry["level"] == "warn"
    assert entry["operation"] == "save"
    assert "exc_info" not in entry
    assert entry["error"]["type"] == "RuntimeError"
    assert entry["error"]["message"] == "wrapped warning"
    assert "ValueError: original diagnostic sentinel" in entry["error"]["traceback"]
    assert "in fail" in entry["error"]["traceback"]


def test_non_serializable_field_values_do_not_break_a_record() -> None:
    database = configure_session_logging()
    # Path objects are common in PixelUp's payloads and are not natively JSON
    # serializable; they are coerced rather than raising.
    log.info("log.revealed", log_file=Path("/tmp/x.log"))

    assert _records(database)[-1]["log_file"] == str(Path("/tmp/x.log"))


def test_unserializable_field_never_drops_the_record() -> None:
    database = configure_session_logging()
    # A dict keyed by a tuple makes json.dumps raise (non-string key); the record
    # is still written with the field coerced.
    log.info("weird.payload", counts={(1, 2): "pair"}, ok="kept")

    entry = _records(database)[-1]
    assert entry["message"] == "weird.payload"
    assert entry["ok"] == "kept"
    assert entry["counts"] == {"(1, 2)": "pair"}


def test_caller_error_field_does_not_clobber_exception_payload() -> None:
    database = configure_session_logging()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("op.failed", error="caller-supplied")

    entry = _records(database)[-1]
    assert isinstance(entry["error"], dict)
    assert entry["error"]["type"] == "RuntimeError"


def test_debug_enabled_reads_env() -> None:
    assert debug_enabled({}) is False
    assert debug_enabled({"PIXELUP_DEBUG": ""}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "0"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "false"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "off"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "1"}) is True
    assert debug_enabled({"PIXELUP_DEBUG": "yes"}) is True


def test_debug_is_suppressed_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXELUP_DEBUG", raising=False)
    database = configure_session_logging()
    log.debug("job.progress", tick=1)

    assert "job.progress" not in [entry["message"] for entry in _records(database)]


def test_debug_is_emitted_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXELUP_DEBUG", "1")
    database = configure_session_logging()
    log.debug("job.progress", tick=1)

    assert "job.progress" in [entry["message"] for entry in _records(database)]


def test_a_working_database_leaves_no_logs_folder(tmp_path: Path) -> None:
    configure_session_logging()
    log.info("image.added", input="a.png")

    flush_records()
    assert not (tmp_path / "logs").exists()


def test_an_unwritable_database_falls_back_to_a_text_file(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()

    configure_session_logging()
    log.info("image.added", input="a.png", job_id=3)

    flush_records()
    fallbacks = list((tmp_path / "logs").glob("*-utc.log"))
    assert len(fallbacks) == 1
    lines = _read_jsonl(fallbacks[0])
    assert [line["message"] for line in lines] == ["log.session_started", "image.added"]
    entry = lines[-1]
    assert entry["input"] == "a.png"
    assert entry["job_id"] == 3
    assert entry["session"].endswith("Z")
    assert entry["records_error"]


def test_with_no_fallback_file_the_record_reaches_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()
    configure_session_logging()
    flush_records()
    shutil.rmtree(tmp_path / "logs")
    (tmp_path / "logs").write_text("not a directory", encoding="utf-8")

    log.info("image.added", input="a.png")

    flush_records()
    entry = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert entry["message"] == "image.added"
    assert entry["records_error"]


def test_excepthook_does_not_stack_across_reconfiguration() -> None:
    # Reconfigure twice in one process; the excepthook must not chain onto a
    # previous copy of itself, or a single crash would be logged once per call.
    configure_session_logging()
    database = configure_session_logging()

    try:
        raise ValueError("synthetic crash")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    crashes = [e for e in _records(database) if e["message"] == "unhandled.exception"]
    assert len(crashes) == 1
    assert crashes[0]["error"]["type"] == "ValueError"


def test_the_stored_listener_hears_each_line_the_database_stored() -> None:
    database = configure_session_logging()
    flush_records()
    heard: list[int] = []
    set_stored_listener(lambda: heard.append(len(_records(database))))

    log.info("image.added", input="a.png")
    log.warning("open.ignored_directory", path="d")

    flush_records()
    # Called after the line is in the database, so a read it starts sees it.
    assert heard == [2, 3]


def test_a_line_that_went_to_the_fallback_file_tells_no_listener(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()
    configure_session_logging()
    heard: list[None] = []
    set_stored_listener(lambda: heard.append(None))

    log.info("image.added", input="a.png")

    flush_records()
    assert heard == []


def test_a_failing_listener_never_costs_the_line(capsys: pytest.CaptureFixture[str]) -> None:
    database = configure_session_logging()

    def fail() -> None:
        raise RuntimeError("listener gone")

    set_stored_listener(fail)
    log.info("image.added", input="a.png")

    assert _records(database)[-1]["message"] == "image.added"
    assert "listener gone" in capsys.readouterr().err


def test_current_session_is_the_session_every_record_carries() -> None:
    database = configure_session_logging()

    assert {record["session"] for record in _records(database)} == {current_session()}


def _existing_records(database: Path, version: int, kind: str | None = "records") -> bytes:
    """An existing database with ``version``, and a header naming ``kind`` (none when
    ``kind`` is None)."""
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE logs (anything TEXT)")
    if kind is not None:
        connection.execute("CREATE TABLE file_header (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO file_header VALUES (?, ?)",
            [("format.owner", "pixelup"), ("format.kind", kind)],
        )
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()
    return database.read_bytes()


def test_the_records_database_is_stamped_at_creation(tmp_path: Path) -> None:
    database = configure_session_logging()
    log.info("image.added")
    flush_records()
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        rows = dict(connection.execute("SELECT key, value FROM file_header").fetchall())
        assert (rows["format.owner"], rows["format.kind"]) == ("pixelup", "records")
        assert rows["writer.name"] == "pixelup"
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("version", "kind", "reason"),
    [
        (0, "records", "no format version"),
        (1, None, "no format header"),
        (1, "backups", "not a pixelup records file"),
        (2, "records", "format version 2"),
    ],
)
def test_unreadable_or_newer_records_are_left_untouched_and_lines_go_to_the_fallback_file(
    tmp_path: Path, version: int, kind: str | None, reason: str
) -> None:
    database = tmp_path / RECORDS_FILE_NAME
    before = _existing_records(database, version, kind)

    configure_session_logging()
    log.info("image.added", input="a.png")

    flush_records()
    assert database.read_bytes() == before
    assert not (tmp_path / f"{RECORDS_FILE_NAME}-wal").exists()
    flush_records()
    fallbacks = list((tmp_path / "logs").glob("*-utc.log"))
    lines = _read_jsonl(fallbacks[0])
    assert [line["message"] for line in lines] == ["log.session_started", "image.added"]
    assert reason in lines[-1]["records_error"]



def test_insert_and_rollback_failure_falls_back_with_primary_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from pixelup.session_log import RecordsHandler, _open_records

    database = tmp_path / "records.sqlite3"
    connection = _open_records(database)

    class FailingCleanup:
        def execute(self, sql, *args):
            if sql.startswith("INSERT"):
                raise sqlite3.OperationalError("primary insert failure")
            return connection.execute(sql, *args)

        def rollback(self):
            connection.rollback()
            raise sqlite3.OperationalError("cleanup rollback failure")

    handler = RecordsHandler(database, session="session", fallback=tmp_path / "fallback.log")
    handler._connection = FailingCleanup()
    try:
        handler.emit(logging.LogRecord("pixelup", logging.INFO, "", 0, "entry", (), None))
        handler.flush_pending()
    finally:
        handler._connection = None
        connection.close()
        handler.close()
    line = _read_jsonl(tmp_path / "fallback.log")[0]
    assert "primary insert failure" in line["records_error"]
    assert "cleanup rollback failure" in line["records_error_details"]["traceback"]


def test_logging_never_waits_on_a_stuck_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # The caller finishes while the store is still held; queued lines land in
    # order after the backing write is released.
    from pixelup import session_log

    database = configure_session_logging()
    flush_records()
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    handler = session_log._records_handler()
    real_insert = handler._insert

    def stuck_insert(entry: dict) -> None:
        entered.set()
        release.wait()
        real_insert(entry)

    def write_lines() -> None:
        log.info("first.line")
        log.info("second.line")
        log.info("third.line")
        returned.set()

    monkeypatch.setattr(handler, "_insert", stuck_insert)
    caller = threading.Thread(target=write_lines)
    caller.start()
    try:
        assert entered.wait(), "the backing write never started"
        assert returned.wait(), "logging waited for the held backing write"
        assert not release.is_set()
        assert handler._pending
    finally:
        release.set()
        caller.join()
        assert not caller.is_alive()
        flush_records()
    assert [entry["message"] for entry in _records(database)][-3:] == [
        "first.line",
        "second.line",
        "third.line",
    ]


def test_a_field_changed_after_logging_is_recorded_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = configure_session_logging()
    values = {"items": [1]}
    log.info("snapshot.line", values=values)
    values["items"].append(2)

    assert _records(database)[-1]["values"] == {"items": [1]}


def test_an_escaped_background_thread_exception_is_recorded(monkeypatch) -> None:
    database = configure_session_logging()
    monkeypatch.setattr(threading, "__excepthook__", lambda _args: None)

    def broken() -> None:
        raise RuntimeError("background broke")

    thread = threading.Thread(target=broken, name="pixelup-test-broken")
    thread.start()
    thread.join()

    [entry] = [e for e in _records(database) if e["message"] == "unhandled.thread_exception"]
    assert entry["thread"] == "pixelup-test-broken"
    assert entry["error"]["type"] == "RuntimeError"
