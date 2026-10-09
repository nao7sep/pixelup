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
        assert flush_records(5)
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
    assert flush_records(5)

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


def test_an_unwritable_database_falls_back_to_a_text_file(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()

    configure_session_logging()
    log.info("image.added", input="a.png", job_id=3)

    assert flush_records(5)
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
    assert flush_records(5)
    shutil.rmtree(tmp_path / "logs")
    (tmp_path / "logs").write_text("not a directory", encoding="utf-8")

    log.info("image.added", input="a.png")

    assert flush_records(5)
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
    assert flush_records(5)
    heard: list[int] = []
    set_stored_listener(lambda: heard.append(len(_records(database))))

    log.info("image.added", input="a.png")
    log.warning("open.ignored_directory", path="d")

    assert flush_records(5)
    # Called after the line is in the database, so a read it starts sees it.
    assert heard == [2, 3]


def test_a_line_that_went_to_the_fallback_file_tells_no_listener(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()
    configure_session_logging()
    heard: list[None] = []
    set_stored_listener(lambda: heard.append(None))

    log.info("image.added", input="a.png")

    assert flush_records(5)
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


def _existing_records(database: Path, version: int) -> bytes:
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE logs (anything TEXT)")
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()
    return database.read_bytes()


def test_the_records_database_is_stamped_at_creation(tmp_path: Path) -> None:
    database = configure_session_logging()
    log.info("image.added")
    assert flush_records(5)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        connection.close()


@pytest.mark.parametrize(("version", "reason"), [(0, "no format version"), (2, "format version 2")])
def test_unmarked_or_newer_records_are_left_untouched_and_lines_go_to_the_fallback_file(
    tmp_path: Path, version: int, reason: str
) -> None:
    database = tmp_path / RECORDS_FILE_NAME
    before = _existing_records(database, version)

    configure_session_logging()
    log.info("image.added", input="a.png")

    assert flush_records(5)
    assert database.read_bytes() == before
    assert not (tmp_path / f"{RECORDS_FILE_NAME}-wal").exists()
    assert flush_records(5)
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
        assert handler.flush_pending(5)
    finally:
        handler._connection = None
        connection.close()
        handler.close()
    line = _read_jsonl(tmp_path / "fallback.log")[0]
    assert "primary insert failure" in line["records_error"]
    assert "cleanup rollback failure" in line["records_error_details"]["traceback"]


def test_logging_never_waits_on_a_stuck_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # The caller only queues the line; a store stuck in its write holds up no one,
    # and the lines land in order once it moves again.
    import time

    from pixelup import session_log

    database = configure_session_logging()
    assert flush_records(5)
    entered = threading.Event()
    release = threading.Event()
    handler = session_log._records_handler()
    real_insert = handler._insert

    def stuck_insert(entry: dict) -> None:
        entered.set()
        assert release.wait(5)
        real_insert(entry)

    monkeypatch.setattr(handler, "_insert", stuck_insert)
    log.info("first.line")
    assert entered.wait(5)
    started = time.monotonic()
    log.info("second.line")
    log.info("third.line")
    assert time.monotonic() - started < 1
    assert flush_records(0.05) is False

    release.set()
    assert flush_records(5)
    assert [entry["message"] for entry in _records(database)][-3:] == [
        "first.line",
        "second.line",
        "third.line",
    ]


def test_a_full_queue_counts_what_it_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelup import session_log

    database = configure_session_logging()
    assert flush_records(5)
    entered = threading.Event()
    release = threading.Event()
    handler = session_log._records_handler()
    real_insert = handler._insert

    def stuck_insert(entry: dict) -> None:
        entered.set()
        assert release.wait(5)
        real_insert(entry)

    monkeypatch.setattr(handler, "_insert", stuck_insert)
    monkeypatch.setattr(session_log, "_PENDING_LIMIT", 2)
    log.info("held.line")
    assert entered.wait(5)
    for index in range(5):
        log.info("burst.line", index=index)
    release.set()
    assert flush_records(5)

    messages = [entry for entry in _records(database) if entry["message"] != "log.session_started"]
    assert [entry["message"] for entry in messages] == [
        "held.line",
        "burst.line",
        "burst.line",
        "log.dropped",
    ]
    assert messages[3]["count"] == 3


def test_a_field_changed_after_logging_is_recorded_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = configure_session_logging()
    values = {"items": [1]}
    log.info("snapshot.line", values=values)
    values["items"].append(2)

    assert _records(database)[-1]["values"] == {"items": [1]}
