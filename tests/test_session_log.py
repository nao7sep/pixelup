import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from pixelup.session_log import (
    RECORDS_FILE_NAME,
    configure_session_logging,
    current_session,
    debug_enabled,
    log,
    set_stored_listener,
)


def _records(database: Path) -> list[dict]:
    """Every log row as the line it stands for: columns and fields in one object."""
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


def test_every_record_carries_its_launch_session() -> None:
    database = configure_session_logging()
    log.info("first.launch")
    time.sleep(0.002)  # a session is named by its start millisecond
    configure_session_logging()
    log.info("second.launch")

    records = _records(database)
    first = {entry["session"] for entry in records[:2]}
    second = {entry["session"] for entry in records[2:]}
    assert len(first) == 1
    assert len(second) == 1
    assert first != second
    session = first.pop()
    assert session.endswith("Z")
    assert session <= records[0]["time"]


def test_domain_ids_are_columns() -> None:
    database = configure_session_logging()
    log.info("job.started", job_id=4)
    log.info("models.install_started", operation_id=2)

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
    shutil.rmtree(tmp_path / "logs")
    (tmp_path / "logs").write_text("not a directory", encoding="utf-8")

    log.info("image.added", input="a.png")

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
    heard: list[int] = []
    set_stored_listener(lambda: heard.append(len(_records(database))))

    log.info("image.added", input="a.png")
    log.warning("open.ignored_directory", path="d")

    # Called after the line is in the database, so a read it starts sees it.
    assert heard == [2, 3]


def test_a_line_that_went_to_the_fallback_file_tells_no_listener(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE_NAME).mkdir()
    configure_session_logging()
    heard: list[None] = []
    set_stored_listener(lambda: heard.append(None))

    log.info("image.added", input="a.png")

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
