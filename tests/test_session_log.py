import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from pixelup.session_log import (
    configure_session_logging,
    debug_enabled,
    get_logger,
    log,
    redact_log_fields,
    session_log_path,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_session_log_path_uses_utc_timestamp(tmp_path: Path) -> None:
    # Millisecond precision is pinned via a non-zero microsecond component:
    # 123456 us must truncate (not round) to 123 ms.
    path = session_log_path(
        state_dir=tmp_path,
        now=datetime(2026, 5, 7, 5, 43, 21, 123456, tzinfo=UTC),
    )

    assert path == tmp_path / "logs" / "20260507-054321-123-utc.log"


def test_configure_writes_jsonl_envelope(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    log.info("image.added", input="a.png", count=3)

    records = _read_jsonl(log_path)
    # The first line is the session-started banner; our event is the last line.
    assert records[0]["message"] == "log.session_started"
    entry = records[-1]
    assert entry["message"] == "image.added"
    assert entry["level"] == "info"
    assert entry["input"] == "a.png"
    assert entry["count"] == 3
    # time is the canonical UTC ISO-8601 millisecond instant with a trailing Z,
    # and it actually parses as an aware instant.
    assert entry["time"].endswith("Z")
    assert "T" in entry["time"]
    parsed = datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_warning_level_renders_as_warn(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    log.warning("open.ignored_non_file", path="x")

    entry = _read_jsonl(log_path)[-1]
    assert entry["level"] == "warn"


def test_logged_fields_are_redacted(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    log.info("auth.try", token="supersecret", user="bob")

    entry = _read_jsonl(log_path)[-1]
    assert entry["token"] == "[redacted]"
    assert entry["user"] == "bob"


def test_reserved_envelope_fields_cannot_be_overwritten(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    # `level` and `time` collide with the envelope; the formatter must keep the
    # real envelope rather than let a caller's field overwrite it. (`message` is
    # the positional argument, so it cannot be passed as a field at all.)
    log.info("real.event", level="bogus", time="bogus", detail="kept")

    entry = _read_jsonl(log_path)[-1]
    assert entry["message"] == "real.event"
    assert entry["level"] == "info"
    assert entry["time"] != "bogus"
    assert entry["detail"] == "kept"


def test_error_includes_type_message_and_cause_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    try:
        try:
            raise ValueError("root cause")
        except ValueError as inner:
            raise RuntimeError("wrapped failure") from inner
    except RuntimeError:
        log.exception("job.failed_unexpectedly", job_id=7)

    entry = _read_jsonl(log_path)[-1]
    assert entry["level"] == "error"
    assert entry["job_id"] == 7
    assert entry["error"]["type"] == "RuntimeError"
    assert entry["error"]["message"] == "wrapped failure"
    traceback_text = entry["error"]["traceback"]
    assert "RuntimeError: wrapped failure" in traceback_text
    # The formatted traceback carries the cause chain.
    assert "ValueError: root cause" in traceback_text


def test_non_serializable_field_values_do_not_break_a_line(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    # Path objects are common in PixelUp's payloads and are not natively JSON
    # serializable; the formatter must coerce them rather than raise.
    log.info("log.revealed", log_file=Path("/tmp/x.log"))

    entry = _read_jsonl(log_path)[-1]
    assert entry["log_file"] == str(Path("/tmp/x.log"))


def test_debug_enabled_reads_env() -> None:
    assert debug_enabled({}) is False
    assert debug_enabled({"PIXELUP_DEBUG": ""}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "0"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "false"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "off"}) is False
    assert debug_enabled({"PIXELUP_DEBUG": "1"}) is True
    assert debug_enabled({"PIXELUP_DEBUG": "yes"}) is True


def test_debug_is_suppressed_unless_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PIXELUP_DEBUG", raising=False)
    log_path = tmp_path / "logs" / "off.log"

    configure_session_logging(log_path)
    log.debug("job.progress", tick=1)

    messages = [entry["message"] for entry in _read_jsonl(log_path)]
    assert "job.progress" not in messages


def test_debug_is_emitted_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PIXELUP_DEBUG", "1")
    log_path = tmp_path / "logs" / "on.log"

    configure_session_logging(log_path)
    log.debug("job.progress", tick=1)

    messages = [entry["message"] for entry in _read_jsonl(log_path)]
    assert "job.progress" in messages


def test_configure_degrades_to_stderr_when_file_unavailable(tmp_path: Path) -> None:
    # A regular file where a directory is expected makes the logs dir uncreatable,
    # so configuration must fall back to stderr instead of failing the launch.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    log_path = blocker / "logs" / "session.log"

    returned = configure_session_logging(log_path)

    assert returned == log_path
    handlers = get_logger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert not isinstance(handlers[0], logging.FileHandler)


def test_configure_degrades_to_stderr_on_same_millisecond_clash(tmp_path: Path) -> None:
    # The session filename is millisecond-paced, so two processes can rarely
    # land on the same one. The file is opened exclusive-create, so the second
    # writer must degrade to the console fallback rather than interleave into
    # the first writer's file.
    log_path = tmp_path / "logs" / "session.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text('{"message": "from first process"}\n', encoding="utf-8")

    returned = configure_session_logging(log_path)
    log.info("second_process.event")

    assert returned == log_path
    handlers = get_logger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert not isinstance(handlers[0], logging.FileHandler)
    # The first writer's content is untouched — the second process never opened
    # (let alone truncated or appended into) the existing file.
    assert log_path.read_text(encoding="utf-8") == '{"message": "from first process"}\n'


def test_redact_masks_denied_keys_case_insensitively() -> None:
    out = redact_log_fields({"Token": "abc", "ApiKey": "k", "user": "bob"})
    assert out == {"Token": "[redacted]", "ApiKey": "[redacted]", "user": "bob"}


def test_redact_does_not_match_substrings() -> None:
    fields = {"tokenCount": 5, "broken": True, "password_hint": "ok"}
    assert redact_log_fields(fields) == fields


def test_redact_recurses_nested_mappings_and_lists() -> None:
    out = redact_log_fields(
        {"outer": {"secret": "s", "ok": 1}, "items": [{"password": "p"}, "raw"]}
    )
    assert out == {
        "outer": {"secret": "[redacted]", "ok": 1},
        "items": [{"password": "[redacted]"}, "raw"],
    }


def test_redact_leaves_scalars_untouched() -> None:
    assert redact_log_fields("token") == "token"
    assert redact_log_fields(42) == 42
    assert redact_log_fields(None) is None


def test_unserializable_field_never_drops_the_line(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    # A dict keyed by a tuple makes json.dumps raise (non-string key); the line
    # must still be written with the envelope and the field coerced, never
    # silently dropped by the logging machinery.
    log.info("weird.payload", counts={(1, 2): "pair"}, ok="kept")

    entry = _read_jsonl(log_path)[-1]
    assert entry["message"] == "weird.payload"
    assert entry["level"] == "info"
    assert entry["ok"] == "kept"
    assert entry["counts"] == {"(1, 2)": "pair"}


def test_caller_error_field_does_not_clobber_exception_payload(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "session.log"

    configure_session_logging(log_path)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("op.failed", error="caller-supplied")

    entry = _read_jsonl(log_path)[-1]
    # `error` is reserved for the exception payload; the caller's field is skipped.
    assert isinstance(entry["error"], dict)
    assert entry["error"]["type"] == "RuntimeError"


def test_excepthook_does_not_stack_across_reconfiguration(tmp_path: Path) -> None:
    # Two distinct paths, as two real sessions would each get: the log file is
    # now opened exclusive-create, so re-opening the same path would itself
    # degrade to the stderr fallback rather than exercise the property under
    # test here.
    first_log_path = tmp_path / "logs" / "first.log"
    second_log_path = tmp_path / "logs" / "second.log"

    # Reconfigure twice in one process; the excepthook must not chain onto a
    # previous copy of itself, or a single crash would be logged once per call.
    configure_session_logging(first_log_path)
    configure_session_logging(second_log_path)

    try:
        raise ValueError("synthetic crash")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    crashes = [e for e in _read_jsonl(second_log_path) if e["message"] == "unhandled.exception"]
    assert len(crashes) == 1
    assert crashes[0]["error"]["type"] == "ValueError"
