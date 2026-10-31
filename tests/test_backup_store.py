from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from pixelup.backup_store import STORE_FILE_NAME, close_backup_store, drain_backups, record
from pixelup.config import write_managed_text
from pixelup.session_log import _error_object, configure_session_logging, flush_records

# How long a test waits on its own threads before it fails instead of hanging.
_THREAD_WAIT_S = 10


class _FailInsertConnection:
    """A delegating proxy over a real sqlite3.Connection whose ``execute`` raises on
    an INSERT and passes everything else through.

    A plain ``connection.execute = ...`` monkeypatch is impossible — sqlite3's
    ``execute`` attribute is read-only on CPython — so failure is injected with this
    proxy instead. ``__getattr__`` delegates every other method (commit, close, …)
    to the wrapped real connection.
    """

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, *rest: Any) -> Any:
        if sql.lstrip().upper().startswith("INSERT"):
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return self._real.execute(sql, *rest)

    def executescript(self, sql: str) -> Any:
        return self._real.executescript(sql)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session: str = "session-1") -> Path:
    """Redirect PIXELUP_DATA_DIR to a throwaway root and return the resolved root.

    The autouse conftest fixture closes the store after each test, so it re-opens
    under this root; belt-and-suspenders, close it here too before the first record
    so we never inherit a prior test's open handle. The session is pinned so a test
    can start a new launch with :func:`_relaunch`.
    """
    root = tmp_path / "home"
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    close_backup_store()
    monkeypatch.setattr("pixelup.backup_store.current_session", lambda: session)
    from pixelup.config import resolve_state_dir

    return resolve_state_dir()


def _relaunch(monkeypatch: pytest.MonkeyPatch, session: str) -> None:
    """End this launch's recorder and start the next one under ``session``."""
    close_backup_store()
    monkeypatch.setattr("pixelup.backup_store.current_session", lambda: session)


def _store_path(home: Path) -> Path:
    return home / STORE_FILE_NAME


def _rows(home: Path, path: Path) -> list[tuple]:
    """Read every backups row for `path`, oldest first, straight from the file.

    Opens its own read-only connection (the writer's singleton is closed first) so
    the assertion sees exactly what landed on disk.
    """
    close_backup_store()
    connection = sqlite3.connect(_store_path(home))
    try:
        return connection.execute(
            "SELECT path, content, content_sha256, byte_size, written_at_utc, session_id"
            " FROM backups WHERE path = ? ORDER BY id",
            (str(path),),
        ).fetchall()
    finally:
        connection.close()


def _all_rows(home: Path) -> list[tuple]:
    close_backup_store()
    connection = sqlite3.connect(_store_path(home))
    try:
        return connection.execute("SELECT path FROM backups ORDER BY id").fetchall()
    finally:
        connection.close()


def test_content_blob_is_byte_identical_including_crlf_and_non_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    # A CR/LF line ending, a UTF-8 BOM, and a lone 0x80 byte that is NOT valid
    # UTF-8: the whole point of a BLOB is that none of these are normalized,
    # dropped, or corrupted the way a decoded-string round-trip would.
    payload = b'\xef\xbb\xbf{\r\n  "quality": 95\r\n}\x80'

    record(target, payload)

    rows = _rows(home, target)
    assert len(rows) == 1
    stored_path, content, digest, byte_size, written_at, session = rows[0]
    assert stored_path == str(target)  # full absolute path, one representation
    assert bytes(content) == payload
    assert byte_size == len(payload)
    assert session == "session-1"
    import hashlib

    assert digest == hashlib.sha256(payload).hexdigest()


def test_written_at_utc_is_serialized_iso_ms_not_the_filename_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    record(target, b"{}\n")

    written_at = _rows(home, target)[0][4]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", written_at)
    assert "-utc" not in written_at


def test_one_session_keeps_one_row_per_file_with_its_last_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    record(target, b'{"quality": 95}\n')
    record(target, b'{"quality": 80}\n')
    record(target, b'{"quality": 70}\n')

    rows = _rows(home, target)
    assert [bytes(row[1]) for row in rows] == [b'{"quality": 70}\n']


def test_each_session_keeps_its_own_row_and_never_changes_earlier_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    record(target, b"first launch\n")
    _relaunch(monkeypatch, "session-2")
    record(target, b"second launch\n")
    record(target, b"second launch, later\n")

    rows = _rows(home, target)
    assert [(bytes(row[1]), row[5]) for row in rows] == [
        (b"first launch\n", "session-1"),
        (b"second launch, later\n", "session-2"),
    ]


def test_a_first_save_equal_to_the_latest_earlier_row_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    record(target, b"unchanged\n")
    _relaunch(monkeypatch, "session-2")
    record(target, b"unchanged\n")

    assert [row[5] for row in _rows(home, target)] == ["session-1"]


def test_the_session_row_follows_a_revert_within_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    record(target, b"original\n")
    _relaunch(monkeypatch, "session-2")
    record(target, b"edited\n")
    drain_backups()  # landed before the revert is saved
    record(target, b"original\n")  # back to the earlier session's content

    rows = _rows(home, target)
    assert [(bytes(row[1]), row[5]) for row in rows] == [
        (b"original\n", "session-1"),
        (b"original\n", "session-2"),
    ]


def test_record_returns_without_waiting_for_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The save never waits on the history: a store stuck in its write leaves
    # record() free to return, and the write lands once the store moves again.
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    entered = threading.Event()
    release = threading.Event()
    real_write = __import__("pixelup.backup_store", fromlist=["_write"])._write

    def blocked_write(*args: Any) -> None:
        entered.set()
        assert release.wait()
        real_write(*args)

    monkeypatch.setattr("pixelup.backup_store._write", blocked_write)
    record(target, b"first\n")
    assert entered.wait()

    record(target, b"second\n")  # returns although the recorder is stuck
    record(target, b"third\n")
    from pixelup import backup_store

    # Every save waits its turn, in save order.
    assert list(backup_store._recorder._pending) == [
        (str(target), b"second\n"),
        (str(target), b"third\n"),
    ]

    release.set()
    drain_backups()
    assert [bytes(row[1]) for row in _rows(home, target)] == [b"third\n"]


def test_writes_from_many_threads_land_as_one_session_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def save() -> None:
        try:
            barrier.wait()
            record(target, b"same\n")
        except Exception as exc:  # noqa: BLE001 - reported by the test thread below.
            errors.append(exc)

    threads = [threading.Thread(target=save) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not any(thread.is_alive() for thread in threads), "a save never finished"
    assert errors == []
    assert len(_rows(home, target)) == 1


def test_two_paths_keep_rows_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    a = home / "config.json"
    b = home / "other.json"
    record(a, b"{}\n")
    record(b, b"{}\n")

    assert len(_rows(home, a)) == 1
    assert len(_rows(home, b)) == 1
    assert len(_all_rows(home)) == 2


def test_record_failure_is_one_warn_and_never_reaches_the_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    warns: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "pixelup.backup_store.log.warning",
        lambda message, **fields: warns.append((message, fields)),
    )
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: _FailInsertConnection(real_connect(*args, **kwargs)),
    )

    record(target, b'{"quality": 95}\n')  # must not raise
    record(target, b'{"quality": 80}\n')  # recording is off for the session
    close_backup_store()

    assert [message for message, _fields in warns] == ["backup_store.record_failed"]
    assert warns[0][1]["file"] == str(target)
    assert "injected" in warns[0][1]["reason"]


def test_open_failure_disables_recording_with_one_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    warns: list[str] = []
    monkeypatch.setattr(
        "pixelup.backup_store.log.warning",
        lambda message, **fields: warns.append(message),
    )
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("no open")),
    )

    record(target, b"{}\n")
    record(target, b"different\n")
    close_backup_store()

    assert warns == ["backup_store.open_failed"]


@pytest.mark.parametrize("stage", ["open", "record"])
def test_backup_failure_keeps_full_diagnostics_without_failing_the_primary_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    database = configure_session_logging()

    def fail(*_args) -> None:
        try:
            raise ValueError("EIO /internal/backups ORIGINAL-BACKUP-SENTINEL")
        except ValueError as cause:
            raise RuntimeError("backup storage failed") from cause

    monkeypatch.setattr(
        "pixelup.backup_store." + ("_open_store" if stage == "open" else "_write"), fail
    )

    write_managed_text(target, "durable\n")
    drain_backups()
    write_managed_text(target, "still durable\n")

    assert target.read_bytes() == b"still durable\n"
    drain_backups()
    flush_records()
    with sqlite3.connect(database) as connection:
        entries = connection.execute(
            "SELECT level, fields FROM logs WHERE message = ?",
            (f"backup_store.{stage}_failed",),
        ).fetchall()
    assert len(entries) == 1
    level, fields = entries[0]
    assert level == "warn"
    error = json.loads(fields)["error"]
    assert error["type"] == "RuntimeError"
    assert error["message"] == "backup storage failed"
    assert "ValueError: EIO /internal/backups ORIGINAL-BACKUP-SENTINEL" in error["traceback"]
    assert "in fail" in error["traceback"]


@pytest.mark.parametrize("stage", ["open", "record"])
@pytest.mark.parametrize("warning_fails", [False, True])
def test_backup_failure_disables_and_drains_with_unavailable_root_or_failed_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, warning_fails: bool
) -> None:
    from pixelup import backup_store

    home = _home(tmp_path, monkeypatch)
    displaced = home.with_name(home.name + "-displaced")
    target = home / "config.json"
    entered, release = threading.Event(), threading.Event()
    real_open = backup_store._open_store
    warnings = []
    recorder = backup_store._recorder

    if stage == "record":
        # The failed insert has an open real SQLite owner. Keep it off the volume
        # being disconnected: Windows cannot rename a directory with open files.
        store_owner = tmp_path / "open-backup-owner"
        store_owner.mkdir()
        monkeypatch.setattr(backup_store, "_store_file", lambda: store_owner / STORE_FILE_NAME)

    def held_failure(*args):
        entered.set()
        assert release.wait()
        if stage == "open":
            return real_open()
        try:
            raise OSError("EIO /internal/backups ORIGINAL-BACKUP-FAILURE-SENTINEL")
        except OSError as cause:
            raise RuntimeError("backup write failed") from cause

    def warning(message, **fields):
        exc_info = fields.get("exc_info")
        error = _error_object(exc_info if isinstance(exc_info, tuple) else sys.exc_info())
        warnings.append((message, fields, error))
        assert not recorder._condition._is_owned()
        if warning_fails:
            raise RuntimeError("optional warning failed too")

    monkeypatch.setattr(backup_store, "_open_store" if stage == "open" else "_write", held_failure)
    monkeypatch.setattr(backup_store.log, "warning", warning)
    try:
        write_managed_text(target, "landed primary save\n")
        assert entered.wait()
        record(target, b"newest pending version")
        home.rename(displaced)
        home.write_text("no longer a directory", encoding="utf-8")
        release.set()
        assert recorder._thread is not None
        recorder._thread.join()

        assert not recorder._thread.is_alive()
        assert recorder._disabled
        assert not recorder._pending
        assert not recorder._writing
        drain_backups()
        record(target, b"abandoned after failure")
        assert not recorder._pending
        assert (displaced / target.name).read_bytes() == b"landed primary save\n"
        assert len(warnings) == 1
        message, fields, error = warnings[0]
        assert message == f"backup_store.{stage}_failed"
        assert fields["file"] == str(target)
        if stage == "open":
            assert error["type"] == "PixelupError"
            assert "FileExistsError" in error["traceback"]
            assert str(home) in error["traceback"]
        else:
            assert error["type"] == "RuntimeError"
            assert error["message"] == "backup write failed"
            assert "OSError: EIO /internal/backups ORIGINAL-BACKUP-FAILURE-SENTINEL" in error[
                "traceback"
            ]
    finally:
        release.set()
        if recorder._thread is not None:
            recorder._thread.join()
        with recorder._condition:
            recorder._disabled = True
            recorder._pending.clear()
            recorder._condition.notify_all()
        if displaced.exists():
            home.unlink()
            displaced.rename(home)


def test_store_sidecars_are_the_stores_own_wal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    record(home / "config.json", b"{}\n")
    close_backup_store()

    store_files = {p.name for p in home.glob("backups.sqlite3*")}
    assert "backups.sqlite3" in store_files
    assert store_files <= {"backups.sqlite3", "backups.sqlite3-wal", "backups.sqlite3-shm"}


def _user_version(file: Path) -> int:
    connection = sqlite3.connect(file)
    try:
        return connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()


def test_a_new_store_records_its_format_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    record(home / "config.json", b"{}\n")
    close_backup_store()
    assert _user_version(_store_path(home)) == 1


@pytest.mark.parametrize("version", [0, 2])
def test_an_unmarked_or_newer_store_is_left_untouched_with_one_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    # A user_version of 0 on a database that has tables is a missing marker.
    home = _home(tmp_path, monkeypatch)
    store = _store_path(home)
    connection = sqlite3.connect(store)
    connection.execute("CREATE TABLE backups (anything TEXT)")
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()
    before = store.read_bytes()
    warns: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "pixelup.backup_store.log.warning",
        lambda message, **fields: warns.append((message, fields)),
    )

    record(home / "config.json", b"{}\n")
    record(home / "config.json", b"other\n")
    close_backup_store()

    assert [message for message, _fields in warns] == ["backup_store.open_failed"]
    assert store.read_bytes() == before
    assert not store.with_name(f"{STORE_FILE_NAME}-wal").exists()
