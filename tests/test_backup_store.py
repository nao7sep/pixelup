from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from pixelup.backup_store import STORE_FILE_NAME, close_backup_store, record


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


class _RecordingConnection:
    def __init__(self, real: sqlite3.Connection, statements: list[str]) -> None:
        self._real = real
        self._statements = statements

    def execute(self, sql: str, *rest: Any) -> Any:
        self._statements.append(sql.strip())
        return self._real.execute(sql, *rest)

    def executescript(self, sql: str) -> Any:
        return self._real.executescript(sql)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect PIXELUP_HOME to a throwaway root and return the resolved root.

    The autouse conftest fixture closes the store singleton after each test, so it
    re-opens under this root; belt-and-suspenders, close it here too before the
    first record so we never inherit a prior test's open handle.
    """
    root = tmp_path / "home"
    monkeypatch.setenv("PIXELUP_HOME", str(root))
    close_backup_store()
    from pixelup.config import resolve_state_dir

    return resolve_state_dir()


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
            "SELECT path, content, content_sha256, byte_size, written_at_utc"
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
    payload = b"\xef\xbb\xbf{\r\n  \"quality\": 95\r\n}\x80"

    record(target, payload)

    rows = _rows(home, target)
    assert len(rows) == 1
    stored_path, content, digest, byte_size, written_at = rows[0]
    assert stored_path == str(target)  # full absolute path, one representation
    # sqlite returns a BLOB as bytes; it must equal the input byte-for-byte.
    assert bytes(content) == payload
    assert byte_size == len(payload)
    # sha256 is over the raw bytes, lowercase hex.
    import hashlib

    assert digest == hashlib.sha256(payload).hexdigest()


def test_written_at_utc_is_serialized_iso_ms_not_the_filename_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    record(target, b"{}\n")

    (_, _, _, _, written_at) = _rows(home, target)[0]
    # Serialized ISO-8601 UTC with milliseconds and a trailing Z, e.g.
    # 2026-07-06T04:05:12.345Z — the timestamp-conventions' data form.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", written_at)
    # And explicitly NOT the yyyymmdd-hhmmss-fff-utc filename stamp.
    assert not re.fullmatch(r"\d{8}-\d{6}-\d{3}-utc", written_at)
    assert "-utc" not in written_at


def test_dedup_skips_an_unchanged_re_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    payload = b'{"quality": 95}\n'

    record(target, payload)
    record(target, payload)  # identical bytes -> no new row

    assert len(_rows(home, target)) == 1


def test_a_changed_save_inserts_a_new_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    record(target, b'{"quality": 95}\n')
    record(target, b'{"quality": 80}\n')  # genuinely different -> new row

    rows = _rows(home, target)
    assert len(rows) == 2
    assert bytes(rows[0][1]) == b'{"quality": 95}\n'
    assert bytes(rows[1][1]) == b'{"quality": 80}\n'


def test_latest_check_and_insert_are_one_immediate_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    statements: list[str] = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: _RecordingConnection(
            real_connect(*args, **kwargs), statements
        ),
    )

    record(home / "config.json", b"{}\n")

    transaction = next(i for i, sql in enumerate(statements) if sql == "BEGIN IMMEDIATE")
    latest = next(i for i, sql in enumerate(statements) if sql.startswith("SELECT content_sha256"))
    insert = next(i for i, sql in enumerate(statements) if sql.startswith("INSERT INTO backups"))
    assert transaction < latest < insert


def test_concurrent_in_process_records_share_one_connection_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    barrier = threading.Barrier(8)

    def save() -> None:
        barrier.wait()
        record(target, b"same\n")

    threads = [threading.Thread(target=save) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(_rows(home, target)) == 1


def test_a_revert_inserts_a_new_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    original = b'{"quality": 95}\n'
    edited = b'{"quality": 80}\n'

    record(target, original)
    record(target, edited)
    record(target, original)  # content returns to an earlier value

    # Dedup compares only against the IMMEDIATELY preceding row, so a revert
    # differs from `edited` and is recorded as the new version it is: 3 rows.
    rows = _rows(home, target)
    assert len(rows) == 3
    assert [bytes(r[1]) for r in rows] == [original, edited, original]


def test_two_paths_dedup_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    a = home / "config.json"
    b = home / "other.json"
    same = b"{}\n"

    record(a, same)
    record(b, same)  # same bytes, different path -> its own first row

    assert len(_rows(home, a)) == 1
    assert len(_rows(home, b)) == 1
    assert len(_all_rows(home)) == 2


def test_record_is_best_effort_no_throw_one_warn_save_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject a store failure at the insert: record() must catch it, log exactly one
    # warn, and return normally — never propagate to the caller (which is a save
    # that already succeeded on disk before record ran).
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

    # Must not raise.
    record(target, b'{"quality": 95}\n')

    # Exactly one warn, naming the file and a reason.
    assert len(warns) == 1
    message, fields = warns[0]
    assert message == "backup_store.record_failed"
    assert fields["file"] == str(target)
    assert "injected" in fields["reason"]


def test_open_failure_disables_recording_with_one_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the store can't even be opened, record() logs ONE warn, disables recording
    # for the session, and every later record() is a silent no-op (no re-warn).
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

    record(target, b"{}\n")  # first attempt: open fails, one warn
    record(target, b"different\n")  # second attempt: already disabled, silent

    assert warns == ["backup_store.open_failed"]


def test_store_sidecars_are_the_stores_own_wal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store-file filter a test uses when it inspects the throwaway root: the
    # store is backups.sqlite3 plus its normal WAL sidecars (-wal, -shm). A test
    # asserting the root's managed-text contents would exclude exactly this set.
    home = _home(tmp_path, monkeypatch)
    record(home / "config.json", b"{}\n")
    close_backup_store()

    store_files = {p.name for p in home.glob("backups.sqlite3*")}
    # backups.sqlite3 always; -wal/-shm may or may not linger after close, but any
    # backups.sqlite3* file present is a store artifact, never stray debris.
    assert "backups.sqlite3" in store_files
    assert store_files <= {"backups.sqlite3", "backups.sqlite3-wal", "backups.sqlite3-shm"}
