from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pixelup.app_config import AppConfig, load_app_config_result, save_app_config
from pixelup.backup_store import STORE_FILE_NAME, close_backup_store
from pixelup.config import write_managed_text
from pixelup.jobs import JobSettings


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "home"
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    close_backup_store()
    from pixelup.config import resolve_state_dir

    return resolve_state_dir()


def _store_rows(home: Path, path: Path) -> list[tuple]:
    close_backup_store()
    connection = sqlite3.connect(home / STORE_FILE_NAME)
    try:
        return connection.execute(
            "SELECT content FROM backups WHERE path = ? ORDER BY id",
            (str(path),),
        ).fetchall()
    finally:
        connection.close()


def test_write_managed_text_records_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    write_managed_text(target, "hello\nworld\n")

    # The file landed atomically with no stray temp beside it...
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"
    assert list(home.glob("*.tmp")) == []
    # ...and the exact bytes were recorded.
    rows = _store_rows(home, target)
    assert len(rows) == 1
    assert bytes(rows[0][0]) == b"hello\nworld\n"


def test_record_fires_strictly_after_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # record() must see the file already at its final location, never before the
    # rename lands (data-backup-conventions): a backup of a save that never happened
    # is the one thing the net must not contain.
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    file_exists_when_recorded: list[bool] = []

    def fake_record(path: Path, data: bytes) -> None:
        file_exists_when_recorded.append(Path(path).exists())

    monkeypatch.setattr("pixelup.backup_store.record", fake_record)

    write_managed_text(target, "content\n")

    assert file_exists_when_recorded == [True]


def test_save_app_config_records_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The app's one RECORD write site end-to-end: saving config.json captures it.
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    for quality in (94, 94, 80):
        candidate = AppConfig(parameters=JobSettings(quality=quality))
        save_app_config(candidate, AppConfig(), load_app_config_result(target).file)

    rows = _store_rows(home, target)
    assert len(rows) == 2  # first save + the changed save; the identical one skipped


def test_a_failing_backup_never_breaks_the_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real composition: inject a store INSERT failure, then save through the
    # choke point. record() (the real one) catches and swallows it, so
    # write_managed_text returns normally and the file is on disk regardless — a
    # backup problem can never break the save that already succeeded.
    import sqlite3
    from typing import Any

    class _FailInsertConnection:
        # A delegating proxy over a real connection; execute raises on INSERT and
        # passes everything else through. (sqlite3's execute is read-only, so it
        # cannot be monkeypatched in place.)
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, sql: str, *rest: Any) -> Any:
            if sql.lstrip().upper().startswith("INSERT"):
                raise sqlite3.OperationalError("disk full (injected)")
            return self._real.execute(sql, *rest)

        def executescript(self, sql: str) -> Any:
            return self._real.executescript(sql)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"

    # Silence the one expected warn so it does not read as a test failure.
    monkeypatch.setattr("pixelup.backup_store.log.warning", lambda *a, **k: None)

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: _FailInsertConnection(real_connect(*args, **kwargs)),
    )

    # Must not raise, and the file must be on disk with the exact content.
    write_managed_text(target, "durable\n")

    assert target.read_text(encoding="utf-8") == "durable\n"


def test_a_failed_write_leaves_the_original_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path, monkeypatch)
    target = home / "config.json"
    write_managed_text(target, "original\n")

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("pixelup.config.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_managed_text(target, "replacement\n")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert list(home.glob("*.tmp")) == []
