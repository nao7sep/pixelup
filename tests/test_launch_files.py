"""The launch check: what each file PixelUp opens to start does when refused
(store-recovery-conventions, What a refused file does)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from config_support import header
from PySide6.QtWidgets import QApplication

from pixelup import gui
from pixelup.backup_store import (
    STORE_FILE_NAME,
    backups_off,
    close_backup_store,
    drain_backups,
    record,
    switch_off_backups,
)
from pixelup.formats import RECORDS_FORMAT, RefusedFile, RefusedFilesError, open_sqlite_store
from pixelup.i18n.message import Message
from pixelup.launch_files import check_launch_files, refused_files_message
from pixelup.session_log import (
    RECORDS_FILE_NAME,
    RecordsOffError,
    configure_session_logging,
    flush_records,
    log,
    records_off,
    records_store,
)


def _newer_store(path: Path) -> None:
    """A store a newer PixelUp wrote: the current header with a higher version."""
    open_sqlite_store(path, RECORDS_FORMAT, "CREATE TABLE logs (id INTEGER)").close()
    with sqlite3.connect(path) as connection:
        if path.name == STORE_FILE_NAME:
            connection.execute("UPDATE file_header SET value = 'backups' WHERE key = 'format.kind'")
        connection.execute("PRAGMA user_version = 2")
    connection.close()


def _files(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


@pytest.mark.parametrize("name", [RECORDS_FILE_NAME, STORE_FILE_NAME])
def test_a_newer_side_store_stops_the_launch_untouched(tmp_path: Path, name: str) -> None:
    _newer_store(tmp_path / name)
    before = _files(tmp_path)

    with pytest.raises(RefusedFilesError) as raised:
        check_launch_files()

    assert raised.value.files == (RefusedFile(tmp_path / name, newer=True),)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("name", [RECORDS_FILE_NAME, STORE_FILE_NAME])
@pytest.mark.parametrize("content", [b"not a database", b"SQLite format 3\x00" + b"\x00" * 200])
def test_an_unreadable_side_store_is_switched_off_untouched(
    tmp_path: Path, name: str, content: bytes
) -> None:
    (tmp_path / name).write_bytes(content)

    launch = check_launch_files()

    assert (launch.records_off, launch.backups_off) == (
        name == RECORDS_FILE_NAME,
        name == STORE_FILE_NAME,
    )
    assert (tmp_path / name).read_bytes() == content


def test_a_store_without_a_header_is_switched_off(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / RECORDS_FILE_NAME) as connection:
        connection.execute("CREATE TABLE logs (id INTEGER)")
    connection.close()
    assert check_launch_files().records_off


def test_every_refused_file_is_named_at_once(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{ broken", encoding="utf-8")
    (tmp_path / "parameters.json").write_text(
        json.dumps(header("parameters", 2)), encoding="utf-8"
    )
    _newer_store(tmp_path / RECORDS_FILE_NAME)
    (tmp_path / STORE_FILE_NAME).write_bytes(b"not a database")
    before = _files(tmp_path)

    with pytest.raises(RefusedFilesError) as raised:
        check_launch_files()

    # An unreadable backup store alone would only switch it off; it is not named.
    assert raised.value.files == (
        RefusedFile(tmp_path / "config.json", newer=False),
        RefusedFile(tmp_path / "parameters.json", newer=True),
        RefusedFile(tmp_path / RECORDS_FILE_NAME, newer=True),
    )
    assert _files(tmp_path) == before


def test_an_ordinary_launch_refuses_nothing(tmp_path: Path) -> None:
    launch = check_launch_files()
    assert (launch.records_off, launch.backups_off) == (False, False)
    # Reading creates no file; resolving the root only prepares its models folder.
    assert [path.name for path in tmp_path.iterdir()] == ["models"]


def test_main_stops_with_every_refused_file_named_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    (tmp_path / "config.json").write_text("{ broken", encoding="utf-8")
    _newer_store(tmp_path / RECORDS_FILE_NAME)
    before = _files(tmp_path)
    monkeypatch.setattr(gui, "MainWindow", lambda **_: pytest.fail("opened the window"))
    shown: list[tuple[Message, Message | None]] = []
    monkeypatch.setattr(
        gui, "show_startup_failure", lambda detail, hint: shown.append((detail, hint))
    )

    assert gui.main() == 1

    assert shown == [
        (
            refused_files_message(
                (
                    RefusedFile(tmp_path / "config.json", newer=False),
                    RefusedFile(tmp_path / RECORDS_FILE_NAME, newer=True),
                )
            ),
            Message("startup.refusedHint"),
        )
    ]
    # Only the writer claim exists beside the refused files: no records, logs,
    # backups or window state were written before stopping.
    assert {path.name for path in tmp_path.iterdir()} == {
        "config.json",
        RECORDS_FILE_NAME,
        "models",
        "writer.lock",
    }
    after = _files(tmp_path)
    del after["writer.lock"]
    assert after == before


def test_records_off_writes_lines_to_the_fallback_and_never_opens_the_store(
    tmp_path: Path,
) -> None:
    store = tmp_path / RECORDS_FILE_NAME
    store.write_bytes(b"not a database")

    configure_session_logging(records_off=True)
    log.info("probe.line", value=1)
    flush_records()
    failures: list[Exception | None] = []
    records_store().read(lambda connection: None, lambda _result, error: failures.append(error))
    flush_records()

    assert records_off()
    assert store.read_bytes() == b"not a database"
    (fallback,) = (tmp_path / "logs").iterdir()
    lines = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines()]
    assert [line["message"] for line in lines] == ["log.session_started", "probe.line"]
    assert len(failures) == 1 and isinstance(failures[0], RecordsOffError)


def test_backups_off_never_opens_the_store_through_a_session_of_saves(tmp_path: Path) -> None:
    store = tmp_path / STORE_FILE_NAME
    store.write_bytes(b"not a database")
    switch_off_backups()

    for text in (b"{}\n", b"{\"a\": 1}\n", b"{\"a\": 2}\n"):
        record(tmp_path / "config.json", text)
    drain_backups()

    assert backups_off()
    assert store.read_bytes() == b"not a database"
    close_backup_store()
    assert not backups_off()


def test_the_window_says_once_which_side_stores_are_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    from pixelup.runner import JobRunner

    monkeypatch.setattr(JobRunner, "schedule", lambda self: None)
    notices: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        gui, "warn_side_store_off", lambda _parent, key, path: notices.append((key, path))
    )
    configure_session_logging(records_off=True)
    switch_off_backups()

    window = gui.MainWindow()
    try:
        qapp.processEvents()
        assert notices == [
            ("notice.recordsOff", tmp_path.resolve() / RECORDS_FILE_NAME),
            ("notice.backupsOff", tmp_path.resolve() / STORE_FILE_NAME),
        ]
    finally:
        window._is_session_shutdown = lambda: True
        window.close()
        window.deleteLater()
        qapp.processEvents()
