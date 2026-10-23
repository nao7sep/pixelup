"""window.ini: disposable window state with no format marker, where an unusable
value leaves the designed size in place."""

from __future__ import annotations

import weakref
from pathlib import Path
from threading import Event, Thread, get_ident

from PySide6.QtCore import QByteArray

from pixelup import window_placement
from pixelup.window_placement import WindowState


def test_a_save_round_trips_the_value(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    state = WindowState(path)
    state.save("recordsWindow/listWidth", 300)
    _settle(state)

    text = path.read_text(encoding="utf-8")
    assert "listWidth=300" in text
    assert int(WindowState(path).value("recordsWindow/listWidth")) == 300


def test_no_file_reads_as_nothing_saved(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    assert WindowState(path).value("recordsWindow/listWidth") is None
    assert not path.exists()


def test_a_file_with_an_old_marker_is_still_read_and_saved(tmp_path: Path) -> None:
    # Files written while window.ini carried a marker keep working; the inert key
    # is left alone rather than cleaned up.
    path = tmp_path / "window.ini"
    path.write_text("[General]\nformatVersion=1\n\n[recordsWindow]\nlistWidth=320\n")

    state = WindowState(path)
    assert int(state.value("recordsWindow/listWidth")) == 320
    state.save("mainWindow/geometry", QByteArray(b"geometry"))
    _settle(state)

    assert WindowState(path).value("mainWindow/geometry") == QByteArray(b"geometry")
    assert int(WindowState(path).value("recordsWindow/listWidth")) == 320


def test_two_states_on_one_file_keep_each_others_keys(tmp_path: Path) -> None:
    # The main window and the Records window each hold a WindowState on the same
    # file; a save through one never drops what the other saved.
    path = tmp_path / "window.ini"
    main = WindowState(path)
    records = WindowState(path)

    main.save("mainWindow/geometry", QByteArray(b"main"))
    records.save("recordsWindow/listWidth", 280)
    main.save("mainWindow/geometry", QByteArray(b"main-2"))
    _settle(main)

    reread = WindowState(path)
    assert reread.value("mainWindow/geometry") == QByteArray(b"main-2")
    assert int(reread.value("recordsWindow/listWidth")) == 280



def _settle(state: WindowState) -> None:
    # Tests may wait for the physical writes; window close never does.
    for thread in list(state._writer._threads):
        thread.join(5)
        assert not thread.is_alive()
    with state._writer._lock:
        assert not state._writer._running
        assert not state._writer._pending


def test_blocked_sync_does_not_block_save_and_coalesces_shared_state(
    tmp_path: Path, monkeypatch
) -> None:
    real_settings = window_placement.QSettings
    entered = Event()
    release = Event()
    writes = []
    storage_threads = []
    gui_thread = get_ident()

    class Settings:
        Format = real_settings.Format
        Status = real_settings.Status

        def __init__(self, *args):
            self.inner = real_settings(*args)
            self.writer_thread = get_ident()

        def value(self, key):
            return self.inner.value(key)

        def status(self):
            return self.inner.status()

        def setValue(self, key, value):
            storage_threads.append((self.writer_thread, get_ident()))
            writes.append((key, value))
            self.inner.setValue(key, value)

        def sync(self):
            entered.set()
            assert release.wait(5)
            self.inner.sync()

    monkeypatch.setattr(window_placement, "QSettings", Settings)
    path = tmp_path / "window.ini"
    main = WindowState(path)
    records = WindowState(path.parent / "nested" / ".." / path.name)
    assert main._writer is records._writer
    main.save("geometry", QByteArray(b"initial"))
    try:
        assert entered.wait(5)
        # sync is held, so these calls prove that the GUI does not wait for it.
        records.save("width", 280)
        mutable = QByteArray(b"latest")
        main.save("geometry", QByteArray(b"obsolete"))
        main.save("geometry", mutable)
        mutable.append(b"-changed")
    finally:
        release.set()
    _settle(main)
    reread = WindowState(path)
    assert reread.value("geometry") == QByteArray(b"latest")
    assert int(reread.value("width")) == 280
    assert [value for key, value in writes if key == "geometry"] == [
        QByteArray(b"initial"), QByteArray(b"latest"),
    ]
    assert all(created == called != gui_thread for created, called in storage_threads)


def test_submission_during_physical_thread_retirement_is_not_lost(tmp_path: Path, monkeypatch):
    retired = Event()
    release = Event()
    started = []

    class RetiringThread(Thread):
        def run(self):
            first = not started
            started.append(self)
            super().run()
            if first:
                retired.set()
                assert release.wait(5)

    monkeypatch.setattr(window_placement, "Thread", RetiringThread)
    path = tmp_path / "window.ini"
    state = WindowState(path)
    state.save("width", 100)
    try:
        assert retired.wait(5)
        state.save("width", 200)
        # The old physical thread remains owned while the new writer starts.
        assert started[0] in state._writer._threads
        assert started[0].is_alive()
    finally:
        release.set()
    _settle(state)
    assert int(WindowState(path).value("width")) == 200


def test_idle_path_writer_is_not_permanently_registered(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    state = WindowState(path)
    writer = weakref.ref(state._writer)
    state.save("width", 100)
    _settle(state)
    del state
    assert writer() is None
    assert path not in window_placement._WRITERS


def test_active_writer_survives_its_window_states(tmp_path: Path, monkeypatch) -> None:
    entered = Event()
    release = Event()
    write = window_placement._WindowStateWriter._write

    def blocked_write(writer, pending):
        entered.set()
        assert release.wait(5)
        write(writer, pending)

    monkeypatch.setattr(window_placement._WindowStateWriter, "_write", blocked_write)
    path = tmp_path / "window.ini"
    state = WindowState(path)
    reference = weakref.ref(state._writer)
    state.save("width", 100)
    try:
        assert entered.wait(5)
        del state
        assert reference() is not None
        replacement = WindowState(path)
        assert replacement._writer is reference()
        replacement.save("width", 200)
    finally:
        release.set()
    _settle(replacement)
    assert int(WindowState(path).value("width")) == 200
