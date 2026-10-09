"""window.ini: disposable window state with no format marker, where an unusable
value leaves the designed size in place."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray

from pixelup.window_placement import WindowState


def test_a_save_round_trips_the_value(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    WindowState(path).save("recordsWindow/listWidth", 300)

    text = path.read_text(encoding="utf-8")
    assert "listWidth=300" in text
    assert "formatVersion" not in text
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

    reread = WindowState(path)
    assert reread.value("mainWindow/geometry") == QByteArray(b"main-2")
    assert int(reread.value("recordsWindow/listWidth")) == 280
