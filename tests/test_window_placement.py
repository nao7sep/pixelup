"""window.ini's format version: written with every save, a newer file left
exactly as it is, and a missing or unusable marker reset."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QByteArray

from pixelup.window_placement import FORMAT_VERSION_KEY, WindowState


def _ini(path: Path, text: str) -> bytes:
    path.write_text(text, encoding="utf-8")
    return path.read_bytes()


def test_a_save_writes_the_format_version_beside_the_value(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    WindowState(path).save("recordsWindow/listWidth", 300)

    text = path.read_text(encoding="utf-8")
    assert f"[General]\n{FORMAT_VERSION_KEY}=1\n" in text
    assert "listWidth=300" in text
    assert int(WindowState(path).value("recordsWindow/listWidth")) == 300


def test_no_file_reads_as_nothing_saved(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    assert WindowState(path).value("recordsWindow/listWidth") is None
    assert not path.exists()


def test_a_newer_file_is_neither_read_nor_written(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    before = _ini(path, "[General]\nformatVersion=2\n\n[recordsWindow]\nlistWidth=320\n")

    state = WindowState(path)
    state.save("recordsWindow/listWidth", 400)
    state.save("mainWindow/geometry", QByteArray(b"geometry"))

    assert state.value("recordsWindow/listWidth") is None
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["window.ini"]


@pytest.mark.parametrize(
    "general",
    [
        "",
        "[General]\nformatVersion=one\n",
        "[General]\nformatVersion=0\n",
        "[General]\nformatVersion=-1\n",
        "[General]\nformatVersion=1.5\n",
    ],
)
def test_a_missing_or_unusable_format_version_resets_the_state(
    tmp_path: Path, general: str
) -> None:
    path = tmp_path / "window.ini"
    _ini(path, f"{general}\n[recordsWindow]\nlistWidth=320\n")

    state = WindowState(path)
    assert state.value("recordsWindow/listWidth") is None

    state.save("recordsWindow/listWidth", 400)
    text = path.read_text(encoding="utf-8")
    assert f"{FORMAT_VERSION_KEY}=1\n" in text
    assert "listWidth=400" in text
    assert "listWidth=320" not in text


@pytest.mark.parametrize("initial", ["[General]\nformatVersion=1\n", "[state]\nold=1\n"])
def test_save_rechecks_a_newer_file_after_construction(tmp_path: Path, initial: str) -> None:
    path = tmp_path / "window.ini"
    _ini(path, initial)
    state = WindowState(path)
    before = _ini(path, "[General]\nformatVersion=2\n\n[state]\nnew=kept\n")

    state.save("state/old", "replacement")
    del state

    assert path.read_bytes() == before


def test_loading_invalid_state_never_defers_a_destructive_flush(tmp_path: Path) -> None:
    path = tmp_path / "window.ini"
    _ini(path, "[state]\nold=1\n")
    state = WindowState(path)
    assert state.value("state/old") is None
    before = _ini(path, "[General]\nformatVersion=2\n\n[state]\nnew=kept\n")

    del state

    assert path.read_bytes() == before
