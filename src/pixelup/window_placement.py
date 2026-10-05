"""Where a durable window reopens: its native geometry blob in window.ini
(window-conventions, Placement implementation, Qt Widgets and PySide).

The main window and the Records window each keep one blob under their own key,
restored before the first frame and saved on every accepted close. Qt's own
restoreGeometry owns display-scale changes and moves a window that would land
off-screen back onto one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import QWidget

from pixelup.config import window_settings_path
from pixelup.formats import WINDOW_STATE_FORMAT_VERSION, format_version
from pixelup.session_log import log

FORMAT_VERSION_KEY = "formatVersion"


class WindowState:
    """window.ini: the disposable window state, placement and pane widths, under
    one format version (store-recovery-conventions).

    A file a newer PixelUp wrote is left exactly as it is: nothing is read from it
    or written to it, so the windows open at their designed sizes. A marker that is
    not a version makes the file's contents unusable, and it is reset.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._settings = QSettings(str(path), QSettings.Format.IniFormat)
        self._newer = False
        stored = self._settings.value(FORMAT_VERSION_KEY)
        if self._settings.status() != QSettings.Status.NoError:
            return
        try:
            version = format_version(_ini_integer(stored))
        except ValueError:
            log.warning("window.state_reset", path=str(path), reason="unusable formatVersion")
            self._settings.clear()
            return
        if version > WINDOW_STATE_FORMAT_VERSION:
            self._newer = True
            log.warning("window.state_newer_format", path=str(path), format_version=version)

    def value(self, key: str) -> object | None:
        """The saved value, or ``None`` when there is none this build may use."""
        if self._newer:
            return None
        value = self._settings.value(key)
        if self._settings.status() != QSettings.Status.NoError:
            log.warning("window.state_load_failed", key=key)
            return None
        return value

    def save(self, key: str, value: object) -> None:
        if self._newer:
            return
        self._settings.setValue(FORMAT_VERSION_KEY, WINDOW_STATE_FORMAT_VERSION)
        self._settings.setValue(key, value)
        self._settings.sync()
        if self._settings.status() != QSettings.Status.NoError:
            log.warning("window.state_save_failed", key=key)


def _ini_integer(value: object) -> object:
    # An INI file holds every value as text.
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("the format version is not an integer")
        return int(value)
    return value


def window_state() -> WindowState:
    return WindowState(window_settings_path())


def restored_window_state(state: Qt.WindowState, platform: str) -> Qt.WindowState:
    """The display state a restored window first appears in.

    Mac always reopens normal; Windows keeps a maximized or fullscreen window so,
    and never reopens minimized.
    """
    if platform == "darwin":
        return Qt.WindowState.WindowNoState
    return state & ~Qt.WindowState.WindowMinimized


def restore_window_geometry(widget: QWidget, state: WindowState, key: str) -> None:
    """Apply the saved geometry before ``widget`` is first shown; missing or
    unusable state leaves the designed initial size in place."""
    geometry = state.value(key)
    if isinstance(geometry, QByteArray) and not geometry.isEmpty():
        if not widget.restoreGeometry(geometry):
            log.warning("window.geometry_restore_failed", key=key)
            return
        restored = restored_window_state(widget.windowState(), sys.platform)
        if restored != widget.windowState():
            widget.setWindowState(restored)


def save_window_geometry(widget: QWidget, state: WindowState, key: str) -> None:
    state.save(key, widget.saveGeometry())
