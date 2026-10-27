"""Where a durable window reopens: its native geometry blob in window.ini
(window-conventions, Placement implementation, Qt Widgets and PySide).

The main window and the Records window each keep one blob under their own key,
restored before the first frame and saved on every accepted close. Qt's own
restoreGeometry owns display-scale changes and moves a window that would land
off-screen back onto one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import QWidget

from pixelup.config import window_settings_path
from pixelup.session_log import log


class WindowState:
    """window.ini: the disposable window state, placement and pane widths.

    It carries no format marker (store-recovery-conventions): every value is
    checked where it is used — Qt validates its own geometry blob and each pane
    width is bounded — and anything unusable, whatever wrote it, leaves the designed
    size in place. A failed read or save costs only a diagnostic.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._settings = QSettings(str(path), QSettings.Format.IniFormat)

    def value(self, key: str) -> object | None:
        """The saved value, or ``None`` when there is none this build may use."""
        value = self._settings.value(key)
        if self._settings.status() != QSettings.Status.NoError:
            log.warning("window.state_load_failed", key=key)
            return None
        return value

    def save(self, key: str, value: object) -> None:
        self._settings.setValue(key, value)
        self._settings.sync()
        if self._settings.status() != QSettings.Status.NoError:
            log.warning("window.state_save_failed", key=key)


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
