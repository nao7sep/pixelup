"""Where a durable window reopens: its native geometry blob in window.ini
(window-conventions, Placement implementation, Qt Widgets and PySide).

The main window and the Records window each keep one blob under their own key,
restored before the first frame and saved on every accepted close. Qt's own
restoreGeometry owns display-scale changes and moves a window that would land
off-screen back onto one.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import QWidget

from pixelup.config import window_settings_path
from pixelup.session_log import log


def window_settings() -> QSettings:
    """The disposable window state: placement and pane widths."""
    return QSettings(str(window_settings_path()), QSettings.Format.IniFormat)


def restored_window_state(state: Qt.WindowState, platform: str) -> Qt.WindowState:
    """The display state a restored window first appears in.

    Mac always reopens normal; Windows keeps a maximized or fullscreen window so,
    and never reopens minimized.
    """
    if platform == "darwin":
        return Qt.WindowState.WindowNoState
    return state & ~Qt.WindowState.WindowMinimized


def restore_window_geometry(widget: QWidget, settings: QSettings, key: str) -> None:
    """Apply the saved geometry before ``widget`` is first shown; missing or
    unusable state leaves the designed initial size in place."""
    geometry = settings.value(key)
    if settings.status() != QSettings.Status.NoError:
        log.warning("window.geometry_load_failed", key=key)
        return
    if isinstance(geometry, QByteArray) and not geometry.isEmpty():
        if not widget.restoreGeometry(geometry):
            log.warning("window.geometry_restore_failed", key=key)
            return
        state = restored_window_state(widget.windowState(), sys.platform)
        if state != widget.windowState():
            widget.setWindowState(state)


def save_window_geometry(widget: QWidget, settings: QSettings, key: str) -> None:
    settings.setValue(key, widget.saveGeometry())
    settings.sync()
    if settings.status() != QSettings.Status.NoError:
        log.warning("window.geometry_save_failed", key=key)
