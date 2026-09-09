from __future__ import annotations

import base64
import ctypes
import sys
from functools import cache

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QWidget

from pixelup.app_state import WindowBounds, WindowMode, WindowPlacement
from pixelup.session_log import log


def restore_window_placement(window: QWidget, saved: WindowPlacement | None) -> WindowMode:
    """Qt owns display recovery and frame/client conversion, including legacy state."""
    if saved is None:
        return "maximized"
    opening = window.geometry()
    try:
        restored = False
        if saved.qt_geometry:
            try:
                geometry = base64.b64decode(saved.qt_geometry, validate=True)
                restored = window.restoreGeometry(geometry)
                if not restored:
                    log.warning("window.geometry_unreadable")
            except ValueError:
                log.warning("window.geometry_unreadable", exc_info=True)
        if not restored and saved.normal_bounds is not None:
            bounds = saved.normal_bounds
            window.setGeometry(QRect(bounds.x, bounds.y, bounds.width, bounds.height))
            # Convert the compatible old client rectangle through Qt's own pair;
            # restoreGeometry also recovers unavailable displays and oversized bounds.
            if not window.restoreGeometry(window.saveGeometry()):
                raise RuntimeError("Qt could not restore legacy window geometry")
    except Exception:  # noqa: BLE001 - disposable placement must not prevent startup.
        window.setGeometry(opening)
        log.warning("window.restore_failed", exc_info=True)
    finally:
        # Qt's blob includes fullscreen/maximized flags. Only our separately saved
        # normal/maximized mode may launch; clear the blob's flags while still hidden.
        window.setWindowState(Qt.WindowState.WindowNoState)
    return saved.mode


@cache
def _cocoa_messages():
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    pointer_message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    bool_message = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    return pointer_message, bool_message, objc.sel_registerName


def is_window_maximized(window: QWidget) -> bool:
    if sys.platform != "darwin" or not window.isMaximized():
        return window.isMaximized()
    # Native AXZoomWindow unzoom has left Qt's WindowMaximized flag stale on macOS.
    # Qt's WId is an NSView; query its NSWindow.isZoomed rather than guessing from size.
    pointer_message, bool_message, selector = _cocoa_messages()
    native_window = pointer_message(int(window.winId()), selector(b"window"))
    if not native_window:
        raise RuntimeError("Qt window has no Cocoa NSWindow")
    return bool_message(native_window, selector(b"isZoomed"))


def capture_window_placement(window: QWidget, previous_mode: WindowMode) -> WindowPlacement:
    transient = window.isMinimized() or window.isFullScreen()
    maximized = is_window_maximized(window) if not transient else False
    mode = previous_mode if transient else "maximized" if maximized else "normal"
    stale_maximized = not transient and window.isMaximized() and not maximized
    normal = window.geometry() if stale_maximized else window.normalGeometry()
    bounds = (
        WindowBounds(normal.x(), normal.y(), normal.width(), normal.height())
        if normal.isValid() else None
    )
    # A stale Qt maximize flag also makes its blob's normal rectangle stale. Keep
    # the actual normal rectangle in the compatible format for this native case.
    geometry = None if stale_maximized else bytes(window.saveGeometry().toBase64()).decode("ascii")
    return WindowPlacement(bounds, mode, geometry)
