"""Whether the OS is ending the session: a logout, restart or shutdown.

Qt emits ``commitDataRequest`` before it asks the windows to close. On Windows
and Linux that happens only when the session ends. macOS sends it for every
``terminate:``, which is also how the menu's Quit, Cmd+Q and the Dock's Quit
arrive; of those quits, only the quit Apple Event the OS sends for a logout,
restart or shutdown carries a reason.
"""

from __future__ import annotations

import ctypes
import sys

from pixelup.session_log import log

_QUIT_EVENT = int.from_bytes(b"quit", "big")
_QUIT_REASON = int.from_bytes(b"why?", "big")
_OBJC = "/usr/lib/libobjc.A.dylib"


def is_session_end_event(event_id: int | None, has_quit_reason: bool) -> bool:
    """Whether the Apple Event being handled is the OS quitting apps to end the session."""
    return event_id == _QUIT_EVENT and has_quit_reason


def os_session_ending() -> bool:
    """Whether the quit Qt is delivering now is the OS ending the session.

    Ask it while ``commitDataRequest`` is being handled, which is when macOS's
    Apple Event is current. If the event cannot be read, the quit counts as the
    user's own, which keeps the prompts that protect the user's work.
    """
    if sys.platform != "darwin":
        return True
    try:
        return is_session_end_event(*_current_apple_event())
    except Exception:  # noqa: BLE001 - a failed probe must not break the quit.
        log.exception("quit.session_probe_failed")
        return False


def _current_apple_event() -> tuple[int | None, bool]:
    """The event ID of the Apple Event being handled, and whether it has a quit reason."""
    objc = ctypes.CDLL(_OBJC)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(restype, receiver, selector: bytes, *args, argtypes=()):
        # objc_msgSend must be called through the prototype of the method it reaches.
        call = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
            ("objc_msgSend", objc)
        )
        return call(receiver, objc.sel_registerName(selector), *args)

    manager_class = objc.objc_getClass(b"NSAppleEventManager")
    if not manager_class:
        return None, False
    manager = send(ctypes.c_void_p, manager_class, b"sharedAppleEventManager")
    event = send(ctypes.c_void_p, manager, b"currentAppleEvent")
    if not event:
        return None, False
    event_id = send(ctypes.c_uint32, event, b"eventID")
    reason = send(
        ctypes.c_void_p,
        event,
        b"attributeDescriptorForKeyword:",
        _QUIT_REASON,
        argtypes=(ctypes.c_uint32,),
    )
    return event_id, bool(reason)
