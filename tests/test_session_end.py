from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pixelup import session_end
from pixelup.session_end import is_session_end_event, os_session_ending

QUIT = int.from_bytes(b"quit", "big")
OPEN = int.from_bytes(b"odoc", "big")


@pytest.mark.parametrize(
    ("event_id", "has_reason", "ending"),
    [
        # A logout, restart or shutdown: the OS's quit event, with its reason.
        (QUIT, True, True),
        # The Dock's Quit: the same event, with no reason.
        (QUIT, False, False),
        # The menu's Quit and Cmd+Q: no Apple Event at all.
        (None, False, False),
        (OPEN, True, False),
    ],
)
def test_only_the_quit_event_with_a_reason_ends_the_session(
    event_id: int | None, has_reason: bool, ending: bool
) -> None:
    assert is_session_end_event(event_id, has_reason) is ending


def test_elsewhere_a_commit_request_is_always_the_session_ending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_end.sys, "platform", "win32")
    monkeypatch.setattr(session_end, "_current_apple_event", lambda: pytest.fail("probed"))
    assert os_session_ending() is True


def test_on_macos_the_current_apple_event_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_end.sys, "platform", "darwin")
    monkeypatch.setattr(session_end, "_current_apple_event", lambda: (QUIT, True))
    assert os_session_ending() is True
    monkeypatch.setattr(session_end, "_current_apple_event", lambda: (QUIT, False))
    assert os_session_ending() is False


def test_a_probe_that_fails_counts_as_the_users_own_quit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> tuple[int | None, bool]:
        raise OSError("no runtime")

    monkeypatch.setattr(session_end.sys, "platform", "darwin")
    monkeypatch.setattr(session_end, "_current_apple_event", broken)
    assert os_session_ending() is False
    assert [r for r in caplog.records if r.getMessage() == "quit.session_probe_failed"]


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the macOS Apple Event manager")
def test_outside_an_apple_event_the_probe_reads_none() -> None:
    assert session_end._current_apple_event() == (None, False)


@pytest.mark.parametrize(
    ("event_id", "reason", "expected"),
    [(QUIT, 444, True), (QUIT, 0, False), (OPEN, 444, False)],
)
def test_apple_event_bridge_reads_the_quit_reason_from_the_current_event(
    monkeypatch, event_id, reason, expected
):
    # Exercise the native adapter's selectors, receiver and argument types, not
    # only its pure classification. No real logout or user Apple Event is sent.
    calls = []

    class NativeCall:
        def __call__(self, value):
            return value

    objc = SimpleNamespace(objc_getClass=NativeCall(), sel_registerName=NativeCall())

    def prototype(restype, *argtypes):
        def bind(symbol):
            assert symbol == ("objc_msgSend", objc)

            def send(receiver, selector, *args):
                calls.append((receiver, selector, args))
                if selector == b"sharedAppleEventManager":
                    assert receiver == b"NSAppleEventManager"
                    return 111
                if selector == b"currentAppleEvent":
                    assert receiver == 111
                    return 222
                assert receiver == 222
                if selector == b"eventID":
                    assert restype is session_end.ctypes.c_uint32
                    return event_id
                assert selector == b"attributeDescriptorForKeyword:"
                assert args == (int.from_bytes(b"why?", "big"),)
                assert argtypes[-1] is session_end.ctypes.c_uint32
                return reason

            return send
        return bind

    monkeypatch.setattr(session_end.sys, "platform", "darwin")
    monkeypatch.setattr(session_end.ctypes, "CDLL", lambda _: objc)
    monkeypatch.setattr(session_end.ctypes, "CFUNCTYPE", prototype)
    assert os_session_ending() is expected
    assert [selector for _, selector, _ in calls] == [
        b"sharedAppleEventManager", b"currentAppleEvent", b"eventID",
        b"attributeDescriptorForKeyword:",
    ]
