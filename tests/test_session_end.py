from __future__ import annotations

import sys

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
