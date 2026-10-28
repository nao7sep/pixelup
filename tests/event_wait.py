"""Run the Qt event loop until a condition holds, sleeping until something happens.

The loop sleeps in Qt's dispatcher until an event arrives, with no bound: a
stalled wait shows in the faulthandler dump at the caller's line, beside its
``what``. Workers in these tests report through ``threading.Event`` and pool
futures as well as through Qt signals, so the session wakes the dispatcher
whenever an event is set or a future settles; without that, a condition only
another thread changes would never be checked again.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from PySide6.QtCore import QAbstractEventDispatcher, QCoreApplication, QEventLoop

_dispatcher: QAbstractEventDispatcher | None = None


def wake() -> None:
    """Make a waiting ``process_until`` check its condition again; any thread may call it."""
    if _dispatcher is not None:
        _dispatcher.wakeUp()


@contextmanager
def waking_on_thread_events(app: QCoreApplication) -> Iterator[None]:
    global _dispatcher
    _dispatcher = QAbstractEventDispatcher.instance(app.thread())
    real_set = threading.Event.set

    def set_and_wake(self: threading.Event) -> None:
        real_set(self)
        wake()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(threading.Event, "set", set_and_wake)
        try:
            yield
        finally:
            _dispatcher = None


def process_until(done: Callable[[], bool], *, what: str) -> None:
    while not done():
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.WaitForMoreEvents)
