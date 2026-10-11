"""Retain background reads through physical return, including obsolete requests."""

from __future__ import annotations

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot


class ReadThreads(QObject):
    idle = Signal()
    _returned = Signal(object)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._threads: set[threading.Thread] = set()
        self._returning: set[threading.Thread] = set()
        self.closing = False
        self._retirement = QTimer(self)
        self._retirement.setSingleShot(True)
        self._retirement.timeout.connect(self._retire)
        self._returned.connect(self._read_returned)

    @property
    def settled(self) -> bool:
        return not any(thread.is_alive() for thread in self._threads)

    def start(self, name: str, read: Callable[[], None]) -> None:
        if self.closing:
            return

        def run() -> None:
            try:
                read()
            finally:
                try:
                    self._returned.emit(threading.current_thread())
                except RuntimeError:
                    pass  # An explicitly disposed presentation has no callback owner.

        thread = threading.Thread(target=run, name=name, daemon=True)
        self._threads.add(thread)
        try:
            thread.start()
        except BaseException:
            self._threads.remove(thread)
            raise

    @Slot(object)
    def _read_returned(self, thread: threading.Thread) -> None:
        self._returning.add(thread)
        self._retire()

    @Slot()
    def _retire(self) -> None:
        ended = {thread for thread in self._returning if not thread.is_alive()}
        self._threads.difference_update(ended)
        self._returning.difference_update(ended)
        if self._returning:
            # The queued signal can arrive before emit() itself returns. Only
            # poll that returning thread, never a read still blocked on storage.
            self._retirement.start(0)
            return
        if not self._threads:
            self.idle.emit()
