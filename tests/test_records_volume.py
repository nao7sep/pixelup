"""The real Records window over thousands of records while new ones arrive live:
the newest page first, paging through all of them by the scroll bar Qt reports,
new records at the top only while the list is there, and a list that never moves
once scrolled down. ``records_demo.py`` opens the same window to watch by eye."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QApplication
from records_demo import seed

from pixelup import records_window
from pixelup.config import resolve_state_dir
from pixelup.records import PAGE_SIZE
from pixelup.records_window import RecordsReads, RecordsWindow, stored_signal
from pixelup.session_log import (
    RECORDS_FILE_NAME,
    configure_session_logging,
    flush_records,
    log,
    records_store,
    set_stored_listener,
)
from pixelup.window_placement import WindowState

TOTAL = 10_000
# A few new records a second.
ARRIVAL_MS = 250


class _ThreadNoted(RecordsReads):
    """The real reads, noting which thread ran each one."""

    def __init__(self, store) -> None:  # type: ignore[no-untyped-def]
        super().__init__(store)
        self.threads: set[int] = set()

    def run(self, connection, op, argument):  # type: ignore[no-untyped-def]
        self.threads.add(threading.get_ident())
        return super().run(connection, op, argument)


def _stored_ids(database: Path) -> list[int]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT id FROM logs ORDER BY time DESC, id DESC").fetchall()
    return [row[0] for row in rows]


def _shown_ids(window: RecordsWindow) -> list[int]:
    return [record.id for record in window.model.records]


def _first_visible_id(window: RecordsWindow) -> int:
    return window.model.records[window.list.indexAt(QPoint(1, 1)).row()].id


def _settled(window: RecordsWindow, reads: RecordsReads) -> bool:
    return (
        window._status == "ready"
        and not window._loading_more
        and not window._live_timer.isActive()
        and not reads.busy
    )


@pytest.fixture
def volume(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(records_window, "LIVE_INTERVAL_MS", 20)
    database = resolve_state_dir() / RECORDS_FILE_NAME
    seed(database, TOTAL)
    configure_session_logging()
    set_stored_listener(stored_signal().emit)
    store = records_store()
    assert store is not None
    reads = _ThreadNoted(store)
    window = RecordsWindow(reads, WindowState(tmp_path / "window.ini"))
    window.show()
    yield database, window, reads
    window.close()


def _arrive(process_until, count: int) -> list[str]:  # type: ignore[no-untyped-def]
    """``count`` new records through the app's logger, one every ARRIVAL_MS."""
    messages = [f"live.record_{next(_numbers)}" for _ in range(count)]
    pending = list(messages)
    timer = QTimer()

    def one() -> None:
        log.info(pending.pop(0))
        if not pending:
            timer.stop()

    timer.timeout.connect(one)
    timer.start(ARRIVAL_MS)
    process_until(lambda: not pending, what="the arrivals")
    flush_records()
    # Delivers the stored signals the writer queued for the window.
    QApplication.processEvents()
    return messages


_numbers = iter(range(1, 1_000_000))


def test_pages_and_live_records_hold_over_thousands_of_records(volume, process_until) -> None:
    database, window, reads = volume
    bar = window.list.verticalScrollBar()

    # (a) The first page: the 100 newest, newest first.
    process_until(lambda: _settled(window, reads), what="the first page")
    assert _shown_ids(window) == _stored_ids(database)[:PAGE_SIZE]
    assert bar.value() == 0

    # (c) At the top, new records appear at the top.
    arrived = _arrive(process_until, 3)
    process_until(
        lambda: _settled(window, reads) and window.model.records[0].message == arrived[-1],
        what="the arrivals at the top",
    )
    assert [record.message for record in window.model.records[:3]] == arrived[::-1]
    assert _shown_ids(window) == _stored_ids(database)[: len(_shown_ids(window))]

    # (d) Scrolled down, the list holds still while records arrive.
    window.list.setCurrentIndex(window.model.index(40))
    selected = window.model.records[40].id
    bar.setValue(bar.maximum() // 2)
    assert bar.maximum() - bar.value() > bar.pageStep()
    process_until(lambda: _settled(window, reads), what="the scroll")
    offset, first_visible, shown = bar.value(), _first_visible_id(window), _shown_ids(window)
    detail = window.detail.toPlainText()
    _arrive(process_until, 5)
    process_until(
        lambda: window._newest_wanted and _settled(window, reads), what="the waiting arrivals"
    )
    assert bar.value() == offset
    assert _first_visible_id(window) == first_visible
    assert _shown_ids(window) == shown
    assert window.model.records[window.list.currentIndex().row()].id == selected
    assert window.detail.toPlainText() == detail

    # Back at the top, the arrivals that waited appear.
    bar.setValue(0)
    process_until(
        lambda: _shown_ids(window)[:9] == _stored_ids(database)[:9], what="the newest page"
    )
    assert _shown_ids(window) == _stored_ids(database)[: len(_shown_ids(window))]

    # (b) Within one screen of the end, the next page loads, through every record,
    # while records keep arriving; those wait, since the list is scrolled down.
    expected = _stored_ids(database)
    timer = QTimer()
    timer.timeout.connect(lambda: log.info(f"live.record_{next(_numbers)}"))
    timer.start(ARRIVAL_MS)
    try:
        while window._more:
            shown_count = window.model.rowCount()
            bar.setValue(bar.maximum())
            process_until(
                lambda count=shown_count: window.model.rowCount() > count
                and _settled(window, reads),
                what="the next page",
            )
    finally:
        timer.stop()
    flush_records()
    assert _shown_ids(window) == expected
    assert len(set(expected)) == len(expected) >= TOTAL

    # The records that arrived meanwhile appear once the list is back at the top.
    QApplication.processEvents()
    bar.setValue(0)
    process_until(
        lambda: _shown_ids(window) == _stored_ids(database), what="the waiting arrivals"
    )

    # (e) Every read ran on the records owner's thread, never the window's.
    assert reads.threads and threading.get_ident() not in reads.threads
