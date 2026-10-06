"""The Records window: the list and its paging, the filters, the selected record,
live updates, the list width, and the window's place beside the main window —
opened once, placed like the main window, and never in the way of quitting."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from pixelup import gui, records_window
from pixelup.app_config import AppConfig, ConfigLoadResult
from pixelup.gui import MainWindow, is_reopen
from pixelup.i18n import localizer
from pixelup.i18n.languages import TAGS
from pixelup.i18n.message import Message
from pixelup.model_registry import ALL_MODELS
from pixelup.records import PAGE_SIZE, RecordDetail, RecordsQuery, RecordsReader
from pixelup.records_window import (
    GEOMETRY_KEY,
    LIST_WIDTH_DEFAULT,
    LIST_WIDTH_KEY,
    LIST_WIDTH_MAX,
    LIST_WIDTH_MIN,
    RecordsReads,
    RecordsWindow,
    detail_sections,
    launch_label,
    saved_list_width,
    stored_signal,
)
from pixelup.runner import JobRunner
from pixelup.session_log import (
    _open_records,
    configure_session_logging,
    log,
    set_stored_listener,
)
from pixelup.window_placement import FORMAT_VERSION_KEY, WindowState, window_state

NEW = "2026-10-01T09:00:00.000Z"
OLD = "2026-09-30T08:00:00.000Z"
WAIT_S = 5


def _time(second: int) -> str:
    return f"2026-10-01T09:{second // 60:02d}:{second % 60:02d}.000Z"


def _seed(database: Path, rows: list[dict]) -> None:
    connection = _open_records(database)
    try:
        for row in rows:
            connection.execute(
                "INSERT INTO logs (session, time, level, message, job_id, operation_id, fields)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("session", NEW),
                    row["time"],
                    row.get("level", "info"),
                    row.get("message", "image.added"),
                    row.get("job_id"),
                    row.get("operation_id"),
                    json.dumps(row.get("fields", {})),
                ),
            )
    finally:
        connection.close()


def _seed_count(database: Path, count: int, *, start: int = 0) -> None:
    _seed(database, [{"time": _time(i), "message": f"m{i}"} for i in range(start, start + count)])


class ScriptedReader(RecordsReader):
    """The real reader, counting the pages it is asked for, failing the ones
    named, and holding one at a gate until the test lets it go."""

    def __init__(self, database: Path) -> None:
        super().__init__(database)
        self.pages: list[tuple[RecordsQuery, object]] = []
        self.fail: set[int] = set()
        self.gate: threading.Event | None = None
        self.waiting = threading.Event()
        self._lock = threading.Lock()

    def page(self, query, after):  # type: ignore[no-untyped-def]
        with self._lock:
            number = len(self.pages)
            self.pages.append((query, after))
        if self.gate is not None:
            self.waiting.set()
            self.gate.wait(WAIT_S)
        if number in self.fail:
            raise sqlite3.OperationalError("disk I/O error in /secret/records.sqlite3")
        return super().page(query, after)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "records.sqlite3"


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_window, "LIVE_INTERVAL_MS", 20)
    monkeypatch.setattr(records_window, "SEARCH_DELAY_MS", 20)


@pytest.fixture
def open_records(qapp: QApplication, database: Path, settings: QSettings, fast: None):
    created: list[tuple[RecordsWindow, RecordsReads]] = []

    def make(reader: RecordsReader | None = None, *, show: bool = True) -> RecordsWindow:
        reads = RecordsReads(reader or RecordsReader(database))
        window = RecordsWindow(reads, WindowState(Path(settings.fileName())))
        created.append((window, reads))
        if show:
            window.show()
        return window

    yield make
    for window, reads in created:
        reads.stop()
        try:
            window.close()
        except RuntimeError:
            pass  # already deleted on its own close


def _until(process_until, done, what: str) -> None:  # type: ignore[no-untyped-def]
    process_until(done, timeout_s=WAIT_S, what=what)


def _loaded(window: RecordsWindow) -> bool:
    return window._status == "ready" and not window._pending


def _settle(qapp: QApplication, rounds: int = 20) -> None:
    for _ in range(rounds):
        qapp.processEvents()
        QTest.qWait(5)


def _messages(window: RecordsWindow) -> list[str]:
    return [record.message for record in window.model.records]


# The list -------------------------------------------------------------------------


def test_lists_the_records_newest_first_with_nothing_selected_yet(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, 3)
    window = open_records()

    # The first page is read on the reads thread; until it answers, the list
    # says it is loading.
    assert window.list.accessibleDescription() == "Loading records…"
    _until(process_until, lambda: _loaded(window), "the first page")

    assert _messages(window) == ["m2", "m1", "m0"]
    assert window.list.selectionModel().selectedIndexes() == []
    assert "Select a record to see everything it holds." in window.detail.toPlainText()


def test_says_when_no_record_matches_and_when_the_records_cannot_be_read(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, 1)
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")
    window.search.setText("nothing matches this")
    _until(process_until, lambda: _loaded(window) and window.model.rowCount() == 0, "search")
    assert window.list.accessibleDescription() == "No records match these filters."

    reader = ScriptedReader(database)
    reader.fail = {0}
    failing = open_records(reader)
    _until(process_until, lambda: failing._status == "failed", "the failed page")

    assert failing.list.accessibleDescription() == "The records could not be read."
    shown = failing.list.accessibleDescription() + failing.detail.toPlainText()
    assert "disk I/O" not in shown and "secret" not in shown


def test_shows_everything_a_selected_record_holds(
    open_records, database: Path, process_until
) -> None:
    fields = {"input": "/photos/a.png", "error": {"type": "ValueError", "message": "boom"}}
    _seed(database, [{"time": _time(1), "level": "error", "job_id": 7, "fields": fields}])
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")

    window.list.setCurrentIndex(window.model.index(0))
    _until(process_until, lambda: window._detail is not None, "the record")

    text = window.detail.toPlainText()
    for expected in ("image.added", "Error", "Time", "Job", "7", "Launch", "Details"):
        assert expected in text
    assert '"input": "/photos/a.png"' in text
    assert '"message": "boom"' in text
    assert window.list.selectionModel().isSelected(window.model.index(0))


def test_the_selection_follows_the_arrow_keys(open_records, database: Path, process_until) -> None:
    _seed_count(database, 3)
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")
    window.list.setFocus()
    window.list.setCurrentIndex(window.model.index(0))

    QTest.keyClick(window.list, Qt.Key.Key_Down)
    _until(process_until, lambda: window._detail is not None, "the record")

    assert window.list.selectionModel().selectedRows()[0].row() == 1
    assert window._detail.message == "m1"


def test_offers_needs_attention_first_among_the_levels_with_every_filter_off(
    open_records, database: Path, process_until
) -> None:
    _seed(database, [{"time": _time(1), "session": OLD}, {"time": _time(2)}])
    window = open_records()
    _until(process_until, lambda: window.launch_filter.count() == 3, "the launches")

    levels = [window.level_filter.itemText(i) for i in range(window.level_filter.count())]
    assert levels == ["All levels", "Needs attention", "Error", "Warning", "Info", "Debug"]
    assert window.level_filter.currentIndex() == 0
    assert window.launch_filter.currentText() == "All launches"
    assert window.search.text() == ""
    assert window._query == RecordsQuery()


def test_has_no_refresh_button(open_records) -> None:
    assert open_records().findChildren(QPushButton) == []


def test_reads_again_with_each_filter_and_searches_once_typing_pauses(
    open_records, database: Path, process_until, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(records_window, "SEARCH_DELAY_MS", 300)
    _seed(
        database,
        [
            {"time": _time(1), "session": OLD, "message": "app.started"},
            {"time": _time(2), "level": "warn", "message": "open.ignored_directory"},
            {"time": _time(3), "level": "error", "message": "job.failed"},
            {"time": _time(4), "message": "image.added"},
        ],
    )
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window) and window.launch_filter.count() == 3, "open")

    window.level_filter.setCurrentIndex(window.level_filter.findData("attention"))
    _until(process_until, lambda: _loaded(window), "the level filter")
    assert _messages(window) == ["job.failed", "open.ignored_directory"]

    window.level_filter.setCurrentIndex(0)
    window.launch_filter.setCurrentIndex(window.launch_filter.findData(OLD))
    _until(
        process_until, lambda: _loaded(window) and _messages(window) == ["app.started"], "launch"
    )

    window.launch_filter.setCurrentIndex(0)
    _until(process_until, lambda: _loaded(window) and len(_messages(window)) == 4, "all")
    reads = len(reader.pages)
    QTest.keyClicks(window.search, "job")
    _settle(QApplication.instance(), 5)
    # Nothing is read while the reader is still typing.
    assert len(reader.pages) == reads
    _until(process_until, lambda: _loaded(window) and _messages(window) == ["job.failed"], "search")
    assert len(reader.pages) == reads + 1


def test_reads_the_next_page_from_the_last_row_once_scrolled_near_the_end(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, PAGE_SIZE * 2 + 10)
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")
    assert window.model.rowCount() == PAGE_SIZE

    window.list.scrollToBottom()
    _until(process_until, lambda: window.model.rowCount() == PAGE_SIZE * 2, "the next page")

    assert reader.pages[1][1] == records_window.cursor_after(window.model.records[:PAGE_SIZE])


def test_reads_the_next_page_when_down_is_pressed_on_the_last_row(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, PAGE_SIZE + 5)
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")
    window.list.setFocus()
    window.list.setCurrentIndex(window.model.index(PAGE_SIZE - 2))

    QTest.keyClick(window.list, Qt.Key.Key_Down)
    _until(process_until, lambda: window.model.rowCount() == PAGE_SIZE + 5, "the next page")


def test_makes_one_request_for_two_scroll_events_together(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, PAGE_SIZE * 3)
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")
    reader.gate = threading.Event()

    window.list.scrollToBottom()
    window.list.verticalScrollBar().setValue(window.list.verticalScrollBar().maximum() - 1)
    window._list_scrolled(0)
    assert reader.waiting.wait(WAIT_S)
    reader.gate.set()
    _until(process_until, lambda: window.model.rowCount() == PAGE_SIZE * 2, "the next page")

    assert len(reader.pages) == 2


def test_reads_the_next_page_by_itself_while_a_page_does_not_fill_the_list(
    open_records, database: Path, process_until, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pixelup.records.PAGE_SIZE", 2)
    _seed_count(database, 5)
    window = open_records()

    _until(process_until, lambda: window.model.rowCount() == 5 and not window._more, "every page")


def test_keeps_a_failed_pages_note_at_the_end_and_reads_it_again_there(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, PAGE_SIZE + 5)
    reader = ScriptedReader(database)
    reader.fail = {1}
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")

    window.list.scrollToBottom()
    _until(process_until, lambda: window._more_failed, "the failed page")
    assert window.end_note.isVisibleTo(window)
    assert window.end_note.text() == "The records could not be read."
    assert window.model.rowCount() == PAGE_SIZE
    # No button: reaching the end again reads it again.
    window.list.scrollToTop()
    window.list.scrollToBottom()
    _until(process_until, lambda: window.model.rowCount() == PAGE_SIZE + 5, "the page again")
    assert window.end_note.isHidden()


# Live updates ---------------------------------------------------------------------


def test_reads_the_newest_page_once_for_a_burst_of_new_records_keeping_the_rows(
    open_records, database: Path, process_until, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(records_window, "LIVE_INTERVAL_MS", 200)
    _seed_count(database, 3)
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")

    _seed_count(database, 2, start=10)
    for _ in range(5):
        stored_signal().emit()
    _settle(QApplication.instance(), 3)
    # The rows stay while the new page loads.
    assert _messages(window) == ["m2", "m1", "m0"]
    _until(process_until, lambda: _messages(window)[:2] == ["m11", "m10"], "the newest page")

    assert len(reader.pages) == 2
    assert _messages(window) == ["m11", "m10", "m2", "m1", "m0"]


def test_leaves_the_list_alone_while_scrolled_down_and_updates_it_back_at_the_top(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, 60)
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")
    window.list.verticalScrollBar().setValue(200)

    _seed_count(database, 1, start=100)
    stored_signal().emit()
    _until(process_until, lambda: window._newest_pending, "the signal")
    assert len(reader.pages) == 1
    assert _messages(window)[0] == "m59"

    window.list.scrollToTop()
    _until(process_until, lambda: _messages(window)[0] == "m100", "the newest page")


def test_keeps_the_selected_record_selected_through_an_update(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, 3)
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")
    window.list.setCurrentIndex(window.model.index(1))
    selected = window.model.records[1].id

    _seed_count(database, 2, start=10)
    stored_signal().emit()
    _until(process_until, lambda: window.model.rowCount() == 5, "the newest page")

    current = window.list.currentIndex()
    assert window.model.records[current.row()].id == selected
    assert window.list.selectionModel().isSelected(current)


def test_ignores_new_record_signals_after_a_failed_read_until_one_succeeds(
    open_records, database: Path, process_until
) -> None:
    # The failed read is itself logged as a record, whose signal would start the
    # next read.
    _seed_count(database, 3)
    reader = ScriptedReader(database)
    reader.fail = {1}
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")

    stored_signal().emit()
    _until(process_until, lambda: window._live_suspended, "the failed read")
    stored_signal().emit()
    stored_signal().emit()
    _settle(QApplication.instance())
    assert len(reader.pages) == 2

    # A read that succeeds starts listening again.
    window.level_filter.setCurrentIndex(window.level_filter.findData("info"))
    _until(process_until, lambda: _loaded(window) and not window._live_suspended, "a good read")
    stored_signal().emit()
    _until(process_until, lambda: len(reader.pages) == 4, "a live read")


def test_stops_listening_for_new_records_once_closed(
    open_records, database: Path, process_until
) -> None:
    _seed_count(database, 3)
    reader = ScriptedReader(database)
    window = open_records(reader)
    _until(process_until, lambda: _loaded(window), "the first page")

    window.close()
    stored_signal().emit()
    _settle(QApplication.instance())

    assert len(reader.pages) == 1


def test_an_open_window_settles_instead_of_reading_its_own_reads_forever(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast: None,
    process_until,
) -> None:
    database = configure_session_logging()
    set_stored_listener(stored_signal().emit)
    reader = ScriptedReader(database)
    reads = RecordsReads(reader)
    window = RecordsWindow(reads, WindowState(tmp_path / "w.ini"))
    try:
        window.show()
        _until(process_until, lambda: _loaded(window), "the first page")
        log.info("image.added", input="a.png")
        _until(process_until, lambda: _messages(window)[0] == "image.added", "the new record")
        _settle(qapp)
        settled = len(reader.pages)
        _settle(qapp, 40)

        assert len(reader.pages) == settled
    finally:
        reads.stop()
        window.close()


# The list width -------------------------------------------------------------------


def test_saves_the_list_width_once_when_a_drag_ends(
    open_records, settings: QSettings, qapp: QApplication
) -> None:
    window = open_records()
    window.splitter.moveSplitter(LIST_WIDTH_MIN + 20, 1)
    qapp.processEvents()
    assert not settings.contains(LIST_WIDTH_KEY)

    QTest.mouseRelease(window.splitter.handle(1), Qt.MouseButton.LeftButton)

    assert int(settings.value(LIST_WIDTH_KEY)) == window.list_pane.width() == LIST_WIDTH_MIN + 20


def test_narrowing_the_window_narrows_the_list_and_saves_nothing(
    open_records, settings: QSettings, qapp: QApplication
) -> None:
    window = open_records()
    window.resize(window.minimumWidth(), window.height())
    qapp.processEvents()

    assert window.list_pane.width() >= LIST_WIDTH_MIN
    assert not settings.contains(LIST_WIDTH_KEY)


def test_restores_the_saved_list_width_before_the_first_frame(
    open_records, settings: QSettings
) -> None:
    settings.setValue(FORMAT_VERSION_KEY, 1)
    settings.setValue(LIST_WIDTH_KEY, LIST_WIDTH_MIN + 10)
    settings.sync()

    window = open_records(show=False)

    assert not window.isVisible()
    assert window.list_pane.width() == LIST_WIDTH_MIN + 10


def test_a_load_failure_names_records_from_a_newer_pixelup() -> None:
    assert records_window.load_failure_note({"newer_format": True}) == Message(
        "records.newerFormat"
    )
    assert records_window.load_failure_note({"newer_format": False}) == Message(
        "records.loadFailed"
    )
    assert records_window.load_failure_note(None) == Message("records.loadFailed")


def test_a_saved_width_is_kept_within_the_list_panes_bounds() -> None:
    assert saved_list_width(None) == LIST_WIDTH_DEFAULT
    assert saved_list_width("wide") == LIST_WIDTH_DEFAULT
    assert saved_list_width("9999") == LIST_WIDTH_MAX
    assert saved_list_width(10) == LIST_WIDTH_MIN
    assert saved_list_width("500") == 500


def test_the_window_minimum_holds_both_panes(open_records) -> None:
    window = open_records()

    assert window.minimumWidth() >= LIST_WIDTH_MIN + records_window.DETAIL_MIN_WIDTH
    assert window.list_pane.width() >= LIST_WIDTH_MIN
    assert window.detail.width() >= records_window.DETAIL_MIN_WIDTH


# Language -------------------------------------------------------------------------


def test_a_language_change_rewrites_the_window(
    open_records, database: Path, process_until, qapp: QApplication
) -> None:
    _seed(database, [{"time": _time(1), "job_id": 2}])
    window = open_records()
    _until(process_until, lambda: _loaded(window), "the first page")
    window.list.setCurrentIndex(window.model.index(0))
    _until(process_until, lambda: window._detail is not None, "the record")

    with localizer.speaking("de"):
        qapp.processEvents()
        assert window.windowTitle() == "Protokoll"
        assert window.level_filter.itemText(0) == "Alle Stufen"
        assert window.launch_filter.itemText(0) == "Alle Starts"
        assert window.search.placeholderText() == "Protokoll durchsuchen"
        assert "Auftrag" in window.detail.toPlainText()
    qapp.processEvents()
    assert window.windowTitle() == "Records"


_LATIN = re.compile(r"[A-Za-z]{2,}")
_KEY = re.compile(r"\brecords\.[a-zA-Z]+\b")


@pytest.mark.parametrize("tag", TAGS)
def test_every_label_of_a_record_speaks_the_language(tag: str, qapp: QApplication) -> None:
    record = RecordDetail(
        1, NEW, NEW, "error", "job.failed", 3, 2, json.dumps({"a": 1, "error": {"b": 2}})
    )
    with localizer.speaking(tag):
        rows, blocks = detail_sections(record, NEW)
        labels = [label for label, _value in (*rows, *blocks)]
        labels.append(launch_label(NEW, NEW))
        for label in labels:
            assert label and not _KEY.search(label), f"{tag}: {label!r}"
        if tag in {"ja", "ko", "zh-Hans", "ru"}:
            assert [label for label in labels if _LATIN.search(label)] == []


# Beside the main window -----------------------------------------------------------


@pytest.fixture
def make_main(qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast: None):
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr("pixelup.gui.load_app_config_result", lambda: ConfigLoadResult(AppConfig()))
    log_file = configure_session_logging()
    created: list[MainWindow] = []

    def make() -> MainWindow:
        models_dir = tmp_path / "home" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for info in ALL_MODELS:
            (models_dir / info.filename).write_bytes(b"ready")
        runtime_dirs = SimpleNamespace(models_dir=models_dir, temp_dir=tmp_path / "home" / "temp")
        window = MainWindow(log_file=log_file, runtime_dirs=runtime_dirs)
        created.append(window)
        return window

    yield make
    for window in created:
        if window._records_reads is not None:
            window._records_reads.stop()
        if window._records_window is not None:
            window._records_window.close()


def test_the_records_button_opens_one_window_and_brings_it_forward_again(make_main) -> None:
    main = make_main()

    main.records_button.click()
    first = main._records_window
    main.records_button.click()

    assert isinstance(first, RecordsWindow)
    assert main._records_window is first
    assert first.isVisible()
    assert first.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose) is False

    first.close()
    assert main._records_window is None
    main.records_button.click()
    assert main._records_window is not first


def test_the_records_window_reopens_where_it_was_closed(make_main, process_until) -> None:
    main = make_main()
    main._open_records_window()
    first = main._records_window
    first.setGeometry(20, 60, first.minimumWidth() + 4, 520)
    expected = first.saveGeometry()
    first.close()

    main._open_records_window()

    assert main._records_window.saveGeometry() == expected
    stored = window_state().value(GEOMETRY_KEY)
    assert stored == expected


def test_closing_the_main_window_closes_the_records_window_and_ends_its_reads(
    make_main,
) -> None:
    main = make_main()
    main._open_records_window()
    records = main._records_window
    reads = main._records_reads
    main._session_shutdown = True

    assert main.close() is True

    assert main._records_window is None
    assert records is not None and reads is not None
    assert reads._thread.isFinished()


def test_a_second_quit_while_a_read_is_still_in_flight_waits_for_it(
    make_main, database: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    gate = threading.Event()
    reader_holder: list[ScriptedReader] = []

    def held_reads(path: Path, parent: object) -> RecordsReads:
        reader = ScriptedReader(path)
        reader.gate = gate
        reader_holder.append(reader)
        return RecordsReads(reader, parent)  # type: ignore[arg-type]

    monkeypatch.setattr(gui, "records_reads", held_reads)
    monkeypatch.setattr(records_window, "STOP_WAIT_MS", 20)
    main = make_main()
    main.show()
    main._open_records_window()
    assert reader_holder[0].waiting.wait(WAIT_S)
    monkeypatch.setattr(gui, "warn_jobs_stopping", lambda _parent: None)

    first = QCloseEvent()
    main.closeEvent(first)
    second = QCloseEvent()
    main.closeEvent(second)

    assert first.isAccepted() is False
    assert second.isAccepted() is False
    assert main.isVisible()
    gate.set()
    _until(process_until, lambda: not main.isVisible(), "the deferred quit")
    assert main._records_reads._thread.isFinished()


def test_no_records_window_opens_once_quitting_has_begun(make_main, monkeypatch) -> None:
    main = make_main()
    monkeypatch.setattr(main.runner, "cleanup_for_quit", lambda: False)
    monkeypatch.setattr(gui, "warn_jobs_stopping", lambda _parent: None)
    main.close()

    main._open_records_window()

    assert main._records_window is None


def test_reopening_the_app_brings_the_main_window_back(make_main, qapp: QApplication) -> None:
    main = make_main()
    main.show()
    main._open_records_window()
    main.showMinimized()
    active = Qt.ApplicationState.ApplicationActive
    main._application_state = active

    main._application_state_changed(active)

    assert not main.isMinimized()


def test_only_becoming_active_while_already_active_is_a_reopen() -> None:
    active = Qt.ApplicationState.ApplicationActive
    inactive = Qt.ApplicationState.ApplicationInactive
    assert is_reopen(active, active)
    assert not is_reopen(inactive, active)
    assert not is_reopen(active, inactive)
