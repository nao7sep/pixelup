"""The Records window: records.sqlite3 as a list of records, newest first, beside
the selected record whole.

It is a durable secondary window with its own placement (window-conventions,
Placement), and there is only ever one: the main window opens it and brings it
forward when it is opened again. Every read runs on the reads thread
(RecordsReads), never on the GUI thread, and the window takes no part in
quitting: it does not keep the app running once the main window has closed.
"""

from __future__ import annotations

import html
import traceback
from collections.abc import Callable, Sequence
from itertools import count
from pathlib import Path

from PySide6.QtCore import (
    QAbstractListModel,
    QDateTime,
    QLocale,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    SignalInstance,
    Slot,
)
from PySide6.QtGui import (
    QAccessible,
    QAccessibleEvent,
    QCloseEvent,
    QColor,
    QFont,
    QFontDatabase,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QSplitter,
    QSplitterHandle,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pixelup.formats import NewerFormatError
from pixelup.i18n import localizer
from pixelup.i18n.localized import localize, unlocalize
from pixelup.i18n.message import Message
from pixelup.records import (
    LEVEL_FILTERS,
    RecordCursor,
    RecordDetail,
    RecordLevelFilter,
    RecordsPage,
    RecordsQuery,
    RecordsReader,
    RecordSummary,
    cursor_after,
    merge_newest_page,
    split_fields,
)
from pixelup.session_log import current_session, log
from pixelup.theme import danger_colours, surfaces, warning_text
from pixelup.ui_common import REGULAR_SPACING, use_regular_spacing
from pixelup.widgets import NoWheelComboBox, repolish
from pixelup.window_placement import (
    WindowState,
    restore_window_geometry,
    save_window_geometry,
    window_state,
)

GEOMETRY_KEY = "recordsWindow/geometry"
LIST_WIDTH_KEY = "recordsWindow/listWidth"
# The list pane is the user-adjustable one (window-conventions, Content-based
# minimum size); the detail pane takes the rest and keeps its own minimum.
LIST_WIDTH_MIN = 320
LIST_WIDTH_DEFAULT = 380
LIST_WIDTH_MAX = 640
# One arrow press moves the list pane's edge this far, as in BigMouth.
KEY_RESIZE_STEP = 16
DETAIL_MIN_WIDTH = 420
INITIAL_SIZE = QSize(1240, 820)
SEARCH_DELAY_MS = 300
# New records are read at most this often while they keep arriving.
LIVE_INTERVAL_MS = 1000
# How long quitting waits for a read in flight, which its own bound already
# limits (records.READ_TIMEOUT_SECONDS) and a cancel interrupts at once.
STOP_WAIT_MS = 2000
_ROW_PADDING_X = 10
_ROW_PADDING_Y = 8
_ROW_LINE_GAP = 2

LEVEL_LABELS = {
    "error": "records.levelError",
    "warn": "records.levelWarn",
    "info": "records.levelInfo",
    "debug": "records.levelDebug",
}
LEVEL_FILTER_LABELS: dict[RecordLevelFilter, str] = {
    "attention": "records.levelAttention",
    "error": "records.levelError",
    "warn": "records.levelWarn",
    "info": "records.levelInfo",
    "debug": "records.levelDebug",
}


def clamp_list_width(width: int) -> int:
    return max(LIST_WIDTH_MIN, min(LIST_WIDTH_MAX, width))


def saved_list_width(value: object) -> int:
    """The list pane's saved width, or its default when none was saved or it is unreadable."""
    try:
        return clamp_list_width(int(str(value)))
    except ValueError:
        return LIST_WIDTH_DEFAULT


def format_record_time(stored: str, locale: QLocale, *, milliseconds: bool = False) -> str:
    """A stored UTC instant as local time in the reader's locale, to the second.

    A text the database holds that is not an instant is shown as it is.
    """
    moment = QDateTime.fromString(stored, Qt.DateFormat.ISODateWithMs)
    if not moment.isValid():
        return stored
    local = moment.toLocalTime()
    time_format = locale.timeFormat(QLocale.FormatType.ShortFormat)
    if "ss" not in time_format:
        time_format = time_format.replace("mm", "mm:ss")
    if milliseconds:
        time_format = time_format.replace("ss", "ss.zzz")
    date = locale.toString(local.date(), QLocale.FormatType.ShortFormat)
    return f"{date} {locale.toString(local.time(), time_format)}"


def launch_label(session: str, this_session: str | None) -> str:
    time = format_record_time(session, localizer.current().locale)
    return localizer.t("records.thisLaunch", time=time) if session == this_session else time


def level_label(level: str) -> str:
    key = LEVEL_LABELS.get(level)
    return level if key is None else localizer.t(key)


def detail_sections(
    record: RecordDetail, this_session: str | None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Every field the record holds, as (label, value) rows and (label, text) blocks."""
    translator = localizer.current()
    rows = [
        (
            localizer.t("records.time"),
            format_record_time(record.time, translator.locale, milliseconds=True),
        )
    ]
    if record.job_id is not None:
        rows.append((localizer.t("records.job"), translator.number(record.job_id)))
    if record.operation_id is not None:
        rows.append((localizer.t("records.modelInstall"), translator.number(record.operation_id)))
    rows.append((localizer.t("records.launch"), launch_label(record.session, this_session)))
    details, error = split_fields(record.fields)
    blocks = []
    if details is not None:
        blocks.append((localizer.t("records.details"), details))
    if error is not None:
        blocks.append((localizer.t("records.error"), error))
    return rows, blocks


def near_end(value: int, maximum: int, page_step: int) -> bool:
    """Within about one screen of the end of what is loaded."""
    return maximum - value <= page_step


class _StoredSignal(QObject):
    stored = Signal()


_stored: _StoredSignal | None = None


def stored_signal() -> SignalInstance:
    """Emitted after each record the database stored, on whichever thread logged it.

    One object for the process, made on the GUI thread, so a receiver there gets
    the signal queued whichever thread emits it.
    """
    global _stored
    if _stored is None:
        _stored = _StoredSignal()
    return _stored.stored


def load_failure_note(error: object) -> Message:
    """What the list says when the records cannot be read: records from a newer
    PixelUp are named as such, since reading them again will not help."""
    if isinstance(error, dict) and error.get("newer_format"):
        return Message("records.newerFormat")
    return Message("records.loadFailed")


class _ReadWorker(QObject):
    answered = Signal(int, object)
    failed = Signal(int, object)

    def __init__(self, reader: RecordsReader) -> None:
        super().__init__()
        self._reader = reader

    @Slot(int, str, object)
    def run(self, request_id: int, op: str, argument: object) -> None:
        if self._reader.cancelled:
            return
        try:
            if op == "page":
                query, after = argument  # type: ignore[misc]
                result: object = self._reader.page(query, after)
            elif op == "detail":
                result = self._reader.detail(argument)  # type: ignore[arg-type]
            else:
                result = self._reader.sessions()
        except Exception as exc:  # noqa: BLE001 - a failed read is answered, never raised into Qt.
            self.failed.emit(
                request_id,
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().rstrip(),
                    "newer_format": isinstance(exc, NewerFormatError),
                },
            )
            return
        self.answered.emit(request_id, result)

    @Slot()
    def close(self) -> None:
        self._reader.close()


class RecordsReads(QObject):
    """The one thread every records read runs on, one read at a time.

    Answers arrive on the GUI thread as ``answered`` or ``failed`` with the id
    ``request`` returned. ``stop`` cancels the read in flight and is part of
    quitting; ``stopped`` says the thread has ended.
    """

    answered = Signal(int, object)
    failed = Signal(int, object)
    stopped = Signal()
    _requested = Signal(int, str, object)

    def __init__(self, reader: RecordsReader, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._ids = count(1)
        self._thread = QThread(self)
        self._worker = _ReadWorker(reader)
        self._worker.moveToThread(self._thread)
        self._requested.connect(self._worker.run)
        self._worker.answered.connect(self.answered)
        self._worker.failed.connect(self.failed)
        # The connection is closed on the thread that opened it, as it ends.
        self._thread.finished.connect(self._worker.close, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self.stopped)
        self._thread.start()

    def request(self, op: str, argument: object = None) -> int:
        request_id = next(self._ids)
        self._requested.emit(request_id, op, argument)
        return request_id

    def stop(self) -> bool:
        """Cancel what is in flight and end the thread; whether it ended within the bound."""
        self._reader.cancel()
        self._thread.quit()
        return self._thread.wait(STOP_WAIT_MS)


def records_reads(database: Path, parent: QObject | None = None) -> RecordsReads:
    return RecordsReads(RecordsReader(database), parent)


class RecordsListModel(QAbstractListModel):
    """The records shown, newest first."""

    RecordRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.records: list[RecordSummary] = []

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self.records)

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> object:
        if not index.isValid() or index.row() >= len(self.records):
            return None
        record = self.records[index.row()]
        if role == self.RecordRole:
            return record
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return record.message
        return None

    def replace(self, records: Sequence[RecordSummary]) -> None:
        self.beginResetModel()
        self.records = list(records)
        self.endResetModel()

    def append(self, records: Sequence[RecordSummary]) -> None:
        if not records:
            return
        start = len(self.records)
        self.beginInsertRows(QModelIndex(), start, start + len(records) - 1)
        self.records.extend(records)
        self.endInsertRows()

    def merge(self, records: list[RecordSummary]) -> None:
        """Show ``records``; new rows ahead of the ones shown are inserted, so the
        view keeps its selection and scroll position."""
        added = len(records) - len(self.records)
        if added >= 0 and records[added:] == self.records:
            if added:
                self.beginInsertRows(QModelIndex(), 0, added - 1)
                self.records = records
                self.endInsertRows()
            return
        self.replace(records)

    def row_of(self, record_id: int) -> int | None:
        return next(
            (row for row, record in enumerate(self.records) if record.id == record_id), None
        )


class _RecordDelegate(QStyledItemDelegate):
    """A row: its time and level, its message, and the start of its fields."""

    def _lines(self, option: QStyleOptionViewItem) -> int:
        return option.fontMetrics.lineSpacing()

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:  # noqa: N802
        line = self._lines(option)
        return QSize(LIST_WIDTH_MIN, 3 * line + 2 * _ROW_LINE_GAP + 2 * _ROW_PADDING_Y)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        record = index.data(RecordsListModel.RecordRole)
        if not isinstance(record, RecordSummary):
            return
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        styled.text = ""
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        # The background, the selection and the keyboard's focus ring are the
        # style's, drawn inside the row as the app's tables draw theirs.
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, styled, painter, widget)

        palette = option.palette
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        ink = palette.color(
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text
        )
        quiet = QColor(ink)
        if not selected:
            quiet.setAlphaF(0.68)
        level_ink = quiet
        if not selected and record.level == "error":
            level_ink = QColor(danger_colours(palette)["text"])
        elif not selected and record.level == "warn":
            level_ink = QColor(warning_text(palette))

        line = self._lines(option)
        rect = option.rect.adjusted(
            _ROW_PADDING_X, _ROW_PADDING_Y, -_ROW_PADDING_X, -_ROW_PADDING_Y
        )
        metrics = option.fontMetrics
        painter.save()
        painter.setFont(option.font)
        time = format_record_time(record.time, localizer.current().locale)
        first = QRect(rect.left(), rect.top(), rect.width(), line)
        painter.setPen(quiet)
        painter.drawText(first, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, time)
        level_left = first.left() + metrics.horizontalAdvance(time) + REGULAR_SPACING
        painter.setPen(level_ink)
        painter.drawText(
            QRect(level_left, first.top(), max(0, first.right() - level_left), line),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            level_label(record.level),
        )

        bold = QFont(option.font)
        bold.setWeight(QFont.Weight.DemiBold)
        painter.setFont(bold)
        painter.setPen(ink)
        second = first.translated(0, line + _ROW_LINE_GAP)
        painter.drawText(
            second,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                record.message, Qt.TextElideMode.ElideRight, second.width()
            ),
        )

        fields = "" if record.fields in ("", "{}") else record.fields
        if fields:
            painter.setFont(option.font)
            painter.setPen(quiet)
            third = second.translated(0, line + _ROW_LINE_GAP)
            painter.drawText(
                third,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(
                    " ".join(fields.split()), Qt.TextElideMode.ElideRight, third.width()
                ),
            )
        painter.restore()


class _RecordList(QListView):
    """The list of records: one listbox (composite-control-conventions, Listbox),
    which says in its own body while it has no rows that it is loading, empty,
    or could not be read."""

    end_reached = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("recordsList")
        self._note: Message | None = None
        self._note_failed = False

    def set_note(self, note: Message | None, *, failed: bool = False) -> None:
        self._note = note
        self._note_failed = failed
        self.sync_note()

    def sync_note(self, *_args: object) -> None:
        shown = self._note is not None and (self.model() is None or self.model().rowCount() == 0)
        self.setAccessibleDescription(localizer.of(self._note) if shown and self._note else "")
        self.viewport().update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override name
        super().keyPressEvent(event)
        # Moving on past the last row loaded reads the next page, the way
        # scrolling to the end does.
        if (
            event.key() in (Qt.Key.Key_Down, Qt.Key.Key_PageDown, Qt.Key.Key_End)
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self.model() is not None
            and self.currentIndex().row() == self.model().rowCount() - 1
        ):
            self.end_reached.emit()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override name
        super().paintEvent(event)
        if self._note is None or (self.model() is not None and self.model().rowCount() > 0):
            return
        painter = QPainter(self.viewport())
        painter.setFont(self.font())
        if self._note_failed:
            colour = QColor(danger_colours(self.palette())["text"])
        else:
            colour = self.palette().color(QPalette.ColorRole.Text)
            colour.setAlphaF(0.68)
        painter.setPen(colour)
        painter.drawText(
            self.viewport().rect().adjusted(16, 16, -16, -16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            localizer.of(self._note),
        )


class _SplitterHandle(QSplitterHandle):
    """The list pane's divider: dragged with the mouse, or focused and moved with
    the keyboard — arrows by ``KEY_RESIZE_STEP``, Home and End to the bounds. A drag
    or a run of keys ends in one ``resize_finished``: on mouse release, or on key
    release or focus loss after a keyed move."""

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._keyed = False

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().mouseReleaseEvent(event)
        splitter = self.splitter()
        if event.button() == Qt.MouseButton.LeftButton and isinstance(splitter, _ListSplitter):
            splitter.resize_finished.emit()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        splitter = self.splitter()
        if not isinstance(splitter, _ListSplitter):
            super().keyPressEvent(event)
            return
        width = splitter.sizes()[0]
        key = event.key()
        if key == Qt.Key.Key_Left:
            target = width - KEY_RESIZE_STEP
        elif key == Qt.Key.Key_Right:
            target = width + KEY_RESIZE_STEP
        elif key == Qt.Key.Key_Home:
            target = LIST_WIDTH_MIN
        elif key == Qt.Key.Key_End:
            target = LIST_WIDTH_MAX
        else:
            super().keyPressEvent(event)
            return
        event.accept()
        splitter.set_list_width(target)
        self._keyed = True

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().keyReleaseEvent(event)
        if not event.isAutoRepeat():
            self._commit_keyed()

    def focusInEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().focusOutEvent(event)
        self._commit_keyed()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().paintEvent(event)
        if self.hasFocus():
            painter = QPainter(self)
            painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Highlight))
            painter.end()

    def _commit_keyed(self) -> None:
        splitter = self.splitter()
        if self._keyed and isinstance(splitter, _ListSplitter):
            self._keyed = False
            splitter.resize_finished.emit()


class _ListSplitter(QSplitter):
    """A splitter that says when a resize of its list pane ends, by drag or keys."""

    resize_finished = Signal()

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 - Qt override name
        return _SplitterHandle(self.orientation(), self)

    def set_list_width(self, width: int) -> None:
        """Give the list pane ``width``, within its bounds and the detail pane's minimum."""
        usable = sum(self.sizes())
        width = max(LIST_WIDTH_MIN, min(width, LIST_WIDTH_MAX, usable - DETAIL_MIN_WIDTH))
        self.setSizes([width, max(DETAIL_MIN_WIDTH, usable - width)])


type _Answer = Callable[[object], None]


class RecordsWindow(QWidget):
    """Filters and the list of records on the left; the selected record on the right."""

    closed = Signal()

    def __init__(self, reads: RecordsReads, state: WindowState | None = None) -> None:
        super().__init__(None, Qt.WindowType.Window)
        # Closing the main window quits PixelUp whether or not this one is open.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        localize(self, title="records.title")
        self._reads = reads
        self._state = state if state is not None else window_state()
        self._pending: dict[int, tuple[_Answer, _Answer]] = {}
        self._query = RecordsQuery()
        self._sessions: tuple[str, ...] = ()
        self._this_session = current_session()
        # A page applies only while the filters it was read for are still the
        # newest ones asked for.
        self._generation = 0
        self._status = "loading"
        self._more = False
        self._loading_more = False
        self._more_failed = False
        # The busy claim for the next page (PLAYBOOK, Own the work in flight).
        self._fetching_more = False
        # New records arrived while the list was scrolled away from the top.
        self._newest_pending = False
        # A failed read is itself logged as a record, whose signal would start the
        # next read; live reads stop after a failure and resume after a read succeeds.
        self._live_suspended = False
        self._selected_id: int | None = None
        self._detail: RecordDetail | None = None
        self._detail_note: Message | None = Message("records.noSelection")
        self._detail_failed = False
        self._closed = False

        self._build()
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DELAY_MS)
        self._search_timer.timeout.connect(self._search_settled)
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(LIVE_INTERVAL_MS)
        self._live_timer.timeout.connect(self._live_tick)

        reads.answered.connect(self._answered)
        reads.failed.connect(self._failed)
        # Queued, so a line logged on this thread never runs this window's code
        # inside the logger.
        stored_signal().connect(self._record_stored, Qt.ConnectionType.QueuedConnection)
        localizer.changed.connect(self._retranslate)

        self._place()
        self._read_sessions()
        self._read_first_page()

    # Building and placing ------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        use_regular_spacing(layout)
        self.splitter = _ListSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.resize_finished.connect(self._save_list_width)

        self.list_pane = QWidget()
        self.list_pane.setMinimumWidth(LIST_WIDTH_MIN)
        self.list_pane.setMaximumWidth(LIST_WIDTH_MAX)
        pane = QVBoxLayout(self.list_pane)
        use_regular_spacing(pane, margins=False)
        self.search = localize(
            QLineEdit(), placeholder="records.search", accessible_name="records.search"
        )
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._search_timer_restart)
        pane.addWidget(self.search)
        filters = QHBoxLayout()
        filters.setSpacing(REGULAR_SPACING)
        self.launch_filter = localize(NoWheelComboBox(), accessible_name="records.launch")
        self.level_filter = localize(NoWheelComboBox(), accessible_name="records.level")
        for combo in (self.launch_filter, self.level_filter):
            # A long launch label shortens rather than widening the pane.
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
            filters.addWidget(combo, 1)
        self.level_filter.addItem(localizer.t("records.allLevels"), None)
        for level in LEVEL_FILTERS:
            self.level_filter.addItem(localizer.t(LEVEL_FILTER_LABELS[level]), level)
        self._fill_launches()
        self.launch_filter.currentIndexChanged.connect(self._filters_changed)
        self.level_filter.currentIndexChanged.connect(self._filters_changed)
        pane.addLayout(filters)

        self.model = RecordsListModel(self)
        self.list = localize(_RecordList(), accessible_name="records.title")
        self.list.setModel(self.model)
        for change in (self.model.modelReset, self.model.rowsInserted):
            change.connect(self.list.sync_note)
        self.list.setItemDelegate(_RecordDelegate(self.list))
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setMinimumHeight(240)
        self.list.selectionModel().currentChanged.connect(self._current_changed)
        self.list.verticalScrollBar().valueChanged.connect(self._list_scrolled)
        self.list.end_reached.connect(self._load_more)
        pane.addWidget(self.list, 1)
        # A page after the first one that is loading, or could not be read.
        self.end_note = QLabel()
        self.end_note.setWordWrap(True)
        self.end_note.hide()
        pane.addWidget(self.end_note)

        self.detail = localize(QTextBrowser(), accessible_name="records.details")
        self.detail.setProperty("surface", "panel")
        self.detail.setOpenLinks(False)
        self.detail.setTabChangesFocus(True)
        self.detail.setMinimumWidth(DETAIL_MIN_WIDTH)
        self.detail.document().setDocumentMargin(12)

        self.splitter.addWidget(self.list_pane)
        self.splitter.addWidget(self.detail)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        localize(self.splitter.handle(1), accessible_name="records.resizeList")
        layout.addWidget(self.splitter)
        self.list.set_note(Message("records.loading"))
        self._render_detail()

    def _place(self) -> None:
        """Size, placement and the list width, all before the first frame."""
        self.winId()
        screen = self.screen() or QApplication.primaryScreen()
        size = INITIAL_SIZE
        if screen is not None:
            available = screen.availableGeometry().size()
            handle = self.windowHandle()
            if handle is not None:
                margins = handle.frameMargins()
                available -= QSize(
                    margins.left() + margins.right(), margins.top() + margins.bottom()
                )
            size = size.boundedTo(available)
        self.resize(size.expandedTo(self.minimumSizeHint()))
        restore_window_geometry(self, self._state, GEOMETRY_KEY)
        self.layout().activate()
        # The saved intent, narrowed only as far as this window needs; the intent
        # itself stays saved for when there is room again.
        usable = self.splitter.width() - self.splitter.handleWidth()
        intent = saved_list_width(self._state.value(LIST_WIDTH_KEY))
        width = max(LIST_WIDTH_MIN, min(intent, usable - DETAIL_MIN_WIDTH))
        self.splitter.setSizes([width, max(DETAIL_MIN_WIDTH, usable - width)])

    @Slot()
    def _save_list_width(self) -> None:
        # Drag or key intent: window-conventions. Only the user's move saves, never a
        # window resize.
        width = clamp_list_width(self.list_pane.width())
        self._state.save(LIST_WIDTH_KEY, width)

    def bring_forward(self) -> None:
        if self.isMinimized():
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override name
        event.accept()
        if self._closed:
            return
        self._closed = True
        save_window_geometry(self, self._state, GEOMETRY_KEY)
        self._search_timer.stop()
        self._live_timer.stop()
        self._pending.clear()
        self._reads.answered.disconnect(self._answered)
        self._reads.failed.disconnect(self._failed)
        stored_signal().disconnect(self._record_stored)
        localizer.changed.disconnect(self._retranslate)
        self.closed.emit()

    # Reads -----------------------------------------------------------------------

    def _read(self, op: str, argument: object, on_answer: _Answer, on_failure: _Answer) -> None:
        self._pending[self._reads.request(op, argument)] = (on_answer, on_failure)

    @Slot(int, object)
    def _answered(self, request_id: int, result: object) -> None:
        entry = self._pending.pop(request_id, None)
        if entry is not None:
            entry[0](result)

    @Slot(int, object)
    def _failed(self, request_id: int, error: object) -> None:
        entry = self._pending.pop(request_id, None)
        if entry is not None:
            entry[1](error)

    def _read_failed(self, read: str, error: object) -> None:
        # Suspended first: the warning below is itself a stored record.
        self._live_suspended = True
        log.warning("records.read_failed", read=read, failure=error)

    def _read_sessions(self) -> None:
        self._read(
            "sessions",
            None,
            self._sessions_read,
            lambda error: self._read_failed("sessions", error),
        )

    def _sessions_read(self, sessions: object) -> None:
        if isinstance(sessions, tuple) and sessions != self._sessions:
            self._sessions = sessions
            self._fill_launches()

    def _read_first_page(self) -> None:
        self._generation += 1
        generation = self._generation
        self._fetching_more = False
        self._newest_pending = False
        self._status = "loading"
        self._more = False
        self._loading_more = False
        self._more_failed = False
        self.model.replace([])
        self.list.set_note(Message("records.loading"))
        self._sync_end_note()
        self._read(
            "page",
            (self._query, None),
            lambda page: self._first_page_read(generation, page),
            lambda error: self._first_page_failed(generation, error),
        )

    def _first_page_read(self, generation: int, page: object) -> None:
        if generation != self._generation or not isinstance(page, RecordsPage):
            return
        self._live_suspended = False
        self._status = "ready"
        self.model.replace(page.records)
        self._more = page.more
        self.list.set_note(Message("records.empty"))
        self._keep_selection()
        self._check_end_soon()

    def _first_page_failed(self, generation: int, error: object) -> None:
        if generation != self._generation:
            return
        self._read_failed("page", error)
        self._status = "failed"
        self.list.set_note(load_failure_note(error), failed=True)
        QAccessible.updateAccessibility(QAccessibleEvent(self.list, QAccessible.Event.Alert))

    def _read_newest(self) -> None:
        """The newest page read again for new records. It joins the rows already
        shown rather than replacing them, so the list never falls back to the
        loading note and the pages already read stay."""
        generation = self._generation
        self._read(
            "page",
            (self._query, None),
            lambda page: self._newest_read(generation, page),
            lambda error: self._newest_failed(generation, error),
        )

    def _newest_read(self, generation: int, page: object) -> None:
        if generation != self._generation or not isinstance(page, RecordsPage):
            return
        self._live_suspended = False
        if self._status != "ready":
            self._first_page_read(generation, page)
            return
        records, more = merge_newest_page(self.model.records, self._more, page)
        self.model.merge(records)
        self._more = more
        self._keep_selection()
        self._check_end_soon()

    def _newest_failed(self, generation: int, error: object) -> None:
        if generation == self._generation:
            self._read_failed("page", error)

    @Slot()
    def _load_more(self) -> None:
        # Loading more: composite-control-conventions, Integration Points. A
        # failed page is read again when the end is reached again.
        if self._status != "ready" or not self._more or self._fetching_more:
            return
        self._fetching_more = True
        generation = self._generation
        self._loading_more = True
        self._more_failed = False
        self._sync_end_note()
        after: RecordCursor | None = cursor_after(self.model.records)
        self._read(
            "page",
            (self._query, after),
            lambda page: self._more_read(generation, page),
            lambda error: self._more_failed_read(generation, error),
        )

    def _more_read(self, generation: int, page: object) -> None:
        if generation != self._generation or not isinstance(page, RecordsPage):
            return
        self._fetching_more = False
        self._live_suspended = False
        self._loading_more = False
        self.model.append(page.records)
        self._more = page.more
        self._sync_end_note()
        self._check_end_soon()

    def _more_failed_read(self, generation: int, error: object) -> None:
        if generation != self._generation:
            return
        self._fetching_more = False
        self._read_failed("page", error)
        self._loading_more = False
        self._more_failed = True
        self._sync_end_note()
        QAccessible.updateAccessibility(QAccessibleEvent(self.end_note, QAccessible.Event.Alert))

    def _check_end_soon(self) -> None:
        # Once the view has laid the new rows out.
        QTimer.singleShot(0, self._check_end)

    @Slot()
    def _check_end(self) -> None:
        # A page that leaves the list short of the end reads the next one; a
        # failed page waits for the reader instead.
        if self._status != "ready" or self._loading_more or self._more_failed:
            return
        bar = self.list.verticalScrollBar()
        if near_end(bar.value(), bar.maximum(), bar.pageStep()):
            self._load_more()

    @Slot(int)
    def _list_scrolled(self, _value: int) -> None:
        bar = self.list.verticalScrollBar()
        if self._newest_pending and bar.value() <= bar.minimum():
            self._newest_pending = False
            self._read_newest()
        if near_end(bar.value(), bar.maximum(), bar.pageStep()):
            self._load_more()

    # Live updates ------------------------------------------------------------------

    @Slot()
    def _record_stored(self) -> None:
        if not self._live_timer.isActive():
            self._live_timer.start()

    @Slot()
    def _live_tick(self) -> None:
        # Checked when the read would start, not when the signal came, so a
        # signal that arrived just before a read failed starts nothing either.
        if self._live_suspended:
            return
        # A stored record reaches the list at once while it is scrolled to the
        # top; otherwise it waits until the list is back there, so the list never
        # moves under the reader.
        self._read_sessions()
        bar = self.list.verticalScrollBar()
        if bar.value() <= bar.minimum():
            self._read_newest()
        else:
            self._newest_pending = True

    # Filters -----------------------------------------------------------------------

    def _fill_launches(self) -> None:
        chosen = self.launch_filter.currentData()
        self.launch_filter.blockSignals(True)
        self.launch_filter.clear()
        self.launch_filter.addItem(localizer.t("records.allLaunches"), None)
        sessions = self._sessions
        if chosen is not None and chosen not in sessions:
            sessions = (chosen, *sessions)
        for session in sessions:
            self.launch_filter.addItem(launch_label(session, self._this_session), session)
        self.launch_filter.setCurrentIndex(max(0, self.launch_filter.findData(chosen)))
        self.launch_filter.blockSignals(False)

    @Slot()
    def _search_timer_restart(self) -> None:
        self._search_timer.start()

    @Slot()
    def _search_settled(self) -> None:
        if self.search.text() != self._query.search:
            self._set_query(
                RecordsQuery(self._query.session, self._query.level, self.search.text())
            )

    @Slot()
    def _filters_changed(self) -> None:
        self._set_query(
            RecordsQuery(
                self.launch_filter.currentData(),
                self.level_filter.currentData(),
                self._query.search,
            )
        )

    def _set_query(self, query: RecordsQuery) -> None:
        if query != self._query:
            self._query = query
            self._read_first_page()

    def _sync_end_note(self) -> None:
        if self._status == "ready" and self._loading_more:
            note: Message | None = Message("records.loading")
            self.end_note.setProperty("severity", "")
        elif self._status == "ready" and self._more_failed:
            note = Message("records.loadFailed")
            self.end_note.setProperty("severity", "error")
        else:
            note = None
        repolish(self.end_note)
        if note is None:
            self.end_note.hide()
            unlocalize(self.end_note, "text")
            self.end_note.clear()
            return
        localize(self.end_note, text=note)
        self.end_note.show()

    # Selection and the detail ------------------------------------------------------

    def _keep_selection(self) -> None:
        """The selected record stays selected while it is still listed, and the
        list does not scroll to it."""
        if self._selected_id is None:
            return
        row = self.model.row_of(self._selected_id)
        if row is None:
            return
        index = self.model.index(row)
        if self.list.currentIndex() == index:
            return
        self.list.setAutoScroll(False)
        try:
            self.list.selectionModel().setCurrentIndex(
                index, self.list.selectionModel().SelectionFlag.ClearAndSelect
            )
        finally:
            self.list.setAutoScroll(True)

    @Slot(QModelIndex, QModelIndex)
    def _current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        # The selection follows the keyboard's position (activation follows focus).
        if not current.isValid():
            return
        selection = self.list.selectionModel()
        if not selection.isSelected(current):
            selection.select(current, selection.SelectionFlag.ClearAndSelect)
        record = self.model.records[current.row()]
        if record.id == self._selected_id:
            return
        self._selected_id = record.id
        self._detail = None
        self._detail_note = None
        self._detail_failed = False
        self._render_detail()
        record_id = record.id
        self._read(
            "detail",
            record_id,
            lambda detail: self._detail_read(record_id, detail),
            lambda error: self._detail_read_failed(record_id, error),
        )

    def _detail_read(self, record_id: int, detail: object) -> None:
        if record_id != self._selected_id:
            return
        if isinstance(detail, RecordDetail):
            self._detail = detail
            self._detail_note = None
            self._detail_failed = False
        else:
            self._detail_note = Message("records.detailFailed")
            self._detail_failed = True
        self._render_detail()

    def _detail_read_failed(self, record_id: int, error: object) -> None:
        if record_id != self._selected_id:
            return
        log.warning("records.read_failed", read="detail", failure=error)
        self._detail_note = Message("records.detailFailed")
        self._detail_failed = True
        self._render_detail()

    def _render_detail(self) -> None:
        palette = self.detail.palette()
        tone = surfaces(palette)
        text = palette.color(QPalette.ColorRole.Text)
        quiet = _blend(text, QColor(tone["content"]), 0.68)
        danger = danger_colours(palette)["text"]
        if self._detail is None:
            if self._detail_note is None:
                self.detail.setHtml("")
                return
            colour = danger if self._detail_failed else quiet
            self.detail.setHtml(
                f'<p style="color:{colour}">{html.escape(localizer.of(self._detail_note))}</p>'
            )
            return
        record = self._detail
        level_colour = {"error": danger, "warn": warning_text(palette)}.get(record.level, quiet)
        rows, blocks = detail_sections(record, self._this_session)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        parts = [
            '<p style="margin:0 0 4px 0; font-weight:600; font-size:large">'
            f"{html.escape(record.message)}</p>",
            f'<p style="margin:0 0 12px 0; color:{level_colour}">'
            f"{html.escape(level_label(record.level))}</p>",
            '<table cellspacing="0" cellpadding="0">',
        ]
        for label, value in rows:
            parts.append(
                f'<tr><td style="color:{quiet}; padding:0 16px 4px 0">{html.escape(label)}</td>'
                f'<td style="padding:0 0 4px 0">{html.escape(value)}</td></tr>'
            )
        parts.append("</table>")
        for label, block in blocks:
            parts.append(
                f'<p style="margin:16px 0 4px 0; color:{quiet}; font-weight:600">'
                f"{html.escape(label)}</p>"
            )
            parts.append(
                f"<p style=\"margin:0; white-space:pre-wrap; font-family:'{mono}'\">"
                f"{html.escape(block)}</p>"
            )
        self.detail.setHtml("".join(parts))

    @Slot()
    def _retranslate(self) -> None:
        for index, level in enumerate((None, *LEVEL_FILTERS)):
            key = "records.allLevels" if level is None else LEVEL_FILTER_LABELS[level]
            self.level_filter.setItemText(index, localizer.t(key))
        self._fill_launches()
        self.list.sync_note()
        self.list.viewport().update()
        self._render_detail()


def _blend(colour: QColor, behind: QColor, amount: float) -> str:
    """``colour`` laid over ``behind`` at ``amount``, as an opaque colour rich text can take."""
    return QColor(
        round(colour.red() * amount + behind.red() * (1 - amount)),
        round(colour.green() * amount + behind.green() * (1 - amount)),
        round(colour.blue() * amount + behind.blue() * (1 - amount)),
    ).name()
