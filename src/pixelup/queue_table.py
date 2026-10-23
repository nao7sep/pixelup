"""The queue's table model reads the existing jobs only when Qt paints a cell."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import QTableView, QWidget

from pixelup.i18n import localizer
from pixelup.i18n.message import Message
from pixelup.jobs import Job

QUEUE_COLUMNS = (
    "queue.columnImage",
    "queue.columnModel",
    "queue.columnScale",
    "queue.columnOutput",
    "queue.columnStatus",
)


class QueueTableModel(QAbstractTableModel):
    """One reference per job, with no cell objects or cached display strings."""

    def __init__(
        self,
        *,
        status_message: Callable[[str], Message],
        scale_label: Callable[[int], str],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs: list[Job] = []
        self._rows: dict[int, int] = {}
        self._status_message = status_message
        self._scale_label = scale_label
        localizer.changed.connect(self.retranslate)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._jobs)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(QUEUE_COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
            and 0 <= section < len(QUEUE_COLUMNS)
        ):
            return localizer.t(QUEUE_COLUMNS[section])
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole
        ):
            return None
        job = self._jobs[index.row()]
        column = index.column()
        tooltip = role == Qt.ItemDataRole.ToolTipRole
        if column == 0:
            return str(job.input_path) if tooltip else job.input_path.name
        if column == 1:
            return job.model
        if column == 2:
            return None if tooltip else self._scale_label(job.settings.scale)
        if column == 3:
            return str(job.output_path) if tooltip else job.output_path.name
        status = self._status_message(job.status) if job.status == "cancelling" else (
            job.message or self._status_message(job.status)
        )
        status_text = localizer.of(status)
        if tooltip:
            return "\n".join((status_text, *(localizer.of(warning) for warning in job.warnings)))
        return status_text

    def append(self, jobs: Sequence[Job]) -> None:
        if not jobs:
            return
        first = len(self._jobs)
        self.beginInsertRows(QModelIndex(), first, first + len(jobs) - 1)
        self._jobs.extend(jobs)
        for row, job in enumerate(jobs, first):
            self._rows[job.id] = row
        self.endInsertRows()

    def refresh(self, job: Job) -> None:
        row = self._rows[job.id]
        self.dataChanged.emit(
            self.index(row, 3), self.index(row, 4),
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
        )

    def retranslate(self) -> None:
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(QUEUE_COLUMNS) - 1)
        if self._jobs:
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self._jobs) - 1, len(QUEUE_COLUMNS) - 1),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole],
            )


class QueueTableView(QTableView):
    """A native table view that paints the queue's message while there are no jobs."""

    def __init__(self, model: QueueTableModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModel(model)
        self.empty_message = Message("queue.empty")
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        model.rowsInserted.connect(self._sync_empty_state)
        model.rowsRemoved.connect(self._sync_empty_state)
        model.modelReset.connect(self._sync_empty_state)
        localizer.changed.connect(self._sync_empty_state)
        self._sync_empty_state()

    @property
    def empty_text(self) -> str:
        return localizer.of(self.empty_message)

    @property
    def empty_state_visible(self) -> bool:
        return self.model().rowCount() == 0

    def _sync_empty_state(self, *_args: object) -> None:
        self.setAccessibleDescription(self.empty_text if self.empty_state_visible else "")
        self.viewport().update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self.empty_state_visible:
            return
        with QPainter(self.viewport()) as painter:
            painter.setFont(self.font())
            color = self.palette().color(QPalette.ColorRole.Text)
            color.setAlphaF(0.68)
            painter.setPen(color)
            painter.drawText(
                self.viewport().rect().adjusted(16, 16, -16, -16),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.empty_text,
            )
