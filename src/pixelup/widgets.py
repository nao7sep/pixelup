from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette, QWheelEvent
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QWidget

from pixelup.devices import DEVICE_CHOICES
from pixelup.paths import OutputFormat


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class EmptyStateTableWidget(QTableWidget):
    """A normal Qt table that paints a message in its viewport while it has no rows.

    The headers, focus handling, selection model, font, and theme remain owned by
    QTableWidget; there is no parallel overlay widget to keep in sync.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        empty_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(rows, columns, parent)
        self.empty_text = empty_text
        model = self.model()
        model.rowsInserted.connect(self._sync_empty_state)
        model.rowsRemoved.connect(self._sync_empty_state)
        model.modelReset.connect(self._sync_empty_state)
        self._sync_empty_state()

    @property
    def empty_state_visible(self) -> bool:
        return self.rowCount() == 0

    def _sync_empty_state(self, *_args: object) -> None:
        self.setAccessibleDescription(self.empty_text if self.empty_state_visible else "")
        self.viewport().update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self.empty_state_visible:
            return

        painter = QPainter(self.viewport())
        painter.setFont(self.font())
        # Start from the table's guaranteed-readable theme foreground rather than
        # relying on every host to correct PlaceholderText, then soften it to secondary.
        color = self.palette().color(QPalette.ColorRole.Text)
        color.setAlphaF(0.68)
        painter.setPen(color)
        painter.drawText(
            self.viewport().rect().adjusted(16, 16, -16, -16),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self.empty_text,
        )


def device_combo() -> NoWheelComboBox:
    """A scroll-safe combo box populated with the shared device choices.

    Items carry the device value as item data, so callers read selection via
    ``currentData()`` and restore it via ``setCurrentIndex(findData(value))``.
    """
    combo = NoWheelComboBox()
    for label, value in DEVICE_CHOICES:
        combo.addItem(label, value)
    return combo


def output_format_combo() -> NoWheelComboBox:
    """A scroll-safe combo box populated with the output formats.

    Items carry the lowercase format value as item data (e.g. ``"png"``).
    """
    combo = NoWheelComboBox()
    for fmt in OutputFormat:
        combo.addItem(fmt.value.upper(), fmt.value)
    return combo
