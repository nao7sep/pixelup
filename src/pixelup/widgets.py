from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QAccessible,
    QAccessibleEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QToolButton,
    QWidget,
)

from pixelup.devices import DEVICE_CHOICES
from pixelup.paths import OutputFormat

ResultSeverity = Literal["information", "warning", "error"]


class ResultCloseButton(QToolButton):
    """Quiet result close control with toolkit-independent drawn geometry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip("Dismiss")
        self.setAccessibleName("Dismiss result")
        self.setAutoRaise(True)
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QToolButton {"
            " border: none; border-radius: 4px; background: transparent; padding: 0;"
            "}"
            "QToolButton:hover { background: palette(midlight); }"
            "QToolButton:pressed { background: palette(mid); }"
            "QToolButton:focus { border: 1px solid palette(highlight); }"
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override name
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().color(QPalette.ColorRole.ButtonText))
        pen.setWidthF(1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(8, 8), QPointF(16, 16))
        painter.drawLine(QPointF(16, 8), QPointF(8, 16))


class OperationResult(QFrame):
    """A persistent inline result whose severity controls behavior and palette.

    The owner decides where the result lives and when its consequence is resolved.
    This widget owns only presentation. If it is hosted by a dialog, revealing new
    content also re-measures that dialog; otherwise Qt can compress the existing
    controls to make the previously hidden result fit.
    """

    def __init__(
        self,
        *,
        object_name: str,
        dismissible: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._announcement = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 5 if dismissible else 10, 7)
        layout.setSpacing(8)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumWidth(0)
        layout.addWidget(self.message_label, 1)

        self.dismiss_button = ResultCloseButton(self)
        self.dismiss_button.clicked.connect(self.clear_result)
        if dismissible:
            layout.addWidget(self.dismiss_button, 0, Qt.AlignmentFlag.AlignTop)
        else:
            self.dismiss_button.hide()
        self.hide()

    def show_result(
        self,
        message: str,
        *,
        severity: ResultSeverity,
        announce: bool = True,
    ) -> None:
        self.message_label.setText(message)
        self.setAccessibleName(message)
        self._apply_severity_style(severity)
        self.show()

        owner = self.window()
        if isinstance(owner, QDialog):
            owner.adjustSize()

        if announce and message != self._announcement:
            event = (
                QAccessible.Event.NameChanged
                if severity == "information"
                else QAccessible.Event.Alert
            )
            QAccessible.updateAccessibility(QAccessibleEvent(self, event))
        self._announcement = message

    def clear_result(self) -> None:
        self.hide()
        self.message_label.clear()
        self.setAccessibleName("")
        self._announcement = ""

    def _apply_severity_style(self, severity: ResultSeverity) -> None:
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        color = {
            "error": "#ff766a" if dark else "#b3261e",
            "warning": "#f2c14e" if dark else "#8a5a00",
            "information": "palette(mid)",
        }[severity]
        object_name = self.objectName()
        self.setStyleSheet(
            f"QFrame#{object_name} {{"
            f" border: 1px solid {color};"
            " border-radius: 5px;"
            " background: palette(base);"
            "}"
        )


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
