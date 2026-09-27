from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import (
    QAccessible,
    QAccessibleEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QResizeEvent,
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
from pixelup.i18n import localizer
from pixelup.i18n.localized import localize, unlocalize
from pixelup.i18n.message import Message
from pixelup.paths import OutputFormat
from pixelup.session_log import log

ResultSeverity = Literal["information", "warning", "error"]


class ResultCloseButton(QToolButton):
    """Quiet result close control with toolkit-independent drawn geometry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        localize(self, tooltip="result.dismiss", accessible_name="result.dismissAccessible")
        self.setAutoRaise(True)
        self.setFixedSize(24, 24)
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
    This widget owns only presentation. If it is hosted by a dialog, appearing and
    disappearing both re-measure that dialog; otherwise Qt compresses the existing
    controls to make the previously hidden result fit, and a dismissed one leaves
    its gap behind.
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
        self._announcement: Message | None = None

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

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override name
        super().resizeEvent(event)
        self._sync_dismiss_position()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override name
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._sync_dismiss_position()

    def _sync_dismiss_position(self) -> None:
        """Keep the dismiss X centered on the message's first line, without
        changing the banner's height or the message's own position.

        A short, single-line message is already centered on the button: Qt
        stretches the label to the button's height and centers the one line
        within it (both AlignVCenter by default), so their centers coincide
        with no help needed. Wrapped copy is the case that needs one: once
        the label's own natural (wrapped) height exceeds the button's, there
        is no more slack for Qt to center within, so the text block renders
        flush against the label's top instead, and the still-AlignTop button
        is left centered on the whole block rather than the first line.
        Rather than resize anything to fix that, the button is nudged by the
        leftover half-difference so it overlaps into the row's own margin
        (matching the reference Qt guidance in scratchpad/xalign.md); only if
        it would not fit inside that margin does this widen the margin, and
        it says so, since a caller relying on a fixed banner height may want
        to know.
        """
        label = self.message_label
        button = self.dismiss_button
        if button.isHidden():
            return
        layout = self.layout()
        if not isinstance(layout, QHBoxLayout):
            return
        # Forces the row to lay out for the current size/text right now,
        # rather than relying on an event loop iteration this method's own
        # callers (show_result, in particular) cannot guarantee has run yet.
        layout.activate()
        if label.height() <= button.height():
            return  # Qt's own centering already lands the X on the first line
        line_height = label.fontMetrics().lineSpacing()
        # > 0: the button is taller than one line, so it must move UP to
        # reach the first line's center (the common case with a fixed-size
        # icon button and ordinary body text).
        shift = round((button.height() - line_height) / 2)
        if shift == 0:
            return
        margins = layout.contentsMargins()
        available = margins.top() if shift > 0 else margins.bottom()
        if abs(shift) > available:
            grown = abs(shift) - available
            if shift > 0:
                margins.setTop(margins.top() + grown)
            else:
                margins.setBottom(margins.bottom() + grown)
            layout.setContentsMargins(margins)
            log.warning(
                "operation_result.dismiss_button_overflow",
                object_name=self.objectName(),
                shift=shift,
                available_margin=available,
                grown_by=grown,
            )
            layout.activate()
        # The layout just placed the button at its normal AlignTop position
        # for this size; move it from there rather than accumulating deltas
        # across repeated resize/font-change calls.
        button.move(button.x(), button.y() - shift)

    def show_result(
        self,
        message: Message,
        *,
        severity: ResultSeverity,
        announce: bool = True,
    ) -> None:
        # Bound rather than written once, so a result still showing when the
        # language changes is rewritten in the new one.
        localize(self.message_label, text=message)
        localize(self, accessible_name=message)
        self._apply_severity_style(severity)
        self.show()
        self._sync_dismiss_position()

        self._remeasure_owner()

        if announce and message != self._announcement:
            event = (
                QAccessible.Event.NameChanged
                if severity == "information"
                else QAccessible.Event.Alert
            )
            QAccessible.updateAccessibility(QAccessibleEvent(self, event))
        self._announcement = message

    @property
    def message(self) -> Message | None:
        """What the result is saying, or None while it is hidden."""
        return self._announcement or None

    def clear_result(self) -> None:
        self.hide()
        unlocalize(self.message_label, "text")
        unlocalize(self, "accessible_name")
        self.message_label.clear()
        self.setAccessibleName("")
        self._announcement = None
        self._remeasure_owner()

    def _remeasure_owner(self) -> None:
        """Let the hosting dialog take its own size again, now that this changed height.

        A DialogShell is bounded and fixed, so ``adjustSize`` has nothing to do
        there — its own ``fit`` is the arithmetic that knows the screen share. The
        method is looked up rather than imported because the shell is built out of
        this module, so naming it here would close the circle.
        """
        owner = self.window()
        if not isinstance(owner, QDialog):
            return
        fit = getattr(owner, "fit", None)
        if callable(fit):
            fit()
            return
        owner.adjustSize()

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
        empty_message: Message,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(rows, columns, parent)
        self.empty_message = empty_message
        # A collection of rows, not a spreadsheet: no cell grid, whose gaps cut the
        # selected row into blocks; no row numbers; and each heading starts where
        # the values under it start.
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        model = self.model()
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


def choice_combo(choices: tuple[tuple[Message | str, object], ...]) -> NoWheelComboBox:
    """A scroll-safe combo box populated with ``(label, value)`` choices.

    Items carry the value as item data, so callers read selection via
    ``currentData()`` and restore it via ``setCurrentIndex(findData(value))``.
    """
    combo = NoWheelComboBox()
    for label, value in choices:
        combo.addItem(localizer.display(label), value)
    return combo


def retranslate_choices(
    combo: QComboBox, choices: tuple[tuple[Message | str, object], ...]
) -> None:
    """Rewrite a choice combo's labels in the current language, keeping its selection."""
    for index, (label, _value) in enumerate(choices):
        combo.setItemText(index, localizer.display(label))


def device_combo() -> NoWheelComboBox:
    """A scroll-safe combo box populated with the shared device choices."""
    return choice_combo(DEVICE_CHOICES)


def output_format_combo() -> NoWheelComboBox:
    """A scroll-safe combo box populated with the output formats.

    Items carry the lowercase format value as item data (e.g. ``"png"``).
    """
    combo = NoWheelComboBox()
    for fmt in OutputFormat:
        combo.addItem(fmt.value.upper(), fmt.value)
    return combo
