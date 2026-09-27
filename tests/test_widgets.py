from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QTableWidgetItem,
    QVBoxLayout,
)

from pixelup.devices import DEVICE_CHOICES
from pixelup.i18n.message import Message
from pixelup.paths import OutputFormat
from pixelup.widgets import (
    EmptyStateTableWidget,
    OperationResult,
    ResultCloseButton,
    device_combo,
    output_format_combo,
)


def test_operation_result_uses_severity_for_behavior_without_repeating_it_in_copy(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    monkeypatch.setattr("pixelup.widgets.QAccessible.updateAccessibility", events.append)
    result = OperationResult(object_name="testResult", dismissible=True)
    try:
        result.show_result("Could not open the image.", severity="error")

        assert result.message_label.text() == "Could not open the image."
        assert result.accessibleName() == "Could not open the image."
        assert result.dismiss_button.text() == ""
        assert result.dismiss_button.accessibleName() == "Dismiss result"
        assert result.dismiss_button.autoRaise() is True
        assert events[0].type() == QAccessible.Event.Alert  # type: ignore[union-attr]

        result.show_result("Could not open the image.", severity="error")
        assert len(events) == 1

        result.show_result("Already open.", severity="information")
        assert result.accessibleName() == "Already open."
        assert events[-1].type() == QAccessible.Event.NameChanged  # type: ignore[union-attr]
    finally:
        result.deleteLater()


def test_operation_result_keeps_dismiss_at_the_upper_end_of_wrapping_copy(
    qapp: QApplication,
) -> None:
    result = OperationResult(object_name="wrappingResult", dismissible=True)
    try:
        result.show_result("Select at least one model before queueing.", severity="warning")

        layout = result.layout()
        assert layout is not None
        assert layout.indexOf(result.dismiss_button) == 1
        assert layout.itemAt(1).alignment() == Qt.AlignmentFlag.AlignTop
    finally:
        result.deleteLater()


@pytest.mark.parametrize(
    "text",
    [
        "Model ready.",
        "Select at least one model before queueing this batch, since nothing "
        "has been chosen yet, so no work can start.",
    ],
    ids=["one-line", "wrapped"],
)
def test_operation_result_centers_dismiss_on_the_first_line_not_the_block(
    qapp: QApplication, text: str
) -> None:
    """The X sits on the message's first line, one line or wrapped alike —
    never drifting toward the middle of a taller, wrapped block — and doing so
    never changes the banner's height or the message's own (unmoved) position.
    See scratchpad/redesign/xalign/pixelup for real-render screenshots and
    measured offsets (both within 1px)."""
    result = OperationResult(object_name="xalignResult", dismissible=True)
    try:
        result.show_result(text, severity="warning")
        result.resize(360, result.sizeHint().height())
        result.show()
        qapp.processEvents()
        qapp.processEvents()

        label = result.message_label
        button = result.dismiss_button
        layout = result.layout()
        assert layout is not None
        margins = layout.contentsMargins()

        # The fix never touches the label's own alignment/margins or the
        # row's margins (for ordinary body text the button fits inside the
        # existing 7px top margin) -- only the button moves.
        assert label.alignment() == (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        assert label.contentsMargins().top() == 0
        assert (margins.top(), margins.bottom()) == (7, 7)

        # The banner's height is exactly what the label/button would need on
        # their own (Qt's own heightForWidth, the same input the layout used
        # to size the label) -- nothing was grown to make room for the button.
        fm = label.fontMetrics()
        block_h = label.heightForWidth(label.width())
        # +2: the severity style's 1px top/bottom border (outside the layout's
        # own margins, via the stylesheet rather than QFrame's frameWidth).
        expected_height = margins.top() + margins.bottom() + max(button.height(), block_h) + 2
        assert result.height() == expected_height

        # And the X is on the message's first line, not the wrapped block's
        # middle: same slack-splitting math QLabel itself uses to center a
        # short line, applied here to find that first line's own center.
        line_height = fm.lineSpacing()
        content_h = label.height()
        slack = max(0, content_h - block_h)
        first_line_center = label.y() + slack / 2 + line_height / 2
        button_center = button.y() + button.height() / 2
        assert abs(button_center - first_line_center) <= 1
    finally:
        result.deleteLater()


def test_dialog_remeasures_when_a_hidden_result_appears(qapp: QApplication) -> None:
    class CountingDialog(QDialog):
        def __init__(self) -> None:
            super().__init__()
            self.adjust_count = 0

        def adjustSize(self) -> None:  # noqa: N802 - Qt override name
            self.adjust_count += 1
            super().adjustSize()

    dialog = CountingDialog()
    layout = QVBoxLayout(dialog)
    result = OperationResult(object_name="dialogResult", dismissible=True)
    layout.addWidget(result)
    try:
        result.show_result(
            "The models could not be installed. Your current model selection is unchanged.",
            severity="error",
        )

        assert dialog.adjust_count == 1
    finally:
        dialog.deleteLater()


def test_empty_state_table_draws_rows_not_cells(qapp: QApplication) -> None:
    # No cell grid, whose gaps cut a selected row into blocks; no row numbers; and
    # each heading starts where the values under it start.
    table = EmptyStateTableWidget(0, 2, empty_message=Message("queue.empty"))
    try:
        assert table.showGrid() is False
        assert table.verticalHeader().isHidden() is True
        alignment = table.horizontalHeader().defaultAlignment()
        assert alignment & Qt.AlignmentFlag.AlignLeft
    finally:
        table.deleteLater()


def test_empty_state_table_tracks_zero_to_one_and_one_to_zero(qapp: QApplication) -> None:
    table = EmptyStateTableWidget(0, 1, empty_message=Message("queue.empty"))
    try:
        table.show()
        table.setFocus()
        qapp.processEvents()
        assert table.empty_state_visible is True
        assert table.accessibleDescription() == "No jobs queued yet."
        assert table.hasFocus() is True

        table.insertRow(0)
        table.setItem(0, 0, QTableWidgetItem("First row"))
        assert table.empty_state_visible is False
        assert table.accessibleDescription() == ""
        assert table.hasFocus() is True

        table.removeRow(0)
        assert table.empty_state_visible is True
        assert table.accessibleDescription() == "No jobs queued yet."
        assert table.hasFocus() is True
    finally:
        table.deleteLater()


def test_device_combo_carries_value_as_item_data(qapp: QApplication) -> None:
    combo = device_combo()
    try:
        pairs = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        assert pairs == [("Auto", "auto"), ("MPS", "mps"), ("CUDA", "cuda"), ("CPU", "cpu")]
        # Every stored value round-trips through findData -> currentData.
        for _label, value in DEVICE_CHOICES:
            combo.setCurrentIndex(combo.findData(value))
            assert combo.currentData() == value
    finally:
        combo.deleteLater()


def test_output_format_combo_carries_lowercase_value_as_item_data(qapp: QApplication) -> None:
    combo = output_format_combo()
    try:
        values = [combo.itemData(i) for i in range(combo.count())]
        assert values == [fmt.value for fmt in OutputFormat]
        for fmt in OutputFormat:
            combo.setCurrentIndex(combo.findData(fmt.value))
            assert OutputFormat(combo.currentData()) == fmt
    finally:
        combo.deleteLater()


def test_result_close_button_keeps_the_platform_arrow_cursor(qapp: QApplication) -> None:
    button = ResultCloseButton()
    try:
        assert button.cursor().shape() == Qt.CursorShape.ArrowCursor
    finally:
        button.deleteLater()
