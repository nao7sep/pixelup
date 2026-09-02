from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication, QDialog, QTableWidgetItem, QVBoxLayout

from pixelup.devices import DEVICE_CHOICES
from pixelup.paths import OutputFormat
from pixelup.widgets import (
    EmptyStateTableWidget,
    OperationResult,
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


def test_empty_state_table_tracks_zero_to_one_and_one_to_zero(qapp: QApplication) -> None:
    table = EmptyStateTableWidget(0, 1, empty_text="Nothing here yet.")
    try:
        table.show()
        table.setFocus()
        qapp.processEvents()
        assert table.empty_state_visible is True
        assert table.accessibleDescription() == "Nothing here yet."
        assert table.hasFocus() is True

        table.insertRow(0)
        table.setItem(0, 0, QTableWidgetItem("First row"))
        assert table.empty_state_visible is False
        assert table.accessibleDescription() == ""
        assert table.hasFocus() is True

        table.removeRow(0)
        assert table.empty_state_visible is True
        assert table.accessibleDescription() == "Nothing here yet."
        assert table.hasFocus() is True
    finally:
        table.deleteLater()


def test_device_combo_carries_value_as_item_data(qapp: QApplication) -> None:
    combo = device_combo()
    try:
        pairs = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        assert pairs == list(DEVICE_CHOICES)
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
