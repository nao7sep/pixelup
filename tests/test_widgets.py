from __future__ import annotations

from PySide6.QtWidgets import QApplication, QTableWidgetItem

from pixelup.devices import DEVICE_CHOICES
from pixelup.paths import OutputFormat
from pixelup.widgets import EmptyStateTableWidget, device_combo, output_format_combo


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
