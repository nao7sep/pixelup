from __future__ import annotations

from PySide6.QtWidgets import QApplication

from pixelup.devices import DEVICE_CHOICES
from pixelup.paths import OutputFormat
from pixelup.widgets import device_combo, output_format_combo


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
