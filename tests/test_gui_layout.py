from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel

from pixelup.gui import _fit_columns


def test_fit_columns_sizes_to_widest_sample(qapp: QApplication) -> None:
    label = QLabel()
    one = _fit_columns(label, "x")
    many = _fit_columns(label, "x", "a considerably longer sample string")
    # The widest sample drives the result (plus the fixed cell padding).
    assert many > one > 0
