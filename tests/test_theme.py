"""The app's own appearance contract (see ``pixelup.theme``).

PixelUp follows the OS theme and owns no colours beyond the one destructive red,
so these pin the two things that would quietly break that: a literal colour
creeping into the sheet, and the sizes drifting apart again into the five
different one-line heights the app used to draw.
"""

from __future__ import annotations

import re

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton

from pixelup import theme


def _palette(window: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    palette.setColor(QPalette.ColorRole.Text, QColor(text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(text))
    return palette


LIGHT = _palette("#efefef", "#000000")
DARK = _palette("#323232", "#f0f0f0")


def test_dark_is_decided_by_the_window_lightness() -> None:
    assert theme.is_dark(DARK)
    assert not theme.is_dark(LIGHT)


def test_each_theme_gets_its_own_destructive_red() -> None:
    light, dark = theme.danger_colours(LIGHT), theme.danger_colours(DARK)
    assert light != dark
    for colours in (light, dark):
        assert set(colours) == {"text", "fill", "fill_hover", "ink"}


def test_the_sheet_names_no_colour_of_its_own_beyond_the_destructive_red() -> None:
    """Every other colour must come from the palette, so the OS theme decides it."""
    for palette in (LIGHT, DARK):
        sheet = theme.build_stylesheet(palette)
        literals = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", sheet))
        allowed = set(theme.danger_colours(palette).values())
        assert literals <= allowed, f"unexpected literal colours: {sorted(literals - allowed)}"
        assert "palette(" in sheet


def test_one_height_for_every_one_line_control(qapp: QApplication, tmp_path) -> None:
    """A text box, a number box, a combo and a button beside them line up."""
    marks = theme.write_arrow_marks(qapp.palette(), tmp_path / "marks")
    qapp.setStyleSheet(theme.build_stylesheet(qapp.palette(), marks))
    try:
        widgets = [QLineEdit(), QComboBox(), QDoubleSpinBox(), QPushButton("Go")]
        for widget in widgets:
            widget.ensurePolished()
        heights = {type(w).__name__: w.sizeHint().height() for w in widgets}
        assert set(heights.values()) == {theme.CONTROL_HEIGHT}, heights
        for widget in widgets:
            widget.deleteLater()
    finally:
        qapp.setStyleSheet("")


def test_fields_are_left_to_the_toolkit_when_their_marks_cannot_be_written() -> None:
    """A field with no arrow is worse than a native one, so the sheet backs off."""
    sheet = theme.build_stylesheet(LIGHT, marks=None)
    assert "QComboBox" not in sheet
    assert "QLineEdit" not in sheet
    assert "QPushButton" in sheet


def test_arrow_marks_are_written_per_theme(tmp_path) -> None:
    marks = theme.write_arrow_marks(DARK, tmp_path / "marks")
    assert marks is not None
    assert set(marks) == {"down", "down-disabled", "up", "up-disabled"}
    for path in marks.values():
        assert path.exists() and path.stat().st_size > 0
    # The mark is the theme's own text colour, so the two themes cannot share a file.
    light = theme.write_arrow_marks(LIGHT, tmp_path / "marks")
    assert light is not None
    assert set(light.values()).isdisjoint(set(marks.values()))


def test_write_arrow_marks_reports_failure_instead_of_raising(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    assert theme.write_arrow_marks(LIGHT, blocked / "marks") is None
