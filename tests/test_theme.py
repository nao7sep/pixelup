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


def _luminance(colour: QColor) -> float:
    channels = []
    for value in (colour.redF(), colour.greenF(), colour.blueF()):
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(first: QColor, second: QColor) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


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
        assert set(colours) == {
            "text", "fill", "fill_hover", "fill_pressed", "ink",
            "text_disabled", "fill_disabled", "ink_disabled",
        }


def test_white_stays_legible_on_the_destructive_fill_it_is_pressed_on() -> None:
    """The dark fill deepens on press rather than lifting, so the label holds."""
    for palette in (LIGHT, DARK):
        colours = theme.danger_colours(palette)
        for key in ("fill", "fill_hover", "fill_pressed"):
            assert _contrast(QColor(colours["ink"]), QColor(colours[key])) >= 4.5, key


def test_a_disabled_destructive_control_keeps_its_red() -> None:
    """Red says the action destroys something, which an unavailable action still does."""
    for palette in (LIGHT, DARK):
        colours = theme.danger_colours(palette)
        window = palette.color(QPalette.ColorRole.Window)
        for live, off in (("text", "text_disabled"), ("fill", "fill_disabled")):
            faded = QColor(colours[off])
            assert faded != QColor(colours[live])
            # Receded towards the window, not replaced by it or by a grey.
            assert faded != window
            assert faded.saturation() > 0


def test_the_accent_press_is_derived_from_the_palette_not_chosen() -> None:
    """A style sheet has no colour arithmetic, so the step is computed — but from
    the OS accent, so a different accent still moves it."""
    blue, green = _palette("#efefef", "#000000"), _palette("#efefef", "#000000")
    blue.setColor(QPalette.ColorRole.Highlight, QColor("#2f6fd0"))
    green.setColor(QPalette.ColorRole.Highlight, QColor("#2f9d55"))
    assert theme.accent_pressed(blue) != theme.accent_pressed(green)
    for palette in (blue, green):
        highlight = palette.color(QPalette.ColorRole.Highlight)
        pressed = QColor(theme.accent_pressed(palette))
        assert pressed != highlight
        assert pressed.value() < highlight.value()


def test_the_sheet_names_no_colour_of_its_own_beyond_the_destructive_red() -> None:
    """Every other colour must come from the palette, so the OS theme decides it."""
    for palette in (LIGHT, DARK):
        sheet = theme.build_stylesheet(palette)
        literals = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", sheet))
        # The accent's pressed step is computed rather than named, because a style
        # sheet cannot darken a colour; it still comes from palette(highlight).
        allowed = set(theme.danger_colours(palette).values()) | {theme.accent_pressed(palette)}
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


def test_every_button_role_answers_a_press_and_a_disable(qapp: QApplication, tmp_path) -> None:
    """Forced and rendered, because neither state shows in a resting window.

    Left unsaid, a role falls through to the standard button's rules: the primary
    and the destructive confirm both turned the toolkit's grey at the moment of
    the click, and the destructive trigger — which the main window really does
    switch off while a job runs on the selected image — drew exactly as it does
    when it is live.
    """
    marks = theme.write_arrow_marks(qapp.palette(), tmp_path / "marks")
    qapp.setStyleSheet(theme.build_stylesheet(qapp.palette(), marks))
    try:
        roles = (("role", "primary"), ("role", "danger"), ("role", "danger-confirm"), (None, None))
        for role, value in roles:
            resting, pressed, off = (QPushButton("Remove") for _ in range(3))
            for button in (resting, pressed, off):
                if role is not None:
                    button.setProperty(role, value)
                button.resize(120, theme.CONTROL_HEIGHT)
                button.ensurePolished()
            pressed.setDown(True)
            off.setEnabled(False)

            def surface(button: QPushButton) -> tuple[str, str, str]:
                image = button.grab().toImage()
                middle = button.height() // 2
                fill = image.pixelColor(8, middle)
                ink = max(
                    (image.pixelColor(x, middle) for x in range(10, button.width() - 10)),
                    key=lambda pixel: abs(pixel.lightness() - fill.lightness()),
                )
                return fill.name(), image.pixelColor(button.width() // 2, 0).name(), ink.name()

            named = value or "standard"
            assert surface(pressed) != surface(resting), f"{named} says nothing when pressed"
            assert surface(off) != surface(resting), f"{named} says nothing when disabled"
            for button in (resting, pressed, off):
                button.deleteLater()
    finally:
        qapp.setStyleSheet("")
