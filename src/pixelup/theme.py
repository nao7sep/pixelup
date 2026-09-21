"""PixelUp's one owned appearance: the sizes every control is built from, and the
style sheet that draws them.

PixelUp owns no colours of its own. It follows the OS light/dark theme through
Fusion's palette (see ``build_app``), so every colour here is a ``palette(...)``
reference rather than a literal, and the one treatment that has no palette role —
a destructive action's red — is the single pair of values in ``DANGER``, chosen
per theme the way the warning button already is.

What this module does own is geometry and state: one height for a standard
control and one for a compact one, one corner radius, and a hover, pressed,
focused and disabled treatment for each control, so nothing is left to the
toolkit's defaults. Sizes are named so a change moves the whole app.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPixmap, QPolygon, QPolygonF
from PySide6.QtWidgets import QApplication

from pixelup.config import resolve_state_dir

# One standard height for a one-line control — a text field, a combo box, a spin
# box or a button beside them — and one compact height for a dense row. Before
# this, the app drew one-line controls at 19, 21, 24, 25 and 26 pixels.
CONTROL_HEIGHT = 30
COMPACT_HEIGHT = 26
RADIUS = 6
# The radius of something nested inside a control (a tick, a chip) stays smaller
# than its container's, per the fleet styling conventions.
INNER_RADIUS = 4
# The combo/spin mark, in logical pixels.
ARROW_WIDTH = 9
ARROW_HEIGHT = 6
# A tick's or a radio's box, inside its own border. Both take the same box so a
# column of checkboxes and radios lines up on one edge.
INDICATOR_SIZE = 14
BORDER_WIDTH = 1
# Qt's min-height is the content box, so the border is subtracted to make the
# named height the one the control actually occupies.
_INNER = CONTROL_HEIGHT - 2 * BORDER_WIDTH
_INNER_COMPACT = COMPACT_HEIGHT - 2 * BORDER_WIDTH
# A spin box reserves extra vertical room for its two stacked buttons on top of
# the border, so asking for the same content height leaves it taller than every
# other field. Its own min-height drops by that reservation instead. The value is
# the toolkit's, not a taste: the one-height test pins the rendered result, so a
# change in Qt's reservation fails there rather than drifting in the interface.
_SPIN_BUTTON_RESERVE = 5
_INNER_SPIN = _INNER - _SPIN_BUTTON_RESERVE


def is_dark(palette: QPalette) -> bool:
    """Whether the resolved theme is the dark one, by the window's own lightness."""
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def _receded(colour: QColor, window: QColor) -> str:
    """``colour`` drawn faded over ``window``, as the same colour at 45 per cent.

    A style sheet cannot fade a widget, so the fade is computed here instead. It is
    a proportion of whatever the colour already is, so it lands the same in both
    themes without a second palette to keep in step.
    """
    amount = 0.45
    blended = QColor(
        round(colour.red() * amount + window.red() * (1 - amount)),
        round(colour.green() * amount + window.green() * (1 - amount)),
        round(colour.blue() * amount + window.blue() * (1 - amount)),
    )
    return blended.name()


def accent_pressed(palette: QPalette) -> str:
    """The accent one step beyond its hover, for the primary button's press.

    Derived from ``palette(highlight)`` rather than chosen here, so the primary
    still follows the OS accent — a style sheet has no colour arithmetic, which is
    the only reason this is computed in Python at all. It deepens in both themes,
    the direction the standard button's own press already moves.
    """
    return palette.color(QPalette.ColorRole.Highlight).darker(118).name()


def accent_disabled(palette: QPalette) -> dict[str, str]:
    """The accent receded, for a primary control that is switched off.

    A switched-off primary still has a shape and still says which action it is —
    the same argument the destructive red already makes for itself. Qt's own
    ``midlight`` was doing neither: it resolves to within two points of the
    window in the dark theme, so the button it filled had no visible edge at all.
    """
    window = palette.color(QPalette.ColorRole.Window)
    fill = _receded(palette.color(QPalette.ColorRole.Highlight), window)
    ink = _receded(palette.color(QPalette.ColorRole.HighlightedText), QColor(fill))
    return {"fill": fill, "ink": ink}


def danger_colours(palette: QPalette) -> dict[str, str]:
    """The one set of colours the app owns, for destructive actions.

    Qt's palette has no error role, so these are named here: a red that can be read
    as letters on the theme's own surface, a fill for the confirming button, and
    the ink that sits on that fill. The dark theme's red is a lighter one, so its
    ink turns dark rather than staying white.

    The fill deepens on press in both themes. In dark that is against the hover's
    own direction, and deliberately: the hover already lifts the fill to where
    white sits at 4.7:1, so a further lift would drop the label under the floor.

    The disabled values are each colour faded over the window, so a destructive
    control that is off keeps its red — red is what says the action destroys
    something, and that does not stop being true while the action is unavailable.
    """
    window = palette.color(QPalette.ColorRole.Window)
    if is_dark(palette):
        colours = {
            "text": "#ff9ba6", "fill": "#b93650",
            "fill_hover": "#c8445e", "fill_pressed": "#a52d45", "ink": "#ffffff",
        }
    else:
        colours = {
            "text": "#a11f34", "fill": "#b42318",
            "fill_hover": "#96190f", "fill_pressed": "#7d140c", "ink": "#ffffff",
        }
    colours["text_disabled"] = _receded(QColor(colours["text"]), window)
    colours["fill_disabled"] = _receded(QColor(colours["fill"]), window)
    colours["ink_disabled"] = _receded(QColor(colours["ink"]), window)
    return colours


def _arrow_pixmap(colour: QColor, *, pointing_down: bool) -> QPixmap:
    """A small solid triangle in ``colour``, drawn at 2x for a sharp mark."""
    width, height, scale = ARROW_WIDTH, ARROW_HEIGHT, 2
    pixmap = QPixmap(width * scale, height * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    # The device pixel ratio already scales the painting, so the polygon is drawn
    # in logical coordinates; multiplying here too would draw outside the box.
    points = (
        [QPoint(0, 0), QPoint(width, 0), QPoint(width // 2, height)]
        if pointing_down
        else [QPoint(0, height), QPoint(width, height), QPoint(width // 2, 0)]
    )
    painter.drawPolygon(QPolygon(points))
    painter.end()
    return pixmap


def _check_pixmap(colour: QColor) -> QPixmap:
    """The tick inside a set checkbox, drawn at 2x for a sharp mark.

    A stroke rather than a filled glyph, so it stays a tick at any size and needs
    no font to be installed.
    """
    scale = 2
    pixmap = QPixmap(INDICATOR_SIZE * scale, INDICATOR_SIZE * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(colour)
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(
        QPolygonF([QPointF(3.0, 7.3), QPointF(5.7, 10.0), QPointF(11.0, 4.3)])
    )
    painter.end()
    return pixmap


def _dot_pixmap(colour: QColor) -> QPixmap:
    """The dot inside a chosen radio, on the same box the tick uses."""
    scale = 2
    pixmap = QPixmap(INDICATOR_SIZE * scale, INDICATOR_SIZE * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    centre = INDICATOR_SIZE / 2
    painter.drawEllipse(QPointF(centre, centre), 3.0, 3.0)
    painter.end()
    return pixmap


def write_control_marks(palette: QPalette, directory: Path) -> dict[str, Path] | None:
    """Write every mark a style sheet cannot draw for this theme, or None if it cannot.

    A style sheet draws no triangle, tick or dot — those sub-controls take an
    image — and an image cannot read a palette role, so each mark is rendered here
    in the theme's own colours and cached beside the app's state. Once a sheet
    styles one of these controls the toolkit stops drawing its mark, so a failed
    write leaves every one of them unstyled instead: a native combo is better than
    one with no arrow, and a native checkbox better than a blank blue square.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "dark" if is_dark(palette) else "light"
        marks: dict[str, Path] = {}
        states = (("", QPalette.ColorRole.Text), ("-disabled", QPalette.ColorRole.PlaceholderText))
        for name, pointing_down in (("down", True), ("up", False)):
            for state, role in states:
                colour = palette.color(role)
                path = directory / f"arrow-{name}{state}-{suffix}.png"
                if not _arrow_pixmap(colour, pointing_down=pointing_down).save(str(path), "PNG"):
                    return None
                marks[f"{name}{state}"] = path

        # A set indicator is filled with the accent, so its mark is the ink that
        # goes on the accent — and receded with the fill when the control is off,
        # so the tick fades with the box rather than standing out of a dead one.
        ink = palette.color(QPalette.ColorRole.HighlightedText)
        ink_disabled = QColor(accent_disabled(palette)["ink"])
        for name, draw in (("check", _check_pixmap), ("radio", _dot_pixmap)):
            for state, colour in (("", ink), ("-disabled", ink_disabled)):
                path = directory / f"{name}{state}-{suffix}.png"
                if not draw(colour).save(str(path), "PNG"):
                    return None
                marks[f"{name}{state}"] = path
        return marks
    except OSError:
        return None


def _field_rules(marks: dict[str, Path]) -> str:
    """Fields, including the arrow marks, which only exist when they were written."""
    return f"""
/* Fields. One height for every one-line field, so a text box, a number box and a
   combo in the same column line up, and a button beside them matches. */
QLineEdit, QComboBox {{
    min-height: {_INNER}px;
}}
QSpinBox, QDoubleSpinBox {{
    min-height: {_INNER_SPIN}px;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    padding: 0 8px;
    border: {BORDER_WIDTH}px solid palette(mid);
    border-radius: {RADIUS}px;
    background-color: palette(base);
    color: palette(text);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: palette(highlight);
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: palette(placeholder-text);
    background-color: palette(window);
}}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
/* A spin box holds its own QLineEdit, which would otherwise take the field rule
   above a second time and stack its height inside the box. */
QAbstractSpinBox QLineEdit {{
    min-height: 0;
    border: none;
    padding: 0;
    background: transparent;
}}
QComboBox {{ padding-right: 26px; }}

/* Once a sheet draws these, the toolkit stops drawing their arrows, and a sheet
   cannot draw a triangle itself — so the app hands it the marks rendered above
   in the theme's own text colour. */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    background: transparent;
}}
/* Without an explicit size the sub-control stretches the mark to its own box. */
QComboBox::down-arrow,
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: {ARROW_WIDTH}px;
    height: {ARROW_HEIGHT}px;
}}
QComboBox::down-arrow {{ image: url({marks["down"].as_posix()}); }}
QComboBox::down-arrow:disabled {{ image: url({marks["down-disabled"].as_posix()}); }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border: none;
    background: transparent;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({marks["up"].as_posix()}); }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({marks["down"].as_posix()}); }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
    image: url({marks["up-disabled"].as_posix()});
}}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: url({marks["down-disabled"].as_posix()});
}}
"""


def _indicator_rules(palette: QPalette, marks: dict[str, Path]) -> str:
    """Ticks and radios, including the marks, which only exist when they were written."""
    off = accent_disabled(palette)
    radius = INDICATOR_SIZE // 2 + BORDER_WIDTH
    return f"""
/* Ticks and radios take a comfortable hit target and the accent when set. Once a
   sheet gives the indicator a border the toolkit stops drawing the tick and the
   dot inside it, so both are handed in as marks — without them a set checkbox is
   a blank square of accent, which is what this app shipped. */
QCheckBox, QRadioButton {{
    spacing: 8px;
    min-height: {COMPACT_HEIGHT}px;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: palette(placeholder-text);
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: {INDICATOR_SIZE}px;
    height: {INDICATOR_SIZE}px;
    border: {BORDER_WIDTH}px solid palette(mid);
    background-color: palette(base);
}}
QCheckBox::indicator {{ border-radius: {INNER_RADIUS}px; }}
/* Half the box plus its border, so the ring is a circle rather than the rounded
   square a smaller radius drew. */
QRadioButton::indicator {{ border-radius: {radius}px; }}
QCheckBox::indicator:hover:!disabled, QRadioButton::indicator:hover:!disabled {{
    border-color: palette(highlight);
}}
QCheckBox:focus::indicator, QRadioButton:focus::indicator {{
    border-color: palette(highlight);
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: palette(highlight);
    border-color: palette(highlight);
}}
QCheckBox::indicator:checked {{ image: url({marks["check"].as_posix()}); }}
QRadioButton::indicator:checked {{ image: url({marks["radio"].as_posix()}); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: palette(window);
}}
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {{
    background-color: {off["fill"]};
    border-color: {off["fill"]};
}}
QCheckBox::indicator:checked:disabled {{ image: url({marks["check-disabled"].as_posix()}); }}
QRadioButton::indicator:checked:disabled {{ image: url({marks["radio-disabled"].as_posix()}); }}
"""


def build_stylesheet(palette: QPalette, marks: dict[str, Path] | None = None) -> str:
    """The app-wide sheet. Every colour is a palette reference except DANGER.

    Without ``marks`` the fields, ticks and radios are left alone entirely, so the
    toolkit keeps drawing them and their marks; everything else is styled either
    way.
    """
    danger = danger_colours(palette)
    off = accent_disabled(palette)
    fields = _field_rules(marks) if marks else ""
    indicators = _indicator_rules(palette, marks) if marks else ""
    return f"""
/* Buttons. A standard button is the app's utility role: the palette's own button
   surface, a visible edge in both themes, and its own hover, pressed, focus and
   disabled states rather than the toolkit's. */
QPushButton {{
    min-height: {_INNER}px;
    padding: 0 14px;
    border: {BORDER_WIDTH}px solid palette(mid);
    border-radius: {RADIUS}px;
    background-color: palette(button);
    color: palette(button-text);
}}
QPushButton:hover:!disabled {{
    background-color: palette(midlight);
}}
QPushButton:pressed:!disabled {{
    background-color: palette(mid);
}}
QPushButton:focus {{
    border-color: palette(highlight);
}}
/* A switched-off button keeps its edge and recedes by its label. Qt's own
   midlight sits within two points of the window in the dark theme, so the border
   it used to draw here simply was not there. */
QPushButton:disabled {{
    color: palette(placeholder-text);
    border-color: palette(mid);
}}

/* The primary role: the one action a surface is really for. It takes the OS
   accent from the palette, so the app still owns no colour of its own. */
QPushButton[role="primary"] {{
    background-color: palette(highlight);
    border-color: palette(highlight);
    color: palette(highlighted-text);
    font-weight: 600;
}}
QPushButton[role="primary"]:hover:!disabled {{
    background-color: palette(highlight);
    border-color: palette(text);
}}
/* Without a press of its own it fell through to the standard button's, which
   replaced the accent with the toolkit's grey at the moment of the click. */
QPushButton[role="primary"]:pressed:!disabled {{
    background-color: {accent_pressed(palette)};
    border-color: palette(text);
}}
QPushButton[role="primary"]:disabled {{
    background-color: {off["fill"]};
    border-color: {off["fill"]};
    color: {off["ink"]};
}}

/* Destructive actions, in the fleet's two roles: a trigger that opens a
   destructive path is outlined, and the confirming button of the dialog that
   asks is filled, so a solid red always means this is the last step. */
QPushButton[role="danger"] {{
    color: {danger["text"]};
    border-color: {danger["text"]};
    background-color: palette(button);
}}
QPushButton[role="danger"]:hover:!disabled {{
    background-color: palette(midlight);
}}
QPushButton[role="danger"]:disabled {{
    color: {danger["text_disabled"]};
    border-color: {danger["text_disabled"]};
}}
QPushButton[role="danger-confirm"] {{
    background-color: {danger["fill"]};
    border-color: {danger["fill"]};
    color: {danger["ink"]};
    font-weight: 600;
}}
QPushButton[role="danger-confirm"]:hover:!disabled {{
    background-color: {danger["fill_hover"]};
    border-color: {danger["fill_hover"]};
}}
QPushButton[role="danger-confirm"]:pressed:!disabled {{
    background-color: {danger["fill_pressed"]};
    border-color: {danger["fill_pressed"]};
}}
QPushButton[role="danger-confirm"]:disabled {{
    background-color: {danger["fill_disabled"]};
    border-color: {danger["fill_disabled"]};
    color: {danger["ink_disabled"]};
}}

/* A compact button sits in a dense row (a table's own actions). */
QPushButton[size="compact"] {{
    min-height: {_INNER_COMPACT}px;
    padding: 0 10px;
}}

{fields}
/* A group's name is a heading with space under it, not a frame cut into a line:
   six framed boxes turned one window into a grid of boxes, and in the dark theme
   those frames all but disappeared. */
QGroupBox {{
    border: none;
    margin-top: 6px;
    padding-top: 18px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0;
    color: palette(text);
}}

/* Collections keep the palette's own base and selection, with the grid dropped:
   rows read as rows, not as cells of a spreadsheet. */
QTableWidget, QTableView {{
    border: {BORDER_WIDTH}px solid palette(mid);
    border-radius: {RADIUS}px;
    background-color: palette(base);
    gridline-color: transparent;
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}}
QHeaderView::section {{
    background-color: palette(window);
    color: palette(text);
    border: none;
    border-bottom: 1px solid palette(mid);
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item, QTableView::item {{
    padding: 4px 6px;
}}

{indicators}
/* A separator is the app's own hairline, in the palette's mid tone, so it stays
   visible in both themes — the band lines in dialogs are drawn with these. */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: palette(mid);
    background-color: palette(mid);
    border: none;
    max-height: 1px;
}}
"""


def apply_theme(app: QApplication, cache_dir: Path | None = None) -> None:
    """Install the app-wide sheet. Call once, after the palette is settled."""
    directory = cache_dir if cache_dir is not None else resolve_state_dir() / "ui-marks"
    marks = write_control_marks(app.palette(), directory)
    app.setStyleSheet(build_stylesheet(app.palette(), marks))
