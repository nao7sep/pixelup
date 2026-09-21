"""PixelUp's one owned appearance: the sizes every control is built from, and the
style sheet that draws them.

PixelUp follows the OS light/dark theme and the OS accent, and has no theme
setting of its own. What it does own is what the platform palette gets wrong for
an app like this one:

- the neutral surfaces, one set per theme (``surfaces``): the macOS palette fills
  a dark field, list or popup with a near-black base and gives buttons the
  window's own colour, so neither read as a control;
- the primary action's colour, which is the OS accent (``accent_colours``) — the
  palette's highlight is the text-selection colour, a muted navy in dark and a
  pale blue under black letters in light;
- a destructive red and a warning amber, which have no palette role at all.

It also owns geometry and state: one height for a standard control and one for a
compact one, one corner radius, and a hover, pressed, focused and disabled
treatment for each control, so nothing is left to the toolkit's defaults. Sizes
and colours are named so a change moves the whole app.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPixmap, QPolygon, QPolygonF
from PySide6.QtWidgets import QApplication

from pixelup.config import resolve_state_dir
from pixelup.fonts import DEFAULT_UI_FONT_SIZE

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
# The app's text and a group heading one step above it. The heading's space is its
# own line plus a short gap, so it sits close above the section it names.
TEXT_SIZE = DEFAULT_UI_FONT_SIZE
GROUP_TITLE_SIZE = DEFAULT_UI_FONT_SIZE + 2
GROUP_TITLE_SPACE = 16
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


# The ink that sits on the accent. White in both themes: the macOS palette's
# highlighted text is black in light, because its highlight is a pale blue.
ACCENT_INK = "#ffffff"


def surfaces(palette: QPalette) -> dict[str, str]:
    """The neutral surfaces the app owns, one set per theme.

    A field, a list and a popup take ``field`` behind a visible ``field_edge``; a
    button takes ``button`` and steps one way through ``button_hover`` to
    ``button_pressed``, deepening in light and lifting in dark, the direction each
    theme's hover already moves. ``hairline`` separates, ``chip`` holds a key.
    """
    if is_dark(palette):
        return {
            "field": "#2a2a2a", "field_edge": "#505050",
            "button": "#464646", "button_hover": "#505050", "button_pressed": "#5b5b5b",
            "button_edge": "#5c5c5c", "hairline": "#4a4a4a", "chip": "#3c3c3c",
        }
    return {
        "field": "#ffffff", "field_edge": "#c4c4c4",
        "button": "#ffffff", "button_hover": "#f1f1f1", "button_pressed": "#e3e3e3",
        "button_edge": "#c4c4c4", "hairline": "#d3d3d3", "chip": "#f6f6f6",
    }


def surfaces_disabled(palette: QPalette) -> dict[str, str]:
    """The neutral surfaces receded over the window, for a control that is off."""
    tone = surfaces(palette)
    window = palette.color(QPalette.ColorRole.Window)
    button = _receded(QColor(tone["button"]), window)
    return {
        "button": button,
        "button_ink": _receded(palette.color(QPalette.ColorRole.ButtonText), QColor(button)),
        "field": _receded(QColor(tone["field"]), window),
        "field_edge": _receded(QColor(tone["field_edge"]), window),
    }


def accent_colours(palette: QPalette) -> dict[str, str]:
    """The primary action's colours: the OS accent, deepening through hover and press.

    The accent comes from the palette's accent role, so it follows the OS setting;
    a style sheet has no colour arithmetic, which is the only reason the steps are
    computed in Python. Off, the fill and its ink recede over the window, so a
    switched-off primary still has a shape and still says which action it is.
    """
    accent = palette.color(QPalette.ColorRole.Accent)
    window = palette.color(QPalette.ColorRole.Window)
    fill_disabled = _receded(accent, window)
    return {
        "fill": accent.name(),
        "fill_hover": accent.darker(110).name(),
        "fill_pressed": accent.darker(124).name(),
        "ink": ACCENT_INK,
        "fill_disabled": fill_disabled,
        "ink_disabled": _receded(QColor(ACCENT_INK), QColor(fill_disabled)),
    }


def accent_disabled(palette: QPalette) -> dict[str, str]:
    """The accent receded, for a set tick or radio that is switched off."""
    colours = accent_colours(palette)
    return {"fill": colours["fill_disabled"], "ink": colours["ink_disabled"]}


def warning_text(palette: QPalette) -> str:
    """An amber that reads as letters on the theme's window, for a standing warning."""
    return "#f2c94c" if is_dark(palette) else "#8a5a00"


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
        ink = QColor(ACCENT_INK)
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


def _field_rules(palette: QPalette, marks: dict[str, Path]) -> str:
    """Fields, including the arrow marks, which only exist when they were written."""
    tone = surfaces(palette)
    off = surfaces_disabled(palette)
    accent = accent_colours(palette)
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
    border: {BORDER_WIDTH}px solid {tone["field_edge"]};
    border-radius: {RADIUS}px;
    background-color: {tone["field"]};
    color: palette(text);
    selection-background-color: {accent["fill"]};
    selection-color: {accent["ink"]};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent["fill"]};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: palette(placeholder-text);
    background-color: {off["field"]};
    border-color: {off["field_edge"]};
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
/* The list a combo opens is the field's own surface, not the palette's base,
   which in the dark theme is all but black. */
QComboBox QAbstractItemView {{
    background-color: {tone["field"]};
    border: {BORDER_WIDTH}px solid {tone["field_edge"]};
    selection-background-color: {accent["fill"]};
    selection-color: {accent["ink"]};
    outline: 0;
}}

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
    tone = surfaces(palette)
    tone_off = surfaces_disabled(palette)
    accent = accent_colours(palette)
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
    border: {BORDER_WIDTH}px solid {tone["field_edge"]};
    background-color: {tone["field"]};
}}
QCheckBox::indicator {{ border-radius: {INNER_RADIUS}px; }}
/* Half the box plus its border, so the ring is a circle rather than the rounded
   square a smaller radius drew. */
QRadioButton::indicator {{ border-radius: {radius}px; }}
QCheckBox::indicator:hover:!disabled, QRadioButton::indicator:hover:!disabled {{
    border-color: {accent["fill"]};
}}
QCheckBox:focus::indicator, QRadioButton:focus::indicator {{
    border-color: {accent["fill"]};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {accent["fill"]};
    border-color: {accent["fill"]};
}}
QCheckBox::indicator:checked {{ image: url({marks["check"].as_posix()}); }}
QRadioButton::indicator:checked {{ image: url({marks["radio"].as_posix()}); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {tone_off["field"]};
    border-color: {tone_off["field_edge"]};
}}
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {{
    background-color: {off["fill"]};
    border-color: {off["fill"]};
}}
QCheckBox::indicator:checked:disabled {{ image: url({marks["check-disabled"].as_posix()}); }}
QRadioButton::indicator:checked:disabled {{ image: url({marks["radio-disabled"].as_posix()}); }}
"""


def build_stylesheet(palette: QPalette, marks: dict[str, Path] | None = None) -> str:
    """The app-wide sheet, from the palette's text and window plus the owned colours.

    Without ``marks`` the fields, ticks and radios are left alone entirely, so the
    toolkit keeps drawing them and their marks; everything else is styled either
    way.
    """
    tone = surfaces(palette)
    tone_off = surfaces_disabled(palette)
    accent = accent_colours(palette)
    danger = danger_colours(palette)
    fields = _field_rules(palette, marks) if marks else ""
    indicators = _indicator_rules(palette, marks) if marks else ""
    return f"""
/* Buttons. A standard button is the app's utility role: its own surface, a
   visible edge in both themes, and its own hover, pressed, focus and disabled
   states rather than the toolkit's. */
QPushButton {{
    min-height: {_INNER}px;
    padding: 0 14px;
    border: {BORDER_WIDTH}px solid {tone["button_edge"]};
    border-radius: {RADIUS}px;
    background-color: {tone["button"]};
    color: palette(button-text);
}}
QPushButton:hover:!disabled {{
    background-color: {tone["button_hover"]};
}}
QPushButton:pressed:!disabled {{
    background-color: {tone["button_pressed"]};
}}
QPushButton:focus {{
    border-color: {accent["fill"]};
}}
/* Off, a button's fill and label recede over the window and its edge stays, so
   it keeps its anatomy and can still be seen by. */
QPushButton:disabled {{
    background-color: {tone_off["button"]};
    border-color: {tone["button_edge"]};
    color: {tone_off["button_ink"]};
}}

/* The main action of a window full of peers: a standard button whose label
   carries the weight, so it leads without a fill that would single it out. */
QPushButton[role="main"] {{
    font-weight: 600;
}}

/* The primary role: a dialog's commit, in the OS accent. */
QPushButton[role="primary"] {{
    background-color: {accent["fill"]};
    border-color: {accent["fill"]};
    color: {accent["ink"]};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover:!disabled {{
    background-color: {accent["fill_hover"]};
    border-color: {accent["fill_hover"]};
}}
QPushButton[role="primary"]:pressed:!disabled {{
    background-color: {accent["fill_pressed"]};
    border-color: {accent["fill_pressed"]};
}}
QPushButton[role="primary"]:focus {{
    border-color: palette(text);
}}
/* Off, the primary stays a filled button in its own accent, faded over the
   window with its ink, so it still says which action it is. */
QPushButton[role="primary"]:disabled {{
    background-color: {accent["fill_disabled"]};
    border-color: {accent["fill_disabled"]};
    color: {accent["ink_disabled"]};
}}

/* Destructive actions, in the fleet's two roles: a trigger that opens a
   destructive path is outlined, and the confirming button of the dialog that
   asks is filled, so a solid red always means this is the last step. */
QPushButton[role="danger"] {{
    color: {danger["text"]};
    border-color: {danger["text"]};
    background-color: {tone["button"]};
}}
QPushButton[role="danger"]:hover:!disabled {{
    background-color: {tone["button_hover"]};
}}
QPushButton[role="danger"]:pressed:!disabled {{
    background-color: {tone["button_pressed"]};
}}
QPushButton[role="danger"]:disabled {{
    background-color: {tone_off["button"]};
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
/* A group's name is a heading that leads its section: a step larger than the
   text, and close above what it names, with the gap between sections doing the
   separating. Six framed boxes had turned one window into a grid of boxes. */
QGroupBox {{
    border: none;
    margin-top: 0;
    padding-top: {GROUP_TITLE_SPACE}px;
    font-size: {GROUP_TITLE_SIZE}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0;
    color: palette(text);
}}
/* A title cannot take a size of its own, so the group carries the heading's and
   everything inside it returns to the app's text. The rule names no control, so a
   field left to the toolkit (no marks) stays entirely the toolkit's; a role that
   sets its own weight is more specific and keeps it. */
QGroupBox * {{
    font-size: {TEXT_SIZE}px;
    font-weight: normal;
}}

/* Collections take the field surface and the palette's selection, with no grid:
   rows read as rows, not as cells of a spreadsheet. */
QTableWidget, QTableView {{
    border: {BORDER_WIDTH}px solid {tone["field_edge"]};
    border-radius: {RADIUS}px;
    background-color: {tone["field"]};
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
}}
QHeaderView::section {{
    background-color: {tone["field"]};
    color: palette(text);
    border: none;
    border-bottom: 1px solid {tone["hairline"]};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item, QTableView::item {{
    padding: 4px 8px;
}}

/* A standing warning is amber letters, not a filled control: a fill read as a
   button beside the button it was about. */
QLabel[severity="warning"] {{
    color: {warning_text(palette)};
}}

/* A key in a reference list: the one mark on the surface is the key itself. */
QLabel#shortcutKey {{
    border: {BORDER_WIDTH}px solid {tone["field_edge"]};
    border-radius: {INNER_RADIUS}px;
    padding: 4px 8px;
    background-color: {tone["chip"]};
    font-weight: 600;
}}

{indicators}
/* A separator is the app's own hairline, so it stays visible in both themes —
   the band lines in dialogs are drawn with these. */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {tone["hairline"]};
    background-color: {tone["hairline"]};
    border: none;
    max-height: 1px;
}}
"""


def apply_theme(app: QApplication, cache_dir: Path | None = None) -> None:
    """Install the app-wide sheet. Call once, after the palette is settled."""
    directory = cache_dir if cache_dir is not None else resolve_state_dir() / "ui-marks"
    marks = write_control_marks(app.palette(), directory)
    app.setStyleSheet(build_stylesheet(app.palette(), marks))
