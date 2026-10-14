from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, QTextBoundaryFinder, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFontMetrics, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QLayout

from .fonts import DEFAULT_UI_FONT_SIZE
from .i18n.message import Message

# A message about several items names a few, each shortened as identifying text,
# and counts the rest (interface-styling-conventions, Labels).
NAMED_IN_MESSAGE = 3
_NAME_WIDTH_CHARACTERS = 40
_BREAK = "\u200b"

REGULAR_SPACING = 10
# The roomy rhythm DialogShell gives a dialog's body band.
DIALOG_MARGIN = 20
DIALOG_SPACING = 16


def use_regular_spacing(layout: QLayout, *, margins: bool = True) -> None:
    margin = REGULAR_SPACING if margins else 0
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(REGULAR_SPACING)


def title_label(text: str) -> QLabel:
    """A large, bold label for the product's own name on the about surface.

    The one heading a native dialog keeps: the OS title bar names every other
    surface, so nothing else here has a heading of its own
    (modal-dialog-conventions).
    """
    label = QLabel(text)
    font = label.font()
    font.setPixelSize(round(DEFAULT_UI_FONT_SIZE * 1.6))
    font.setBold(True)
    label.setFont(font)
    return label


def secondary_label(text: str) -> QLabel:
    """A muted label for supporting text (version, captions, copyright).

    Uses the placeholder-text role, whose value apply_palette_fixes corrects.
    """
    label = QLabel(text)
    label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    return label


# Qt ships PlaceholderText at 63/255 — right for ghost text in an empty field, too
# faint for captions meant to be read. Body text is 216/255; this sits below it.
_SECONDARY_TEXT_ALPHA = 150

_GROUPS = (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled)


def apply_palette_fixes(app: QApplication) -> None:
    """Fix two palette roles Qt gets wrong. Call once, after setStyle.

    ButtonText: Qt hands the Inactive group opaque black in both themes, so in dark
    mode every combo and button turns black-on-dark once the window loses focus.
    Copying Active over it fixes dark and is near-identical in light. (Highlight also
    differs between the groups, but that one is correct — macOS greys out a selection
    in an unfocused window — so it is left alone.)

    PlaceholderText: see _SECONDARY_TEXT_ALPHA. No real placeholders exist here, so
    this role serves only secondary_label.
    """
    pal = QPalette(app.palette())
    pal.setColor(
        QPalette.ColorGroup.Inactive,
        QPalette.ColorRole.ButtonText,
        pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText),
    )
    for group in _GROUPS:
        secondary = QColor(pal.color(group, QPalette.ColorRole.WindowText))
        secondary.setAlpha(_SECONDARY_TEXT_ALPHA)
        pal.setColor(group, QPalette.ColorRole.PlaceholderText, secondary)
    app.setPalette(pal)


def open_url(url: str) -> None:
    if not QDesktopServices.openUrl(QUrl(url)):
        raise RuntimeError(f"Could not open URL: {url}")


def identifying_name(name: str, metrics: QFontMetrics) -> str:
    """``name`` shortened to a short run inside a sentence, keeping its beginning."""
    width = metrics.averageCharWidth() * _NAME_WIDTH_CHARACTERS
    return metrics.elidedText(name, Qt.TextElideMode.ElideRight, width)


def named_values(names: Sequence[str], metrics: QFontMetrics) -> tuple[str | Message, ...]:
    """The first few ``names``, shortened, then a count of the rest, as a list value."""
    shown = tuple(identifying_name(name, metrics) for name in names[:NAMED_IN_MESSAGE])
    rest = len(names) - len(shown)
    return (*shown, Message.of("common.more", count=rest)) if rest else shown


def wrap_anywhere(text: str) -> str:
    """``text`` with a zero-width break between its graphemes.

    A wrapping QLabel breaks only between words, so a long path would otherwise
    overflow or widen its container; this lets it wrap whole, like RowDetail.
    """
    # QTextBoundaryFinder counts UTF-16 units, which differ from Python's indices
    # wherever a character lies outside the Basic Multilingual Plane.
    units = text.encode("utf-16-le")
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    pieces: list[str] = []
    start = 0
    while (end := finder.toNextBoundary()) not in (-1, start):
        pieces.append(units[start * 2 : end * 2].decode("utf-16-le"))
        start = end
    return _BREAK.join(pieces)
