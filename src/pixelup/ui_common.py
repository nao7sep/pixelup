from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QLayout

REGULAR_SPACING = 10

# A thin, rounded, themed scroll bar applied app-wide. Fusion ships a thick,
# square scroll bar, so per the window-chrome conventions we override it with a
# slim pill. Colors come from palette() roles (not hard-coded hex) so the bar
# follows whatever OS light/dark theme the unowned palette resolves to.
_SCROLLBAR_QSS = """
QScrollBar:vertical {
    width: 12px;
    background: transparent;
    margin: 0;
}
QScrollBar:horizontal {
    height: 12px;
    background: transparent;
    margin: 0;
}
QScrollBar::handle {
    background: palette(mid);
    border-radius: 6px;
    min-height: 24px;
    min-width: 24px;
    margin: 2px;
}
QScrollBar::handle:hover {
    background: palette(dark);
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
    width: 0;
    background: transparent;
}
"""


def use_regular_spacing(layout: QLayout, *, margins: bool = True) -> None:
    margin = REGULAR_SPACING if margins else 0
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(REGULAR_SPACING)


def title_label(text: str) -> QLabel:
    """A large, bold label for a dialog's primary heading (e.g. the app name)."""
    label = QLabel(text)
    font = label.font()
    font.setPointSizeF(font.pointSizeF() * 1.6)
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


def apply_scrollbar_style(app: QApplication) -> None:
    """Install the thin, rounded, palette-themed scroll bar app-wide.

    Merges with any existing application stylesheet rather than clobbering it, so
    later QSS (none today) is preserved.
    """
    existing = app.styleSheet()
    app.setStyleSheet(f"{existing}\n{_SCROLLBAR_QSS}" if existing else _SCROLLBAR_QSS)


def open_url(url: str) -> None:
    if not QDesktopServices.openUrl(QUrl(url)):
        raise RuntimeError(f"Could not open URL: {url}")
