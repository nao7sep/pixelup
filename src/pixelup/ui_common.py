from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QLayout

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
