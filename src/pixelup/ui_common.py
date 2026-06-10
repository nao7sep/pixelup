from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLayout

REGULAR_SPACING = 10


def use_regular_spacing(layout: QLayout, *, margins: bool = True) -> None:
    margin = REGULAR_SPACING if margins else 0
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(REGULAR_SPACING)


def open_url(url: str) -> None:
    if not QDesktopServices.openUrl(QUrl(url)):
        raise RuntimeError(f"Could not open URL: {url}")
