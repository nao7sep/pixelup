from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel

from pixelup import __version__
from pixelup.about_dialog import AboutDialog


def test_about_dialog_includes_required_metadata(qapp: QApplication) -> None:
    dialog = AboutDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert dialog.windowTitle() == "About PixelUp"
        assert dialog.windowTitle() in text
        assert "PixelUp" in text
        assert __version__ in text
        assert "MIT License" in text
    finally:
        dialog.deleteLater()
