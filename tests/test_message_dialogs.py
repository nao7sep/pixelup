from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from pixelup.message_dialogs import StartupFailureDialog


def test_startup_failure_uses_only_safe_authored_copy_without_a_severity_icon(
    qapp: QApplication,
) -> None:
    dialog = StartupFailureDialog(
        "PixelUp could not read its storage or application state.",
        "Choose a writable PixelUp storage folder and try again.",
    )
    try:
        assert dialog.windowTitle() == "PixelUp could not start"
        assert dialog.findChildren(QMessageBox) == []
        assert dialog.message_label.text() == (
            "PixelUp could not read its storage or application state."
        )
        assert dialog.hint_label.text() == (
            "Choose a writable PixelUp storage folder and try again."
        )
        assert dialog.buttons.standardButtons() == QDialogButtonBox.StandardButton.Close
        assert dialog.minimumSize() == dialog.sizeHint()
    finally:
        dialog.deleteLater()
