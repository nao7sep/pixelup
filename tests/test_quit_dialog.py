from __future__ import annotations

from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton

from pixelup.quit_dialog import QuitConfirmDialog, quit_confirmation_text


def test_quit_confirmation_text_pluralizes() -> None:
    assert quit_confirmation_text(0) == "Open images will be closed. Quit PixelUp?"
    assert quit_confirmation_text(1) == "1 active job will be abandoned. Quit PixelUp?"
    assert quit_confirmation_text(2) == "2 active jobs will be abandoned. Quit PixelUp?"


def _footer_button_labels(dialog: QuitConfirmDialog) -> list[str]:
    for layout in dialog.findChildren(QHBoxLayout):
        buttons = [
            layout.itemAt(i).widget()
            for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), QPushButton)
        ]
        if buttons:
            return [button.text() for button in buttons]
    return []


def test_quit_dialog_button_order_and_styling(qapp: QApplication) -> None:
    dialog = QuitConfirmDialog(3)
    try:
        # Cancel-left, destructive-right, deterministically (the QMessageBox this
        # replaced reversed this on macOS via DestructiveRole).
        assert _footer_button_labels(dialog) == ["Cancel", "Quit"]
        cancel = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Cancel")
        quit_button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Quit")
        assert cancel.isDefault()  # Enter/closes-safe defaults to Cancel
        assert quit_button.styleSheet()  # danger styling applied to the destructive action
    finally:
        dialog.deleteLater()
