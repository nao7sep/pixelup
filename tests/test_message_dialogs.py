from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

import pixelup.message_dialogs as message_dialogs
from pixelup.message_dialogs import MessageDialog, StartupFailureDialog


def test_config_reset_uses_an_icon_free_authored_dialog(
    qapp: QApplication,
    monkeypatch,
) -> None:
    shown: list[MessageDialog] = []
    monkeypatch.setattr(
        MessageDialog,
        "exec",
        lambda dialog: shown.append(dialog),
    )

    message_dialogs.warn_config_reset(None)

    assert len(shown) == 1
    dialog = shown[0]
    try:
        text = dialog.message_label.text()
        assert dialog.windowTitle() == "PixelUp"
        assert dialog.findChildren(QMessageBox) == []
        assert text == (
            "Your settings file was unreadable and has been reset to defaults.\n\n"
            "A preserved copy remains available, and its location is recorded in the log."
        )
        assert ".invalid" not in text
    finally:
        dialog.deleteLater()


def test_startup_failure_fits_short_copy_without_a_severity_icon(
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
        assert dialog.body_scroll.height() == dialog.body.sizeHint().height()
        assert dialog.body_scroll.verticalScrollBarPolicy() == (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
    finally:
        dialog.deleteLater()


def test_startup_failure_caps_only_the_body_for_long_copy(
    qapp: QApplication,
) -> None:
    dialog = StartupFailureDialog("A long explanation. " * 400, "Try again.")
    try:
        assert dialog.body.sizeHint().height() > dialog.body_scroll.height()
        assert dialog.body_scroll.height() <= 420
        assert dialog.layout().indexOf(dialog.buttons) > dialog.layout().indexOf(
            dialog.body_scroll
        )
        dialog.show()
        qapp.processEvents()
        assert dialog.buttons.isVisibleTo(dialog)
    finally:
        dialog.close()
        dialog.deleteLater()
