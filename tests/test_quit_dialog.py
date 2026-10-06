from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton

from pixelup.i18n import localizer
from pixelup.i18n.languages import TAGS
from pixelup.i18n.localizer import english
from pixelup.quit_dialog import QuitConfirmDialog, QuitSaveFailedDialog, quit_confirmation_text


def test_quit_confirmation_text_pluralizes() -> None:
    text = english().of
    assert text(quit_confirmation_text(0)) == "Open images will be closed. Quit PixelUp?"
    assert text(quit_confirmation_text(1)) == "1 active job will be abandoned. Quit PixelUp?"
    assert text(quit_confirmation_text(2)) == "2 active jobs will be abandoned. Quit PixelUp?"


def _footer_button_labels(dialog: QDialog) -> list[str]:
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
        # The destructive action declares its role; the app-wide sheet (theme.py)
        # draws it. It used to carry its own style sheet, which is why this asked
        # for a non-empty styleSheet() before.
        assert quit_button.property("role") == "danger-confirm"
        # The OS draws this window's title bar, and that bar is the header: a
        # heading inside the body would say the same thing twice.
        assert dialog.windowTitle() not in {
            label.text() for label in dialog.findChildren(QLabel)
        }
    finally:
        dialog.deleteLater()


def _button(dialog: QDialog, label: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == label)


def test_save_failed_dialog_offers_cancel_retry_and_quit_anyway(qapp: QApplication) -> None:
    dialog = QuitSaveFailedDialog()
    try:
        # Cancel first and the act that loses the edits last, labelled with the act
        # (modal-dialog-conventions).
        assert _footer_button_labels(dialog) == ["Cancel", "Retry", "Quit anyway"]
        retry = _button(dialog, "Retry")
        assert retry.isDefault()
        assert retry.property("role") == "primary"
        assert _button(dialog, "Quit anyway").property("role") == "danger-confirm"
        assert "has not quit" in "".join(label.text() for label in dialog.findChildren(QLabel))
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("label", "choice"),
    [("Cancel", "cancel"), ("Retry", "retry"), ("Quit anyway", "quit_anyway"), (None, "cancel")],
)
def test_save_failed_dialog_names_the_answer(
    qapp: QApplication, label: str | None, choice: str
) -> None:
    dialog = QuitSaveFailedDialog()
    try:
        # None is Escape or the title bar's close: the dialog's reject.
        press = dialog.reject if label is None else _button(dialog, label).click
        QTimer.singleShot(0, press)
        assert dialog.choose() == choice
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("tag", TAGS)
def test_save_failed_dialog_buttons_keep_their_whole_labels(tag: str, qapp: QApplication) -> None:
    # The dialog's width is fixed, so the three labels must fit it in every language
    # rather than be squeezed below what their text needs.
    with localizer.speaking(tag):
        dialog = QuitSaveFailedDialog()
        try:
            dialog.show()
            qapp.processEvents()
            for button in dialog.findChildren(QPushButton):
                assert button.width() >= button.sizeHint().width(), f"{tag}: {button.text()}"
        finally:
            dialog.close()
            dialog.deleteLater()
