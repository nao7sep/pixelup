from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

import pixelup.message_dialogs as message_dialogs
from pixelup.message_dialogs import MessageDialog, StartupFailureDialog


def test_side_store_notice_uses_an_icon_free_authored_dialog(
    qapp: QApplication,
    monkeypatch,
) -> None:
    shown: list[MessageDialog] = []
    monkeypatch.setattr(
        MessageDialog,
        "exec",
        lambda dialog: shown.append(dialog),
    )

    store = Path.home() / ".pixelup" / "records.sqlite3"
    message_dialogs.warn_side_store_off(None, "notice.recordsOff", store)

    assert len(shown) == 1
    dialog = shown[0]
    try:
        text = dialog.message_label.text()
        assert dialog.windowTitle() == "PixelUp"
        assert dialog.findChildren(QMessageBox) == []
        assert text == (
            "PixelUp can't read its records file, so Records is off until PixelUp restarts."
            f" The file is left as it is:\n{store}\n\n"
            "Repair or restore the file, or move it away to start fresh records."
        )
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
        dialog.show()
        qapp.processEvents()

        # Authored copy this short should simply be there to read: the body opens
        # at its own height and the bar that would otherwise imply hidden text
        # stays away (modal-dialog-conventions).
        assert not dialog.body_scroll.verticalScrollBar().isVisible()
        assert dialog.body.height() <= dialog.body_scroll.viewport().height()
        assert dialog.body_scroll.verticalScrollBarPolicy() == (
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_startup_failure_caps_only_the_body_for_long_copy(
    qapp: QApplication,
) -> None:
    dialog = StartupFailureDialog("A long explanation. " * 400, "Try again.")
    try:
        dialog.show()
        qapp.processEvents()
        screen = dialog.screen() or QApplication.primaryScreen()

        # The body is what gives: it runs past its viewport and scrolls, while the
        # dialog stays inside its share of the screen and the footer stays put.
        assert dialog.body.height() > dialog.body_scroll.viewport().height()
        assert dialog.body_scroll.verticalScrollBar().isVisible()
        assert dialog.height() <= screen.availableGeometry().height() * 0.85
        assert dialog.buttons.isVisibleTo(dialog)

        outer = dialog.layout()
        assert outer.indexOf(dialog.buttons.parentWidget()) > outer.indexOf(dialog.body_scroll)

        # And the remedy for a body this long is the scroll bar, not an edge to
        # drag: the surface is fixed, so nothing is left hanging on the user
        # finding a handle.
        assert dialog.minimumSize() == dialog.maximumSize() == dialog.size()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_recovery_path_is_always_plain_text(qapp: QApplication) -> None:
    from pixelup.i18n.message import Message

    path = "<b>settings & capture</b>.sqlite3"
    dialog = MessageDialog("app.name", Message.of("notice.recordsOff", path=path))
    try:
        assert dialog.message_label.textFormat() == Qt.TextFormat.PlainText
        assert path in dialog.message_label.text()
        assert dialog.hint_label.textFormat() == Qt.TextFormat.PlainText
    finally:
        dialog.deleteLater()
