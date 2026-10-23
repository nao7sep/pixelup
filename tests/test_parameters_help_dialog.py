from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel, QScrollArea

from pixelup.parameters_help_dialog import ParametersHelpDialog


def test_parameters_help_dialog_covers_every_parameter(qapp: QApplication) -> None:
    dialog = ParametersHelpDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert dialog.windowTitle() == "Parameters help"
        # Named by its title bar, not again inside the body.
        assert dialog.windowTitle() not in text
        # One entry per Parameters-panel control.
        for name in (
            "Scale",
            "Denoise",
            "Alpha mode",
            "Output format",
            "Quality",
            "Tile size",
            "Device",
            "Strip metadata",
            "Target profile",
        ):
            assert name in text
    finally:
        dialog.deleteLater()


def test_parameters_help_dialog_has_labelled_close(qapp: QApplication) -> None:
    # Modal-dialog conventions: an informational surface carries an explicit
    # labelled Close button; Escape and ✕ are supplementary.
    dialog = ParametersHelpDialog()
    try:
        box = dialog.findChild(QDialogButtonBox)
        assert box is not None
        assert box.button(QDialogButtonBox.StandardButton.Close) is not None
    finally:
        dialog.deleteLater()


def test_the_manual_opens_at_a_reading_height_and_scrolls(qapp: QApplication) -> None:
    # Nine entries of prose is a manual, not a notice: opening at the height of
    # the whole thing makes a reference surface as tall as the display allows. It
    # takes a reading height and scrolls — with the content area itself scrolling,
    # so there is no bordered panel inside the body to read as a second surface.
    dialog = ParametersHelpDialog()
    try:
        dialog.show()
        qapp.processEvents()

        assert dialog.body.height() > dialog.body_scroll.viewport().height()
        assert dialog.body_scroll.verticalScrollBar().isVisible()
        assert dialog.height() < dialog.body.height()
        # One scroll region, and it is the body band itself.
        assert dialog.findChildren(QScrollArea) == [dialog.body_scroll]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_the_manual_reads_on_the_content_surface(qapp: QApplication) -> None:
    """Its body is the same surface as the lists and the models panel."""
    dialog = ParametersHelpDialog()
    try:
        assert dialog.body.property("surface") == "content"
    finally:
        dialog.deleteLater()


def test_manual_keyboard_scrolls_while_the_footer_stays_fixed(qapp: QApplication) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    dialog = ParametersHelpDialog()
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        close = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close)
        assert dialog.focusWidget() is close
        QTest.keyClick(close, Qt.Key.Key_Tab)
        assert dialog.focusWidget() is dialog.body_scroll
        bar = dialog.body_scroll.verticalScrollBar()
        assert bar.maximum() > 0
        footer_position = close.mapTo(dialog, close.rect().topLeft())
        for key, modifiers, start, expected in (
            (Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier, 0, bar.singleStep()),
            (Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier, bar.singleStep(), 0),
            (Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier, 0, bar.pageStep()),
            (Qt.Key.Key_PageUp, Qt.KeyboardModifier.NoModifier, bar.pageStep(), 0),
            (Qt.Key.Key_End, Qt.KeyboardModifier.NoModifier, 0, bar.maximum()),
            (Qt.Key.Key_Home, Qt.KeyboardModifier.NoModifier, bar.maximum(), 0),
            (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, 0, bar.pageStep()),
            (Qt.Key.Key_Space, Qt.KeyboardModifier.ShiftModifier, bar.pageStep(), 0),
        ):
            bar.setValue(start)
            QTest.keyClick(dialog.body_scroll, key, modifiers)
            assert bar.value() == min(bar.maximum(), expected)
            assert close.mapTo(dialog, close.rect().topLeft()) == footer_position
        QTest.keyClick(dialog.body_scroll, Qt.Key.Key_Tab)
        assert dialog.focusWidget() is close
        QTest.keyClick(close, Qt.Key.Key_Space)
        assert not dialog.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
