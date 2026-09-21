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
