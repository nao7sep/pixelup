from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from pixelup.parameters_help_dialog import ParametersHelpDialog


def test_parameters_help_dialog_covers_every_parameter(qapp: QApplication) -> None:
    dialog = ParametersHelpDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        # One entry per Parameters-panel control.
        for name in (
            "Scale",
            "Face enhancement",
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
