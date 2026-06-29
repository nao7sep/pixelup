from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

# Named, greppable home for PixelUp's informational alerts. A QMessageBox is a
# framework primitive, not a native picker, so per the modal-dialog conventions it
# is wrapped in named surfaces rather than called inline from feature code. Each
# alert is its own named function so it can be located by name.
_APP = "PixelUp"


def _info(parent: QWidget, text: str) -> None:
    QMessageBox.information(parent, _APP, text)


def warn_no_images(parent: QWidget) -> None:
    _info(parent, "Open or select at least one image.")


def warn_no_models(parent: QWidget) -> None:
    _info(parent, "Select at least one model.")


def warn_image_in_use(parent: QWidget) -> None:
    _info(parent, "Pending or running jobs still use this image.")
