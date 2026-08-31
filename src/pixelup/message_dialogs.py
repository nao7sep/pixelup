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


def warn_config_reset(parent: QWidget, quarantined_name: str) -> None:
    _info(
        parent,
        "Your settings file was unreadable and has been reset to defaults.\n\n"
        f"The unreadable file was kept as {quarantined_name} in the PixelUp folder.",
    )


def warn_config_save_failed(parent: QWidget) -> None:
    _info(
        parent,
        "PixelUp could not save your settings. Your changes are still shown so you can try again.",
    )


def warn_jobs_stopping(parent: QWidget) -> None:
    _info(
        parent,
        "PixelUp is still stopping active work. "
        "It will close as soon as everything has stopped safely.",
    )


def show_startup_failure(detail: str, hint: str | None) -> None:
    remedy = f"\n\n{hint}" if hint else ""
    QMessageBox.critical(
        None,
        "PixelUp could not start",
        "PixelUp could not open its storage or application state.\n\n"
        f"{detail}{remedy}",
    )
