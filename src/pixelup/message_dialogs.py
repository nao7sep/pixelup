from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import use_dialog_spacing

# Named, greppable home for PixelUp's informational alerts. A QMessageBox is a
# framework primitive, not a native picker, so per the modal-dialog conventions it
# is wrapped in named surfaces rather than called inline from feature code. Each
# alert is its own named function so it can be located by name.
_APP = "PixelUp"


def _info(parent: QWidget, text: str) -> None:
    QMessageBox.information(parent, _APP, text)


def warn_config_reset(parent: QWidget, quarantined_name: str) -> None:
    _info(
        parent,
        "Your settings file was unreadable and has been reset to defaults.\n\n"
        f"The unreadable file was kept as {quarantined_name} in the PixelUp folder.",
    )


def warn_jobs_stopping(parent: QWidget) -> None:
    _info(
        parent,
        "PixelUp is still stopping active work. "
        "It will close as soon as everything has stopped safely.",
    )


class StartupFailureDialog(QDialog):
    """A deliberately plain fatal-startup surface without a severity icon."""

    def __init__(self, user_message: str, user_hint: str | None) -> None:
        super().__init__(None, Qt.WindowType.Dialog)
        self.setWindowTitle("PixelUp could not start")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)
        self.message_label = QLabel(user_message)
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumWidth(340)
        self.message_label.setMaximumWidth(480)
        layout.addWidget(self.message_label)

        self.hint_label = QLabel(user_hint or "")
        self.hint_label.setWordWrap(True)
        self.hint_label.setMaximumWidth(480)
        self.hint_label.setVisible(bool(user_hint))
        layout.addWidget(self.hint_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.adjustSize()
        self.setMinimumSize(self.sizeHint())


def show_startup_failure(user_message: str, user_hint: str | None) -> None:
    StartupFailureDialog(user_message, user_hint).exec()
