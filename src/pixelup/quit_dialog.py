from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import title_label, use_dialog_spacing, use_regular_spacing

# Destructive actions get danger styling per the modal-dialog conventions; red is
# not a standard palette role, so it is set explicitly here.
_DANGER_QSS = (
    "QPushButton { background-color: #c0392b; color: white; padding: 4px 14px; }"
    "QPushButton:hover { background-color: #e74c3c; }"
)


def quit_confirmation_text(active: int) -> str:
    if not active:
        return "Open images will be closed. Quit PixelUp?"
    noun = "active job" if active == 1 else "active jobs"
    return f"{active} {noun} will be abandoned. Quit PixelUp?"


class QuitConfirmDialog(QDialog):
    """Confirm quitting when images are open or work is in progress.

    A custom dialog rather than QMessageBox: QMessageBox lays its buttons out by
    role per platform, and on macOS that pushes a DestructiveRole button to the
    far left — the opposite of the convention's Cancel-left / destructive-right
    order. Building the footer by hand keeps the order deterministic everywhere.
    ``exec()`` returns ``Accepted`` to quit, ``Rejected`` (Cancel, Escape) to stay.
    """

    def __init__(self, active: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quit PixelUp?")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)
        layout.addWidget(title_label("Quit PixelUp?"))

        message = QLabel(quit_confirmation_text(active))
        message.setWordWrap(True)
        layout.addWidget(message)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        use_regular_spacing(footer_layout, margins=False)
        footer_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.setDefault(True)
        cancel_button.clicked.connect(self.reject)

        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet(_DANGER_QSS)
        quit_button.clicked.connect(self.accept)

        # Cancel before Quit → Cancel on the left, the destructive action on the
        # right; the stretch right-aligns the pair.
        footer_layout.addWidget(cancel_button)
        footer_layout.addWidget(quit_button)
        layout.addWidget(footer)
