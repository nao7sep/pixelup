from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from pixelup.dialog_shell import NOTICE_WIDTH, DialogShell


def quit_confirmation_text(active: int) -> str:
    if not active:
        return "Open images will be closed. Quit PixelUp?"
    noun = "active job" if active == 1 else "active jobs"
    return f"{active} {noun} will be abandoned. Quit PixelUp?"


class QuitConfirmDialog(DialogShell):
    """Confirm quitting when images are open or work is in progress.

    A custom dialog rather than QMessageBox: QMessageBox lays its buttons out by
    role per platform, and on macOS that pushes a DestructiveRole button to the
    far left — the opposite of the convention's Cancel-left / destructive-right
    order. Building the footer by hand keeps the order deterministic everywhere.
    ``exec()`` returns ``Accepted`` to quit, ``Rejected`` (Cancel, Escape) to stay.
    """

    def __init__(self, active: int, parent: QWidget | None = None) -> None:
        super().__init__("Quit PixelUp?", parent, width=NOTICE_WIDTH)

        message = QLabel(quit_confirmation_text(active))
        message.setWordWrap(True)
        self.body_layout.addWidget(message)

        cancel_button = QPushButton("Cancel")
        cancel_button.setDefault(True)
        cancel_button.clicked.connect(self.reject)

        quit_button = QPushButton("Quit")
        quit_button.setProperty("role", "danger-confirm")
        quit_button.clicked.connect(self.accept)

        # Cancel before Quit → Cancel on the left, the destructive action on the
        # right; the footer band's own stretch right-aligns the pair.
        self.add_footer_widget(cancel_button)
        self.add_footer_widget(quit_button)
        # The safe action, so a reflexive Enter or Space never quits.
        self.set_initial_focus(cancel_button)
        self.fit()
