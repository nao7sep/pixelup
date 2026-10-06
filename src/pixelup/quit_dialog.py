from __future__ import annotations

from typing import Literal

from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from pixelup.dialog_shell import NOTICE_WIDTH, DialogShell
from pixelup.i18n.localized import localize
from pixelup.i18n.message import Message


def quit_confirmation_text(active: int) -> Message:
    if not active:
        return Message("quit.openImages")
    return Message.of("quit.activeJobs", count=active)


class QuitConfirmDialog(DialogShell):
    """Confirm quitting when images are open or work is in progress.

    A custom dialog rather than QMessageBox: QMessageBox lays its buttons out by
    role per platform, and on macOS that pushes a DestructiveRole button to the
    far left — the opposite of the convention's Cancel-left / destructive-right
    order. Building the footer by hand keeps the order deterministic everywhere.
    ``exec()`` returns ``Accepted`` to quit, ``Rejected`` (Cancel, Escape) to stay.
    """

    def __init__(self, active: int, parent: QWidget | None = None) -> None:
        super().__init__("quit.title", parent, width=NOTICE_WIDTH)

        message = localize(QLabel(), text=quit_confirmation_text(active))
        message.setWordWrap(True)
        self.body_layout.addWidget(message)

        cancel_button = localize(QPushButton(), text="quit.cancel")
        cancel_button.setDefault(True)
        cancel_button.clicked.connect(self.reject)

        quit_button = localize(QPushButton(), text="quit.quit")
        quit_button.setProperty("role", "danger-confirm")
        quit_button.clicked.connect(self.accept)

        # Cancel before Quit → Cancel on the left, the destructive action on the
        # right; the footer band's own stretch right-aligns the pair.
        self.add_footer_widget(cancel_button)
        self.add_footer_widget(quit_button)
        # The safe action, so a reflexive Enter or Space never quits.
        self.set_initial_focus(cancel_button)
        self.fit()


QuitSaveChoice = Literal["cancel", "retry", "quit_anyway"]


class QuitSaveFailedDialog(DialogShell):
    """A quit the user started, held because the parameters did not save
    (unsaved-edits-conventions, Quitting).

    ``choose()`` runs the dialog and names the answer; Escape and the title bar's
    close answer Cancel, which keeps PixelUp open with the edits still shown.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("app.name", parent, width=NOTICE_WIDTH)
        self._choice: QuitSaveChoice = "cancel"

        message = localize(QLabel(), text="quit.saveFailed")
        message.setWordWrap(True)
        self.body_layout.addWidget(message)

        cancel_button = localize(QPushButton(), text="quit.cancel")
        cancel_button.clicked.connect(self.reject)

        retry_button = localize(QPushButton(), text="quit.retry")
        retry_button.setProperty("role", "primary")
        retry_button.setDefault(True)
        retry_button.clicked.connect(lambda: self._answer("retry"))

        quit_button = localize(QPushButton(), text="quit.quitAnyway")
        quit_button.setProperty("role", "danger-confirm")
        quit_button.clicked.connect(lambda: self._answer("quit_anyway"))

        self.add_footer_widget(cancel_button)
        self.add_footer_widget(retry_button)
        self.add_footer_widget(quit_button)
        # Retry keeps the edits, so a reflexive Enter can only try the save again.
        self.set_initial_focus(retry_button)
        self.fit()

    def _answer(self, choice: QuitSaveChoice) -> None:
        self._choice = choice
        self.accept()

    def choose(self) -> QuitSaveChoice:
        self._choice = "cancel"
        self.exec()
        return self._choice
