from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QWidget,
)

from pixelup.dialog_shell import NOTICE_WIDTH, DialogShell

_APP = "PixelUp"


class MessageDialog(DialogShell):
    """An icon-free message whose body grows naturally, then scrolls.

    The window title and footer stay fixed. Only the prose body is bounded, so a
    short one-shot notice does not inherit a large arbitrary dialog height and a
    long startup explanation cannot push its Close button off screen.
    """

    def __init__(
        self,
        title: str,
        user_message: str,
        user_hint: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title,
            parent,
            width=NOTICE_WIDTH,
            passive_body_name="Message details",
        )

        self.message_label = QLabel(user_message)
        self.message_label.setWordWrap(True)
        self.body_layout.addWidget(self.message_label)

        self.hint_label = QLabel(user_hint or "")
        self.hint_label.setWordWrap(True)
        self.hint_label.setVisible(bool(user_hint))
        self.body_layout.addWidget(self.hint_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        self.add_footer_widget(self.buttons)
        self.fit()


def _show_message(parent: QWidget | None, text: str) -> None:
    MessageDialog(_APP, text, parent=parent).exec()


def warn_config_reset(parent: QWidget | None) -> None:
    _show_message(
        parent,
        "Your settings file was unreadable and has been reset to defaults.\n\n"
        "A preserved copy remains available, and its location is recorded in the log.",
    )


def warn_jobs_stopping(parent: QWidget | None) -> None:
    _show_message(
        parent,
        "PixelUp is still stopping active work. "
        "It will close as soon as everything has stopped safely.",
    )


class StartupFailureDialog(MessageDialog):
    """A deliberately plain fatal-startup surface without a severity icon.

    It is built without a parent, which is also what gives it a taskbar button of
    its own: it is not a dialog over a window here, it is the whole application,
    and a window the shell does not list is one the user cannot bring back.
    """

    def __init__(self, user_message: str, user_hint: str | None) -> None:
        super().__init__(
            "PixelUp could not start",
            user_message,
            user_hint,
        )


def show_startup_failure(user_message: str, user_hint: str | None) -> None:
    StartupFailureDialog(user_message, user_hint).exec()
