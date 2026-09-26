from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QWidget,
)

from pixelup.dialog_shell import NOTICE_WIDTH, DialogShell
from pixelup.i18n.localized import localize
from pixelup.i18n.message import Message


class MessageDialog(DialogShell):
    """An icon-free message whose body grows naturally, then scrolls.

    The window title and footer stay fixed. Only the prose body is bounded, so a
    short one-shot notice does not inherit a large arbitrary dialog height and a
    long startup explanation cannot push its Close button off screen.
    """

    def __init__(
        self,
        title_key: str,
        message: Message,
        hint: Message | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title_key, parent, width=NOTICE_WIDTH)

        self.message_label = localize(QLabel(), text=message)
        self.message_label.setWordWrap(True)
        self.body_layout.addWidget(self.message_label)

        self.hint_label = QLabel()
        if hint is not None:
            localize(self.hint_label, text=hint)
        self.hint_label.setWordWrap(True)
        self.hint_label.setVisible(hint is not None)
        self.body_layout.addWidget(self.hint_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        self.add_footer_widget(self.buttons)
        self.set_initial_focus(self.buttons.button(QDialogButtonBox.StandardButton.Close))
        self.fit()


def _show_message(parent: QWidget | None, message: Message) -> None:
    MessageDialog("app.name", message, parent=parent).exec()


def warn_config_reset(parent: QWidget | None) -> None:
    _show_message(parent, Message("notice.configReset"))


def warn_jobs_stopping(parent: QWidget | None) -> None:
    _show_message(parent, Message("notice.stopping"))


class StartupFailureDialog(MessageDialog):
    """A deliberately plain fatal-startup surface without a severity icon.

    It is built without a parent, which is also what gives it a taskbar button of
    its own: it is not a dialog over a window here, it is the whole application,
    and a window the shell does not list is one the user cannot bring back.
    """

    def __init__(self, message: Message, hint: Message | None) -> None:
        super().__init__("startup.title", message, hint)


def show_startup_failure(message: Message, hint: Message | None) -> None:
    StartupFailureDialog(message, hint).exec()
