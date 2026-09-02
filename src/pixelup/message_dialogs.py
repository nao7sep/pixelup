from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import use_dialog_spacing, use_regular_spacing

_APP = "PixelUp"
_BODY_MIN_WIDTH = 340
_BODY_MAX_WIDTH = 480
_BODY_MAX_HEIGHT = 420


class MessageDialog(QDialog):
    """An icon-free message whose body grows naturally, then scrolls.

    The window title and footer stay fixed. Only the prose body is capped, so a
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
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        use_regular_spacing(body_layout, margins=False)

        self.message_label = QLabel(user_message)
        self.message_label.setWordWrap(True)
        body_layout.addWidget(self.message_label)

        self.hint_label = QLabel(user_hint or "")
        self.hint_label.setWordWrap(True)
        self.hint_label.setVisible(bool(user_hint))
        body_layout.addWidget(self.hint_label)

        self.body_scroll = QScrollArea()
        self.body_scroll.setObjectName("messageBodyScroll")
        self.body_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.body_scroll.setWidget(self.body)
        layout.addWidget(self.body_scroll)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._fit_to_content()

    def _fit_to_content(self) -> None:
        self.body.setMinimumWidth(_BODY_MIN_WIDTH)
        self.body.setMaximumWidth(_BODY_MAX_WIDTH)
        self.body.layout().activate()

        screen = self.screen() or QApplication.primaryScreen()
        work_height = screen.availableGeometry().height() if screen is not None else 720
        body_cap = min(_BODY_MAX_HEIGHT, max(160, int(work_height * 0.65)))
        body_height = min(self.body.sizeHint().height(), body_cap)

        # A fixed viewport height is intentional: it is the natural body height
        # until the cap, while the outer dialog and footer keep their own hints.
        self.body_scroll.setFixedHeight(body_height)
        self.body_scroll.setMinimumWidth(_BODY_MIN_WIDTH)
        self.body_scroll.setMaximumWidth(_BODY_MAX_WIDTH)
        self.adjustSize()


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
    """A deliberately plain fatal-startup surface without a severity icon."""

    def __init__(self, user_message: str, user_hint: str | None) -> None:
        super().__init__(
            "PixelUp could not start",
            user_message,
            user_hint,
        )


def show_startup_failure(user_message: str, user_hint: str | None) -> None:
    StartupFailureDialog(user_message, user_hint).exec()
