from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from pixelup import __version__
from pixelup.dialog_shell import NOTICE_WIDTH, DialogShell
from pixelup.i18n.localized import localize
from pixelup.i18n.message import Message
from pixelup.session_log import log
from pixelup.ui_common import open_url, secondary_label, title_label, use_regular_spacing
from pixelup.widgets import OperationResult

PROJECT_URL = "https://github.com/nao7sep/pixelup"
ISSUES_URL = "https://github.com/nao7sep/pixelup/issues"


class AboutDialog(DialogShell):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        opener: Callable[[str], None] = open_url,
    ) -> None:
        super().__init__("about.title", parent, width=NOTICE_WIDTH)
        self._opener = opener

        # The one heading a native dialog keeps. It does not repeat the window's
        # title — it names the product, which is what an about surface is for, and
        # carries the version beside it (modal-dialog-conventions).
        name = title_label("PixelUp")
        version = localize(
            secondary_label(""), text=Message.of("about.version", version=__version__)
        )
        copy = localize(QLabel(), text="about.description")
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        use_regular_spacing(links_layout, margins=False)
        github_button = localize(QPushButton(), text="about.github")
        github_button.clicked.connect(
            lambda: self._open_external(PROJECT_URL, Message("about.openGitHubFailed"))
        )
        issues_button = localize(QPushButton(), text="about.reportIssue")
        issues_button.clicked.connect(
            lambda: self._open_external(ISSUES_URL, Message("about.openIssuesFailed"))
        )
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        self.launch_result = OperationResult(
            object_name="aboutLaunchResult",
            dismissible=True,
        )

        meta = localize(secondary_label(""), text="about.copyright")

        # A graduated rhythm rather than the body's uniform spacing: the heading
        # groups with its version and the sections below stay distinct.
        self.body_layout.setSpacing(0)
        self.body_layout.addWidget(name)
        self.body_layout.addSpacing(4)
        self.body_layout.addWidget(version)
        self.body_layout.addSpacing(12)
        self.body_layout.addWidget(copy)
        self.body_layout.addSpacing(16)
        self.body_layout.addWidget(links)
        self.body_layout.addSpacing(16)
        self.body_layout.addWidget(self.launch_result)
        self.body_layout.addWidget(meta)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Close has RejectRole, so `rejected` covers both the button click and
        # Escape with a single, unambiguous close path.
        buttons.rejected.connect(self.reject)
        self.add_footer_widget(buttons)
        self.set_initial_focus(buttons.button(QDialogButtonBox.StandardButton.Close))
        self.fit()

    def _open_external(self, url: str, failure: Message) -> None:
        try:
            self._opener(url)
        except Exception:  # noqa: BLE001 - native URL handlers can fail arbitrarily.
            log.exception("about.external_open_failed", url=url)
            self.launch_result.show_result(failure, severity="error")
            return
        self.launch_result.clear_result()
