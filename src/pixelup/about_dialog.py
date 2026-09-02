from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelup import __version__
from pixelup.session_log import log
from pixelup.ui_common import open_url, secondary_label, title_label, use_regular_spacing
from pixelup.widgets import OperationResult

PROJECT_URL = "https://github.com/nao7sep/pixelup"
ISSUES_URL = "https://github.com/nao7sep/pixelup/issues"


class AboutDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        opener: Callable[[str], None] = open_url,
    ) -> None:
        super().__init__(parent)
        self._opener = opener
        self.setWindowTitle("About PixelUp")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        # A graduated rhythm rather than the uniform default: the outer margins
        # give the surface room, spacing is added explicitly per section so the
        # heading groups with its version and the sections below stay distinct.
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(0)

        name = title_label("About PixelUp")
        version = secondary_label(f"Version {__version__}")
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        use_regular_spacing(links_layout, margins=False)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: self._open_external(PROJECT_URL, "GitHub"))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(
            lambda: self._open_external(ISSUES_URL, "the issue tracker")
        )
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        self.launch_result = OperationResult(
            object_name="aboutLaunchResult",
            dismissible=True,
        )

        meta = secondary_label("© 2026 Yoshinao Inoguchi · MIT License")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Close has RejectRole, so `rejected` covers both the button click and
        # Escape with a single, unambiguous close path.
        buttons.rejected.connect(self.reject)

        layout.addWidget(name)
        layout.addSpacing(4)
        layout.addWidget(version)
        layout.addSpacing(12)
        layout.addWidget(copy)
        layout.addSpacing(16)
        layout.addWidget(links)
        layout.addSpacing(16)
        layout.addWidget(self.launch_result)
        layout.addWidget(meta)
        layout.addSpacing(16)
        layout.addWidget(buttons)

    def _open_external(self, url: str, destination: str) -> None:
        try:
            self._opener(url)
        except Exception:  # noqa: BLE001 - native URL handlers can fail arbitrarily.
            log.exception("about.external_open_failed", url=url)
            self.launch_result.show_result(
                f"Could not open {destination}. Check the log and try again.",
                severity="error",
            )
            return
        self.launch_result.clear_result()
