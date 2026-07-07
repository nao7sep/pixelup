from __future__ import annotations

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
from pixelup.ui_common import open_url, secondary_label, title_label, use_regular_spacing

PROJECT_URL = "https://github.com/nao7sep/pixelup"
ISSUES_URL = "https://github.com/nao7sep/pixelup/issues"


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About PixelUp")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        # A graduated rhythm rather than the uniform default: the outer margins
        # give the surface room, spacing is added explicitly per section so the
        # heading groups with its version and the sections below stay distinct.
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(0)

        name = title_label("PixelUp")
        version = secondary_label(f"Version {__version__}")
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        use_regular_spacing(links_layout, margins=False)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: open_url(PROJECT_URL))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(lambda: open_url(ISSUES_URL))
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        meta = secondary_label("(c) 2026 Yoshinao Inoguchi - MIT License")

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
        layout.addWidget(meta)
        layout.addSpacing(16)
        layout.addWidget(buttons)
