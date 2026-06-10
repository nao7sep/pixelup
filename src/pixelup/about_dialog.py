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
from pixelup.ui_common import open_url, use_regular_spacing

PROJECT_URL = "https://github.com/nao7sep/pixelup"
ISSUES_URL = "https://github.com/nao7sep/pixelup/issues"


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About PixelUp")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)
        name = QLabel("PixelUp")
        version = QLabel(f"Version {__version__}")
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        use_regular_spacing(links_layout, margins=False)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: open_url(PROJECT_URL))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(lambda: open_url(ISSUES_URL))
        links_layout.addStretch()
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        meta = QLabel("(c) 2026 Yoshinao Inoguchi - MIT License")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Close has RejectRole, so `rejected` covers both the button click and
        # Escape with a single, unambiguous close path.
        buttons.rejected.connect(self.reject)

        layout.addWidget(name)
        layout.addWidget(version)
        layout.addWidget(copy)
        layout.addWidget(links)
        layout.addWidget(meta)
        layout.addWidget(buttons)
