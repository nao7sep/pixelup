from __future__ import annotations

import sys

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import use_regular_spacing


def command_modifier_name() -> str:
    return "Cmd" if sys.platform == "darwin" else "Ctrl"


class ShortcutsDialog(QDialog):
    """Named catalogue of every shortcut PixelUp binds."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)

        group = QLabel("General")
        font = group.font()
        font.setBold(True)
        group.setFont(font)
        layout.addWidget(group)

        modifier = command_modifier_name()
        shortcuts = QFormLayout()
        use_regular_spacing(shortcuts, margins=False)
        shortcuts.addRow("Open Settings", QLabel(f"{modifier}+Comma"))
        shortcuts.addRow(
            "Show keyboard shortcuts",
            QLabel(f"{modifier}+Slash/Question"),
        )
        layout.addLayout(shortcuts)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
