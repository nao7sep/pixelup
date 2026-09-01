from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import secondary_label, title_label, use_dialog_spacing


def command_modifier_name() -> str:
    return "Cmd" if sys.platform == "darwin" else "Ctrl"


class ShortcutsDialog(QDialog):
    """Named catalogue of every shortcut PixelUp binds."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setModal(True)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)

        layout.addWidget(title_label("Keyboard shortcuts"))
        introduction = secondary_label("Use these shortcuts anywhere in PixelUp.")
        layout.addWidget(introduction)

        group = QFrame()
        group.setObjectName("shortcutGroup")
        group.setStyleSheet(
            "QFrame#shortcutGroup {"
            " border: 1px solid palette(mid);"
            " border-radius: 7px;"
            " background: palette(base);"
            "}"
            "QLabel#shortcutKey {"
            " border: 1px solid palette(mid);"
            " border-radius: 4px;"
            " padding: 4px 8px;"
            " background: palette(window);"
            " font-weight: 600;"
            "}"
        )
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(14, 14, 14, 14)
        group_layout.setSpacing(12)

        group_heading = QLabel("General")
        group_font = group_heading.font()
        group_font.setBold(True)
        group_heading.setFont(group_font)
        group_layout.addWidget(group_heading)

        modifier = command_modifier_name()
        group_layout.addWidget(_shortcut_row("Open Settings", f"{modifier}+Comma"))
        group_layout.addWidget(
            _shortcut_row(
                "Show keyboard shortcuts",
                f"{modifier}+Slash/Question",
            )
        )
        layout.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _shortcut_row(action: str, chord: str) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)

    action_label = QLabel(action)
    key_label = QLabel(chord)
    key_label.setObjectName("shortcutKey")
    key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    layout.addWidget(action_label, 1)
    layout.addWidget(key_label, 0)
    return row
