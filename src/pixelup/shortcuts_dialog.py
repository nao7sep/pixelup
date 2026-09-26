from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pixelup.dialog_shell import FORM_WIDTH, DialogShell
from pixelup.i18n.localized import localize
from pixelup.ui_common import secondary_label


def command_modifier_name() -> str:
    return "Cmd" if sys.platform == "darwin" else "Ctrl"


def shortcut_chord(*keys: str) -> str:
    """A chord as the shortcuts list shows it. Key tokens stay English in every
    language, because they name the keycaps (keyboard-shortcut-conventions)."""
    return "+".join((command_modifier_name(), *keys))


class ShortcutsDialog(DialogShell):
    """Named catalogue of every shortcut PixelUp binds."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("shortcuts.title", parent, width=FORM_WIDTH)

        introduction = localize(secondary_label(""), text="shortcuts.introduction")
        self.body_layout.addWidget(introduction)

        # A reference list, read and never navigated, so it carries no card: the
        # heading and the space between rows separate it, and the one mark on the
        # surface is each key (theme.py draws the key chip).
        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 4, 0, 0)
        group_layout.setSpacing(12)

        group_heading = localize(QLabel(), text="shortcuts.groupGeneral")
        group_font = group_heading.font()
        group_font.setBold(True)
        group_heading.setFont(group_font)
        group_layout.addWidget(group_heading)

        group_layout.addWidget(_shortcut_row("shortcuts.openSettings", shortcut_chord("Comma")))
        group_layout.addWidget(
            _shortcut_row("shortcuts.showShortcuts", shortcut_chord("Slash/Question"))
        )
        self.body_layout.addWidget(group)
        self.body_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.add_footer_widget(buttons)
        self.set_initial_focus(buttons.button(QDialogButtonBox.StandardButton.Close))
        self.fit()


def _shortcut_row(action_key: str, chord: str) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)

    action_label = localize(QLabel(), text=action_key)
    key_label = QLabel(chord)
    key_label.setObjectName("shortcutKey")
    key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    layout.addWidget(action_label, 1)
    layout.addWidget(key_label, 0)
    return row
