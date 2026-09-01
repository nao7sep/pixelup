from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from pixelup.shortcuts_dialog import ShortcutsDialog, command_modifier_name


def test_shortcuts_dialog_catalogues_every_bound_chord(qapp: QApplication) -> None:
    dialog = ShortcutsDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        modifier = command_modifier_name()

        assert dialog.windowTitle() == "Keyboard shortcuts"
        assert "Keyboard shortcuts" in text
        assert "General" in text
        assert "Open Settings" in text
        assert f"{modifier}+Comma" in text
        assert "Show keyboard shortcuts" in text
        assert f"{modifier}+Slash/Question" in text
        assert len(dialog.findChildren(QLabel, "shortcutKey")) == 2
        assert dialog.findChild(QDialogButtonBox) is not None
    finally:
        dialog.deleteLater()
