from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from pixelup import shortcuts_dialog
from pixelup.shortcuts_dialog import ShortcutsDialog, command_modifier_name


def test_shortcuts_dialog_catalogues_every_bound_chord(qapp: QApplication) -> None:
    dialog = ShortcutsDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        modifier = command_modifier_name()

        assert dialog.windowTitle() == "Keyboard shortcuts"
        # Named by its title bar, not again inside the body.
        assert "Keyboard shortcuts" not in text
        assert "General" in text
        assert "Open Settings" in text
        assert f"{modifier}+Comma" in text
        assert "Show keyboard shortcuts" in text
        assert f"{modifier}+Slash/Question" in text
        assert "Close window" in text
        assert f"{modifier}+W" in text
        assert "Open images" in text
        assert f"{modifier}+O" in text
        assert "Remove the selected image" in text
        assert "Backspace/Delete" in text
        assert "Queue selected image" in text
        assert f"{modifier}+Enter" in text
        assert len(dialog.findChildren(QLabel, "shortcutKey")) == 6
        assert dialog.findChild(QDialogButtonBox) is not None
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize(("platform", "name"), [("darwin", "Cmd"), ("win32", "Ctrl")])
def test_the_command_modifier_is_named_for_the_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, name: str
) -> None:
    monkeypatch.setattr(shortcuts_dialog.sys, "platform", platform)
    assert command_modifier_name() == name
