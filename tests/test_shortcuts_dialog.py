from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFrame, QLabel

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
        assert len(dialog.findChildren(QLabel, "shortcutKey")) == 2
        assert dialog.findChild(QDialogButtonBox) is not None
    finally:
        dialog.deleteLater()


def test_the_shortcut_list_carries_no_card(qapp: QApplication) -> None:
    # A reference list, read and never navigated: its heading and the space between
    # rows separate it, and the one mark on the surface is each key. The card it sat
    # in was also what filled the dark theme with the palette's near-black base.
    dialog = ShortcutsDialog()
    try:
        assert dialog.findChild(QFrame, "shortcutGroup") is None
        for label in dialog.findChildren(QLabel, "shortcutKey"):
            assert label.styleSheet() == ""
            assert label.parentWidget().styleSheet() == ""
    finally:
        dialog.deleteLater()
