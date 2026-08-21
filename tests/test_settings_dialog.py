from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from pixelup.app_config import MAX_CONCURRENT_JOBS, AppConfig
from pixelup.jobs import JobSettings
from pixelup.settings_dialog import SettingsDialog

# The dialog holds only what the main window does not show: the UI font, the
# model-download toggle, and the concurrent job count. The image-processing
# parameters live in the main window's Parameters panel and are no business of this
# dialog — not to edit, and not to reset.


def test_round_trips_config_without_changes(qapp: QApplication) -> None:
    config = AppConfig(
        max_concurrent_jobs=4,
        auto_download=False,
        font_family="Courier New",
    )
    dialog = SettingsDialog(config)
    try:
        assert dialog.config() == config
        assert dialog.is_dirty() is False
    finally:
        dialog.deleteLater()


def test_round_trips_boundary_values_without_spurious_dirty(qapp: QApplication) -> None:
    # The spinbox range must accommodate the full config domain, otherwise a
    # max-valued config would be clamped on display and report false-dirty.
    config = AppConfig(max_concurrent_jobs=MAX_CONCURRENT_JOBS)
    dialog = SettingsDialog(config)
    try:
        assert dialog.config() == config
        assert dialog.is_dirty() is False
    finally:
        dialog.deleteLater()


def test_commit_disabled_until_dirty_and_back(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        assert dialog.is_dirty() is False
        assert dialog.ok_button.isEnabled() is False

        dialog.concurrent.setValue(dialog.concurrent.value() + 1)
        assert dialog.is_dirty() is True
        assert dialog.ok_button.isEnabled() is True

        # Reverting back to the opened value clears the dirty flag again.
        dialog.concurrent.setValue(AppConfig().max_concurrent_jobs)
        assert dialog.is_dirty() is False
        assert dialog.ok_button.isEnabled() is False
    finally:
        dialog.deleteLater()


def test_font_family_field_reflects_config(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig(font_family="Courier New"))
    try:
        assert dialog.font_family.text() == "Courier New"
        assert dialog.config().font_family == "Courier New"
        assert dialog.is_dirty() is False
    finally:
        dialog.deleteLater()


def test_font_family_change_marks_dirty_and_normalizes(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        dialog.font_family.setText("  Menlo  ")
        assert dialog.is_dirty() is True
        assert dialog.ok_button.isEnabled() is True
        assert dialog.config().font_family == "Menlo"
    finally:
        dialog.deleteLater()


def test_blank_font_family_falls_back_to_default(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        dialog.font_family.setText("")
        assert dialog.config().font_family == AppConfig().font_family
    finally:
        dialog.deleteLater()


def test_auto_download_toggle_marks_dirty(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig(auto_download=True))
    try:
        assert dialog.auto_download.isChecked() is True
        dialog.auto_download.setChecked(False)
        assert dialog.config().auto_download is False
        assert dialog.is_dirty() is True
    finally:
        dialog.deleteLater()


def test_auto_download_opt_in_discloses_material_download_size(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))

        assert "3–67 MB" in text
        assert "543 MB" in text
        assert "disk use" in text
    finally:
        dialog.deleteLater()


def test_dialog_exposes_no_parameter_controls(qapp: QApplication) -> None:
    # One home per thing: output format, quality, tile and device belong to the main
    # window's Parameters panel now, and must not reappear here as a second home.
    dialog = SettingsDialog(AppConfig())
    try:
        for removed in ("format", "quality", "tile", "device"):
            assert not hasattr(dialog, removed)
    finally:
        dialog.deleteLater()


def test_dialog_has_no_reset_button(qapp: QApplication) -> None:
    # Nothing left in here is a stale-able built-in or a tuned, interacting set: a
    # font is a preference, a toggle is a toggle, a job count is one number. So there
    # is no reset, and no handler behind one.
    dialog = SettingsDialog(AppConfig())
    try:
        labels = [button.text() for button in dialog.findChildren(QPushButton)]
        assert not any("Reset" in label for label in labels)
        # `_restore_defaults` is the name the removed handler actually had (it backed
        # a "Restore defaults" button). This asserted `_reset_settings_to_defaults`
        # before — a name that has never existed in this codebase, so it passed
        # unconditionally and would have passed with the real handler still in place.
        assert not hasattr(dialog, "_restore_defaults")
    finally:
        dialog.deleteLater()


def test_accepting_the_dialog_leaves_the_parameters_panel_untouched(qapp: QApplication) -> None:
    # The dialog must carry the settings it does not show straight through. Building a
    # fresh AppConfig here instead would silently reset the user's whole Parameters
    # panel to the built-ins every time they changed their font.
    parameters = JobSettings(quality=10, tile=1024, device="cpu", face_enhance=True)
    dialog = SettingsDialog(AppConfig(parameters=parameters))
    try:
        dialog.font_family.setText("Menlo")
        dialog.concurrent.setValue(4)

        committed = dialog.config()

        assert committed.parameters == parameters
        assert committed.font_family == "Menlo"
        assert committed.max_concurrent_jobs == 4
    finally:
        dialog.deleteLater()
