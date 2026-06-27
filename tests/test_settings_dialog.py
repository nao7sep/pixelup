from __future__ import annotations

from PySide6.QtWidgets import QApplication

from pixelup.app_config import MAX_CONCURRENT_JOBS, MAX_TILE, AppConfig
from pixelup.paths import OutputFormat
from pixelup.settings_dialog import SettingsDialog


def test_round_trips_config_without_changes(qapp: QApplication) -> None:
    config = AppConfig(
        max_concurrent_jobs=4,
        output_format=OutputFormat.JPG,
        quality=80,
        tile=512,
        device="cpu",
        auto_download=False,
    )
    dialog = SettingsDialog(config)
    try:
        assert dialog.config() == config
        assert dialog.is_dirty() is False
    finally:
        dialog.deleteLater()


def test_round_trips_boundary_values_without_spurious_dirty(qapp: QApplication) -> None:
    # The spinbox ranges must accommodate the full config domain, otherwise a
    # max-valued config would be clamped on display and report false-dirty.
    config = AppConfig(
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
        tile=MAX_TILE,
        output_format=OutputFormat.WEBP,
    )
    dialog = SettingsDialog(config)
    try:
        assert dialog.config() == config
        assert dialog.is_dirty() is False
        # The format combo must remain data-backed (value, not display text).
        assert dialog.format.currentData() == config.output_format.value
    finally:
        dialog.deleteLater()


def test_commit_disabled_until_dirty_and_back(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        assert dialog.is_dirty() is False
        assert dialog.ok_button.isEnabled() is False

        dialog.quality.setValue(dialog.quality.value() + 1)
        assert dialog.is_dirty() is True
        assert dialog.ok_button.isEnabled() is True

        # Reverting back to the opened value clears the dirty flag again.
        dialog.quality.setValue(AppConfig().quality)
        assert dialog.is_dirty() is False
        assert dialog.ok_button.isEnabled() is False
    finally:
        dialog.deleteLater()


def test_restore_defaults_marks_dirty_when_config_differs(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig(quality=10, tile=256, auto_download=False))
    try:
        assert dialog.ok_button.isEnabled() is False
        dialog._restore_defaults()
        assert dialog.config() == AppConfig()
        assert dialog.is_dirty() is True
        assert dialog.ok_button.isEnabled() is True
    finally:
        dialog.deleteLater()


def test_restore_defaults_stays_clean_when_already_default(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig())
    try:
        dialog._restore_defaults()
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


def test_restore_defaults_resets_font_family(qapp: QApplication) -> None:
    dialog = SettingsDialog(AppConfig(font_family="Courier New"))
    try:
        dialog._restore_defaults()
        assert dialog.font_family.text() == AppConfig().font_family
        assert dialog.config().font_family == AppConfig().font_family
    finally:
        dialog.deleteLater()
