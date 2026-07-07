from __future__ import annotations

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from pixelup.fonts import (
    DEFAULT_UI_FONT_FAMILY,
    DEFAULT_UI_FONT_SIZE,
    apply_ui_font,
    build_ui_font,
    normalize_font_family,
    parse_font_families,
    resolve_ui_font_family,
)


def test_normalize_font_family_trims_and_keeps_value() -> None:
    assert normalize_font_family("  Arial  ", DEFAULT_UI_FONT_FAMILY) == "Arial"


def test_normalize_font_family_falls_back_on_blank_or_non_string() -> None:
    assert normalize_font_family("", DEFAULT_UI_FONT_FAMILY) == DEFAULT_UI_FONT_FAMILY
    assert normalize_font_family("   ", DEFAULT_UI_FONT_FAMILY) == DEFAULT_UI_FONT_FAMILY
    assert normalize_font_family(None, DEFAULT_UI_FONT_FAMILY) == DEFAULT_UI_FONT_FAMILY
    assert normalize_font_family(42, DEFAULT_UI_FONT_FAMILY) == DEFAULT_UI_FONT_FAMILY


def test_parse_font_families_splits_strips_quotes_and_drops_empties() -> None:
    assert parse_font_families('"Helvetica Neue", Segoe UI , , \'Roboto\'') == [
        "Helvetica Neue",
        "Segoe UI",
        "Roboto",
    ]


def test_resolve_ui_font_family_returns_first_installed(qapp: QApplication) -> None:
    installed = QFontDatabase.families()
    assert installed  # offscreen Qt still ships built-in families
    present = installed[0]
    # An absent family listed first must be skipped in favor of the installed one.
    assert resolve_ui_font_family(f"No Such Font 99999, {present}") == present


def test_resolve_ui_font_family_returns_none_when_nothing_installed(qapp: QApplication) -> None:
    assert resolve_ui_font_family("No Such Font 99999, Also Missing 88888") is None


def test_build_ui_font_uses_fixed_size_and_resolved_family(qapp: QApplication) -> None:
    present = QFontDatabase.families()[0]
    font = build_ui_font(f"No Such Font 99999, {present}")
    assert font.pointSize() == DEFAULT_UI_FONT_SIZE
    assert font.family() == present


def test_build_ui_font_keeps_size_when_family_unresolved(qapp: QApplication) -> None:
    # An entirely-absent family never crashes; Qt's default family is kept and the
    # explicit UI size still applies.
    font = build_ui_font("No Such Font 99999")
    assert font.pointSize() == DEFAULT_UI_FONT_SIZE


def test_apply_ui_font_sets_application_font(qapp: QApplication) -> None:
    saved = qapp.font()
    try:
        present = QFontDatabase.families()[0]
        apply_ui_font(qapp, present)
        assert qapp.font().family() == present
        assert qapp.font().pointSize() == DEFAULT_UI_FONT_SIZE
    finally:
        qapp.setFont(saved)
