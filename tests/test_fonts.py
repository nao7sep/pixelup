from __future__ import annotations

import pytest
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


@pytest.fixture
def installed_font(monkeypatch: pytest.MonkeyPatch) -> str:
    # Windows' offscreen Qt plugin can expose no system font database at all. Patch
    # the toolkit query so these tests exercise PixelUp's selection logic rather than
    # depending on fonts supplied by the headless platform plugin.
    present = "PixelUp Test Font"
    monkeypatch.setattr(QFontDatabase, "families", staticmethod(lambda: [present]))
    return present


def test_resolve_ui_font_family_returns_first_installed(
    qapp: QApplication, installed_font: str
) -> None:
    # An absent family listed first must be skipped in favor of the installed one.
    assert resolve_ui_font_family(f"No Such Font 99999, {installed_font}") == installed_font


def test_resolve_ui_font_family_returns_none_when_nothing_installed(qapp: QApplication) -> None:
    assert resolve_ui_font_family("No Such Font 99999, Also Missing 88888") is None


def test_build_ui_font_uses_fixed_size_and_resolved_family(
    qapp: QApplication, installed_font: str
) -> None:
    font = build_ui_font(f"No Such Font 99999, {installed_font}")
    assert font.pixelSize() == DEFAULT_UI_FONT_SIZE
    assert font.family() == installed_font


def test_build_ui_font_uses_canonical_stack_when_configured_family_is_unresolved(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFontDatabase, "families", staticmethod(lambda: ["Segoe UI"]))
    font = build_ui_font("No Such Font 99999")
    assert font.pixelSize() == DEFAULT_UI_FONT_SIZE
    assert font.family() == "Segoe UI"


def test_build_ui_font_keeps_size_when_no_canonical_family_is_installed(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QFontDatabase, "families", staticmethod(list))
    font = build_ui_font("No Such Font 99999")
    assert font.pixelSize() == DEFAULT_UI_FONT_SIZE


def test_apply_ui_font_sets_application_font(
    qapp: QApplication, installed_font: str
) -> None:
    saved = qapp.font()
    try:
        apply_ui_font(qapp, installed_font)
        assert qapp.font().family() == installed_font
        assert qapp.font().pixelSize() == DEFAULT_UI_FONT_SIZE
    finally:
        qapp.setFont(saved)
