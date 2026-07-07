from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

# Canonical default UI font family for PixelUp. PixelUp bundles no font, so the
# default is a small cross-platform stack of system UI faces — macOS (Helvetica
# Neue), Windows (Segoe UI), and common Linux fallbacks (Roboto, then Arial).
# The family is free text the user can override; resolution picks the first of
# these actually installed (see resolve_ui_font_family), so one default reads
# well on every platform without bundling anything.
DEFAULT_UI_FONT_FAMILY = "Helvetica Neue, Segoe UI, Roboto, Arial"

# The UI font size is deliberate and fixed, not user-configurable. Per the
# app-chrome-conventions the UI font is family-only — a base-size knob breaks
# Qt's pixel-based layouts — yet every element still gets an explicit size rather
# than the bare toolkit default. 13pt is PixelUp's base; PixelUp set no explicit
# font at all before this, inheriting whatever Qt/OS picked.
DEFAULT_UI_FONT_SIZE = 13


def normalize_font_family(value: Any, default: str) -> str:
    """Normalize a persisted or entered family string, falling back when blank.

    The stored value is free text (possibly a comma-separated stack). This only
    trims it and substitutes the default when it is missing or empty; it does not
    touch the font database, so it stays usable at config-load time with no
    running QApplication. Matching an entered family to an installed one happens
    later, at apply time, in resolve_ui_font_family.
    """
    if not isinstance(value, str):
        return default
    text = value.strip()
    return text if text else default


def parse_font_families(value: str) -> list[str]:
    """Split a comma-separated family string into trimmed, unquoted names."""
    names: list[str] = []
    for part in value.split(","):
        name = part.strip().strip("\"'").strip()
        if name:
            names.append(name)
    return names


def resolve_ui_font_family(value: str) -> str | None:
    """Return the first requested family actually installed, or None.

    Native Qt renders one family, so — per the app-chrome-conventions' native
    input rule — PixelUp resolves the free-text (possibly comma-separated) string
    itself: the first listed family present in the font database wins. None means
    nothing matched, so the caller leaves Qt's own default family in place.
    """
    from PySide6.QtGui import QFontDatabase

    installed = set(QFontDatabase.families())
    for name in parse_font_families(value):
        if name in installed:
            return name
    return None


def build_ui_font(value: str) -> QFont:
    """Build the UI font from a family string at the fixed UI size.

    Resolves the family against installed fonts; when none match, the QFont keeps
    Qt's default family — a graceful implicit fallback so a typo never yields
    broken text. The size is always the explicit DEFAULT_UI_FONT_SIZE.
    """
    from PySide6.QtGui import QFont

    font = QFont()
    family = resolve_ui_font_family(value)
    if family is not None:
        font.setFamily(family)
    font.setPointSize(DEFAULT_UI_FONT_SIZE)
    return font


def apply_ui_font(app: QApplication, value: str) -> None:
    """Set PixelUp's UI font on the application from a family string."""
    app.setFont(build_ui_font(value))
