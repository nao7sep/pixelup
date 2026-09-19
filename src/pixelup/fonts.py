from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

# The persisted, user-facing default is blank: it means "use the operating
# system's own UI font" without exposing that implementation detail in Settings.
DEFAULT_UI_FONT_FAMILY = ""

# The built-in stack earlier versions resolved, and stored as the setting's value.
# A saved copy of it reads as the blank default.
LEGACY_UI_FONT_FAMILY_STACK = "Helvetica Neue, Segoe UI, Roboto, Arial"

# The UI font size is deliberate and fixed, not user-configurable. Per the
# app-chrome-conventions the UI font is family-only — a base-size knob breaks
# Qt's pixel-based layouts. The application owns one inherited 13-logical-pixel
# base; point sizing would convert through platform DPI and make the Windows UI
# substantially larger.
DEFAULT_UI_FONT_SIZE = 13


def normalize_font_family(value: Any, default: str = DEFAULT_UI_FONT_FAMILY) -> str:
    """Normalize a persisted or entered family string.

    The stored value is free text (possibly a comma-separated stack). This only
    trims it and substitutes the user-facing default when it is missing or empty;
    it does not touch the font database, so it stays usable at config-load time
    with no running QApplication. The former built-in stack is collapsed to blank
    because it has the same effective behavior and should not remain exposed in
    settings written before this distinction existed. Matching an entered family
    to an installed one happens later, at apply time, in resolve_ui_font_family.
    """
    if not isinstance(value, str):
        return default
    text = value.strip()
    if not text or text == LEGACY_UI_FONT_FAMILY_STACK:
        return default
    return text


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
    itself: the first listed family present in the font database wins.
    """
    from PySide6.QtGui import QFontDatabase

    installed = set(QFontDatabase.families())
    for name in parse_font_families(value):
        if name in installed:
            return name
    return None


def build_ui_font(value: str) -> QFont:
    """Build the UI font from a family string at the fixed UI size.

    Resolves the family against installed fonts. A blank or unresolved value
    takes the operating system's own UI family, Qt's general system font (San
    Francisco on macOS, Segoe UI on Windows, the desktop's font on Linux), per
    the app-chrome-conventions. The size is always the explicit
    DEFAULT_UI_FONT_SIZE.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    font = QFont()
    family = resolve_ui_font_family(value)
    if family is None:
        family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    if family:
        font.setFamily(family)
    font.setPixelSize(DEFAULT_UI_FONT_SIZE)
    return font


def apply_ui_font(app: QApplication, value: str) -> None:
    """Set PixelUp's UI font on the application from a family string."""
    app.setFont(build_ui_font(value))
