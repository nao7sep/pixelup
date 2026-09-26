from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QLocale

# The interface languages PixelUp speaks (localization-conventions): the ten tags,
# the name each one is shown under, how a saved preference is read, and how
# System resolves against the computer's own languages. A tag is BCP 47 and never
# shown to the user; a name is written in its own language so a reader finds it
# whatever language is currently showing.

SYSTEM = "system"
"""The saved value that means "follow the computer"."""

ENGLISH = "en"
"""The source language, and the fallback for a computer outside the set."""

# The picker's order: the Latin-script languages alphabetically by their own
# names, then Cyrillic, then Chinese, Japanese and Korean.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("de", "Deutsch"),
    ("en", "English"),
    ("es", "Español"),
    ("fr", "Français"),
    ("it", "Italiano"),
    ("pt-BR", "Português"),
    ("ru", "Русский"),
    ("zh-Hans", "中文"),
    ("ja", "日本語"),
    ("ko", "한국어"),
)

TAGS: tuple[str, ...] = tuple(tag for tag, _name in LANGUAGES)

# Qt names its own translation files (qtbase_<name>.qm) by the older locale
# naming. English needs none: Qt's own text is English.
_QT_TRANSLATION_NAMES = {
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt-BR": "pt_BR",
    "ru": "ru",
    "zh-Hans": "zh_CN",
    "ja": "ja",
    "ko": "ko",
}


def name_of(tag: str) -> str:
    """The name ``tag`` is shown under, or the tag itself outside the set."""
    return dict(LANGUAGES).get(tag, tag)


def normalize_preference(saved: object) -> str:
    """A saved preference as the app acts on it: a tag in the set, or System.

    A missing, blank or unrecognized value means System, so a hand-edited file can
    never leave the app without a language.
    """
    if not isinstance(saved, str) or not saved.strip():
        return SYSTEM
    text = saved.strip()
    if text.casefold() == SYSTEM:
        return SYSTEM
    for tag in TAGS:
        if tag.casefold() == text.casefold():
            return tag
    return SYSTEM


def resolve(preference: object, computer_languages: Iterable[str]) -> str:
    """The language a preference resolves to, given the computer's languages in order."""
    normalized = normalize_preference(preference)
    return match(computer_languages) if normalized == SYSTEM else normalized


def match(computer_languages: Iterable[str]) -> str:
    """The first of the computer's languages that is in the set, and English when none is.

    Every Chinese locale resolves to Simplified Chinese, every Portuguese to
    Brazilian Portuguese and every Spanish to the one neutral Spanish, because the
    app ships only that one variety of each.
    """
    for candidate in computer_languages:
        tag = _match_one(candidate)
        if tag is not None:
            return tag
    return ENGLISH


def _match_one(candidate: str) -> str | None:
    # "ja-JP", "zh-Hant-TW" and "pt_BR" all arrive here; only the primary language
    # subtag decides. Qt answers "C" when the computer names no locale at all.
    parts = [part for part in candidate.strip().replace("_", "-").split("-") if part]
    if not parts:
        return None
    primary = parts[0].casefold()
    return {"zh": "zh-Hans", "pt": "pt-BR"}.get(primary, primary if primary in TAGS else None)


def qt_translation_name(tag: str) -> str | None:
    """The name of Qt's own translation file for ``tag``, or None for English."""
    return _QT_TRANSLATION_NAMES.get(tag)


def formatting_locale(tag: str, computer_locale: QLocale) -> QLocale:
    """The locale dates and numbers are formatted in (timestamp-conventions).

    The computer's own regional format when the computer already works in the
    interface language, so a reader keeps the separators they chose, and
    otherwise the language's own.
    """
    if _match_one(computer_locale.bcp47Name()) == tag:
        return computer_locale
    return QLocale(tag)
