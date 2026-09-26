"""The catalogue gate (localization-conventions).

English defines the key set; every language has exactly it; placeholders and
plural forms match; nothing is left in English by accident; and no file carries a
character a reader cannot see. These read the files in the repository rather than
the parsed catalogues, because the hidden-character and trailing-newline checks
are about the file itself, which parsing erases.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from pixelup.i18n import plural
from pixelup.i18n.catalogue import LOCALES_DIR, catalogue, locale_file
from pixelup.i18n.languages import LANGUAGES, TAGS

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TAGS = ("en", "de", "es", "fr", "it", "pt-BR", "ru", "ja", "ko", "zh-Hans")

# Keys whose whole value is a name or a symbol, the same in every language: the
# product and brand names, and "4000 x 3000", whose x is a multiplication sign.
BRAND_KEYS = frozenset({"app.name", "about.github", "images.size"})

# Values a language shares with English on purpose. A listed entry that no longer
# matches English fails too, so the list cannot rot.
_UNITS = ["units.bytes", "units.gigabytes", "units.kilobytes", "units.megabytes"]
SAME_AS_ENGLISH: dict[str, list[str]] = {
    # "Version" and "Status" are the German words, System is the standard German
    # label for following the computer, and the byte units are the same symbols.
    "de": [
        "about.version",
        "managedModels.columnStatus",
        "queue.columnStatus",
        "settings.languageSystem",
        *_UNITS,
    ],
    # "General" is the Spanish word for a general settings group.
    "es": ["shortcuts.groupGeneral", *_UNITS],
    # French writes Version, Image(s) and Action as English does; its byte units
    # are its own (o, Ko, Mo, Go).
    "fr": [
        "about.version",
        "images.columnImage",
        "images.title",
        "managedModels.columnAction",
        "queue.columnImage",
    ],
    # Italian keeps "Output" as a column name; "Uscita" would read as an exit.
    "it": ["queue.columnOutput", *_UNITS],
    # "Status" is the everyday Brazilian Portuguese column name.
    "pt-BR": ["managedModels.columnStatus", "queue.columnStatus", *_UNITS],
    "ru": [],
    "ja": [*_UNITS],
    "ko": [*_UNITS],
    "zh-Hans": [*_UNITS],
}

_PLACEHOLDER = re.compile(r"\{([a-zA-Z]+)\}")
# Read as text, not as JSON: parsing makes an escape and the literal character
# the same, so this is the only place a literal no-break space can be caught
# (hidden-character-conventions). Built from code points, so this file holds
# none of the characters it looks for.
_HIDDEN = re.compile(
    "["
    + "".join(
        f"\\u{code:04x}" if isinstance(code, int) else f"\\u{code[0]:04x}-\\u{code[1]:04x}"
        for code in (
            (0x0000, 0x0008),
            0x000B,
            0x000C,
            (0x000E, 0x001F),
            0x007F,
            0x00A0,
            0x00AD,
            0x2007,
            (0x200B, 0x200F),
            0x2028,
            0x2029,
            (0x202A, 0x202F),
            0x2060,
            (0x2066, 0x2069),
            0xFEFF,
        )
    )
    + "]"
)


def _raw(tag: str) -> dict[str, object]:
    return json.loads(locale_file(tag).read_text(encoding="utf-8"))


def _forms(entry: object) -> list[tuple[str, str]]:
    if isinstance(entry, dict):
        return [(form, text) for form, text in entry.items()]
    assert isinstance(entry, str)
    return [("", entry)]


def _placeholders(text: str) -> list[str]:
    return sorted(set(_PLACEHOLDER.findall(text)))


def test_the_set_is_exactly_the_ten_interface_languages() -> None:
    files = sorted(path.stem for path in LOCALES_DIR.glob("*.json"))
    assert files == sorted(EXPECTED_TAGS)
    assert sorted(TAGS) == sorted(EXPECTED_TAGS)
    # The picker's order: Latin-script languages by their own names, then
    # Cyrillic, then Chinese, Japanese and Korean.
    assert [tag for tag, _name in LANGUAGES] == [
        "de", "en", "es", "fr", "it", "pt-BR", "ru", "zh-Hans", "ja", "ko",
    ]


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_every_language_has_exactly_english_keys(tag: str) -> None:
    assert sorted(_raw(tag)) == sorted(_raw("en"))


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_every_value_is_present_and_trimmed(tag: str) -> None:
    for key, entry in _raw(tag).items():
        for form, text in _forms(entry):
            assert text.strip(), f"{tag}: {key} {form} is empty"
            assert text == text.strip(), f"{tag}: {key} {form} has surrounding space"


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_plural_entries_use_exactly_their_language_categories(tag: str) -> None:
    english = _raw("en")
    expected = sorted(plural.categories_of(tag))
    for key, entry in _raw(tag).items():
        if isinstance(english[key], dict):
            assert isinstance(entry, dict), f"{tag}: {key} should be a plural entry"
            assert sorted(entry) == expected, f"{tag}: {key}"
        else:
            assert isinstance(entry, str), f"{tag}: {key} should be a sentence"


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_every_form_keeps_english_placeholders(tag: str) -> None:
    english = _raw("en")
    for key, entry in _raw(tag).items():
        expected = sorted(
            {name for _form, text in _forms(english[key]) for name in _placeholders(text)}
        )
        for form, text in _forms(entry):
            assert _placeholders(text) == expected, f"{tag}: {key} {form}"


@pytest.mark.parametrize("tag", [tag for tag in EXPECTED_TAGS if tag != "en"])
def test_no_language_copies_english(tag: str) -> None:
    english = _raw("en")
    allowed = SAME_AS_ENGLISH[tag]
    matched: set[str] = set()
    for key, entry in _raw(tag).items():
        if key in BRAND_KEYS:
            continue
        english_forms = dict(_forms(english[key]))
        for form, text in _forms(entry):
            # Only a value with words of its own can be a missed translation; the
            # letters inside {count} are a placeholder's name, not text.
            if not any(character.isalpha() for character in _PLACEHOLDER.sub("", text)):
                continue
            if english_forms.get(form, english_forms.get("other")) != text:
                continue
            assert key in allowed, f"{tag}: {key} is still the English {text!r}"
            matched.add(key)
    # A listed key that no longer matches English is stale and must go.
    assert sorted(matched) == sorted(set(allowed))


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_no_file_holds_a_character_the_reader_cannot_see(tag: str) -> None:
    text = locale_file(tag).read_text(encoding="utf-8")
    found = _HIDDEN.search(text)
    assert found is None, (
        f"{tag}.json holds U+{ord(found.group()):04X} at offset {found.start()}; "
        "write it as an escape"
    )
    assert text.endswith("\n")


@pytest.mark.parametrize("tag", EXPECTED_TAGS)
def test_the_shipped_catalogue_parses(tag: str) -> None:
    # The app reads the catalogues through its own loader; this is that path.
    assert catalogue(tag).keys() == _raw("en").keys()


def test_the_macos_bundle_declares_every_language() -> None:
    spec = (ROOT / "pixelup.spec").read_text(encoding="utf-8")
    declared = re.findall(r'"CFBundleLocalizations":\s*\[(.*?)\]', spec, flags=re.S)
    assert len(declared) == 1
    listed = re.findall(r'"([^"]+)"', declared[0])
    assert sorted(listed) == sorted(EXPECTED_TAGS)


def test_the_catalogues_ship_in_the_package_and_the_frozen_app() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "i18n/locales/*.json" in pyproject["tool"]["setuptools"]["package-data"]["pixelup"]
    spec = (ROOT / "pixelup.spec").read_text(encoding="utf-8")
    assert '("src/pixelup/i18n/locales/*.json", "pixelup/i18n/locales")' in spec
