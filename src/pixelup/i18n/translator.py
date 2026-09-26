from __future__ import annotations

import re

from PySide6.QtCore import QLocale

from pixelup.i18n import plural
from pixelup.i18n.catalogue import catalogue
from pixelup.i18n.languages import ENGLISH
from pixelup.i18n.message import Message

_PLACEHOLDER = re.compile(r"\{([a-zA-Z]+)\}")


class Translator:
    """Turns a key and its values into the words on screen, in one language.

    Immutable and cheap: the app keeps one for the current language and builds a
    new one when the language changes, so nothing has to be told that the words
    underneath it moved.
    """

    def __init__(self, tag: str, locale: QLocale) -> None:
        self.tag = tag
        self.locale = locale
        self._catalogue = catalogue(tag)
        self._english = self._catalogue if tag == ENGLISH else catalogue(ENGLISH)

    def t(self, key: str, /, **values: object) -> str:
        """The words for ``key``, with its values filled in."""
        return self._render(key, tuple(values.items()))

    def of(self, message: Message) -> str:
        """The words for a held message."""
        return self._render(message.key, message.values)

    def number(self, value: int) -> str:
        """A whole number, grouped for the reader's locale."""
        return self.locale.toString(value)

    def _render(self, key: str, values: tuple[tuple[str, object], ...]) -> str:
        template = self._template(key, values)
        if template is None:
            return key
        if not values:
            return template
        named = dict(values)
        return _PLACEHOLDER.sub(
            lambda match: self._value(named[match.group(1)])
            if match.group(1) in named
            else match.group(0),
            template,
        )

    def _template(self, key: str, values: tuple[tuple[str, object], ...]) -> str | None:
        # A key the language is missing falls back to English rather than to
        # nothing, so a catalogue that slipped past the gate costs the reader one
        # English sentence, not a blank control.
        language = self.tag
        entry = self._catalogue.get(key)
        if entry is None:
            entry = self._english.get(key)
            language = ENGLISH
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        # The form follows the language the words came from, so a fallback
        # sentence still agrees with its own number.
        count = dict(values).get("count")
        category = (
            plural.category_for(language, count)
            if isinstance(count, int) and not isinstance(count, bool)
            else plural.OTHER
        )
        return entry.get(category, entry.get(plural.OTHER))

    def _value(self, value: object) -> str:
        if isinstance(value, Message):
            return self.of(value)
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            # A number filled into a sentence is formatted for the reader's
            # locale, so its separators look the way they do everywhere else on
            # their computer.
            return self.number(value)
        if isinstance(value, float):
            # The interface's only fractional values are sizes, shown to one place.
            return self.locale.toString(value, "f", 1)
        if isinstance(value, tuple | list):
            # Lists of names use the locale's own list formatting, never ", ".
            return self.locale.createSeparatedList([str(item) for item in value])
        return str(value)
