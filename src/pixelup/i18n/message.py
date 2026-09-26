from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# This module imports nothing from Qt or the rest of PixelUp, so the processing
# modules (upscale, models, imaging, ...) can say what went wrong as a Message
# without depending on the interface that renders it.


@dataclass(frozen=True, slots=True)
class Message:
    """Text as it is held rather than shown: a catalogue key and its values.

    A processing module, a worker or a widget keeps one of these, and the
    translator renders it at the moment it reaches the screen, so a sentence
    follows a language change instead of freezing in the language it was built
    in, and no English is assembled anywhere but the catalogue
    (localization-conventions).

    A value may be another Message, rendered in the same language; a whole
    number, formatted for the reader's locale; a tuple of names, joined with the
    locale's list formatting; or a plain string, shown as it is (a filename, a
    model name).
    """

    key: str
    values: tuple[tuple[str, object], ...] = ()

    @classmethod
    def of(cls, key: str, /, **values: object) -> Message:
        return cls(key, tuple(values.items()))

    def value(self, name: str) -> object | None:
        for value_name, value in self.values:
            if value_name == name:
                return value
        return None

    def __str__(self) -> str:
        # The key, never English: a Message is meant to be rendered by the
        # translator, and if one ever reaches the screen unrendered the
        # rendered-key gate fails on it instead of a fixed language slipping by.
        return self.key


def join(key: str, parts: Sequence[Message]) -> Message:
    """Join counted or finished messages through one entry holding {first} and {rest}.

    Each part keeps its own plural form and the language decides order and
    separator, which a joined string cannot do.
    """
    if not parts:
        raise ValueError("join needs at least one message")
    if len(parts) == 1:
        return parts[0]
    return Message.of(key, first=parts[0], rest=join(key, parts[1:]))
