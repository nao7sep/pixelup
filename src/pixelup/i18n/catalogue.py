from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from pixelup.i18n.languages import TAGS

# One language's text: flat dotted keys, each holding either a sentence or, where
# a count changes the words around it, one sentence per plural form. The files
# ship inside the package (and the frozen app, beside this module), so the
# catalogues cannot be separated from the code they were tested with.

LOCALES_DIR = Path(__file__).parent / "locales"

Entry = str | dict[str, str]


def locale_file(tag: str) -> Path:
    return LOCALES_DIR / f"{tag}.json"


@cache
def catalogue(tag: str) -> dict[str, Entry]:
    """The catalogue for ``tag``, read once per process."""
    if tag not in TAGS:
        raise ValueError(f"not an interface language: {tag}")
    return parse(tag, locale_file(tag).read_text(encoding="utf-8"))


def parse(tag: str, text: str) -> dict[str, Entry]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{tag}: a catalogue is one JSON object")
    entries: dict[str, Entry] = {}
    for key, value in data.items():
        if isinstance(value, str):
            entries[key] = value
        elif isinstance(value, dict) and all(isinstance(form, str) for form in value.values()):
            entries[key] = dict(value)
        else:
            raise ValueError(f"{tag}: {key} is neither a sentence nor a set of plural forms")
    return entries
