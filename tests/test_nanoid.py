from __future__ import annotations

import re

from pixelup.nanoid import ALPHABET, DEFAULT_SIZE, nanoid

CANONICAL = re.compile(rf"[{re.escape(ALPHABET)}]{{{DEFAULT_SIZE}}}")


def test_nanoid_default_length_is_21() -> None:
    assert len(nanoid()) == 21


def test_nanoid_respects_explicit_size() -> None:
    assert len(nanoid(10)) == 10


def test_nanoid_uses_only_the_url_safe_alphabet() -> None:
    assert CANONICAL.fullmatch(nanoid())


def test_nanoid_two_calls_differ() -> None:
    # Astronomically unlikely to collide; a match here would indicate a broken
    # generator (e.g. a fixed seed) rather than genuine bad luck.
    assert nanoid() != nanoid()
