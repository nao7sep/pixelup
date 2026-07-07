from __future__ import annotations

import secrets

# The standard nanoid URL-safe alphabet: 64 symbols, so a byte's low 6 bits
# (``b & 63``) index it with zero modulo bias — 64 divides 256 evenly, unlike
# uuid4's 16-symbol hex alphabet which needed no such care but also carried
# hyphens and version/variant nibbles pixelup never used.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
DEFAULT_SIZE = 21


def nanoid(size: int = DEFAULT_SIZE) -> str:
    """Generate a URL- and filename-safe, crypto-random id.

    Dependency-free stand-in for the JS/other-language ``nanoid`` package:
    same 64-char alphabet, same default length (21 chars, comparable
    collision resistance to a uuid4). Each byte from ``secrets.token_bytes``
    is masked to its low 6 bits to index the alphabet, which is perfectly
    uniform since 64 divides 256 evenly (no rejection sampling needed).
    """
    return "".join(ALPHABET[b & 63] for b in secrets.token_bytes(size))
