from __future__ import annotations

import re

# text-cleanup-conventions
_BREAK_RUN = re.compile(r"\s*[\r\n]+\s*")


def single_line(text: str) -> str:
    """The single-line pattern with its defaults: line breaks flattened, ends trimmed."""
    return _BREAK_RUN.sub(" ", text).strip()
