"""Settings-file shorthand for tests: the real load boundary, and the header
every settings file carries."""

import json
from pathlib import Path

from pixelup import __version__
from pixelup.app_config import AppConfig, load_app_config_result


def load_app_config(root: Path | None = None) -> AppConfig:
    """Use the real load boundary, which refuses an unreadable or newer file."""
    return load_app_config_result(root).config


def header(kind: str, version: int = 1) -> dict[str, object]:
    """The header PixelUp writes on a file of ``kind`` (file-identity-conventions)."""
    return {
        "format": {"owner": "pixelup", "kind": kind, "version": version},
        "writer": {"name": "pixelup", "version": __version__},
    }


def stored_content(path: Path) -> dict[str, object]:
    """A saved settings file's content, after checking it carries its own header
    (the kind is the file's stem: ``config`` or ``parameters``)."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert {key: stored.pop(key) for key in ("format", "writer")} == header(path.stem)
    return stored
