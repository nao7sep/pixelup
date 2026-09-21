"""PixelUp package."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_source_tree() -> str | None:
    """The version ``pyproject.toml`` declares, for a run from the source tree.

    ``importlib.metadata`` answers from installed distribution metadata, which the
    frozen app and an editable install both have and a bare ``python -m pixelup`` on
    ``PYTHONPATH`` does not. ``pyproject.toml`` is the one place the version is
    declared — the PyInstaller spec reads it too — so a source run reads it rather
    than calling itself "unknown", which reached the About dialog, the session log
    and the sidecar written beside every output.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


try:
    __version__ = version("pixelup")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = _version_from_source_tree() or "unknown"
