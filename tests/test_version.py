"""The app's version comes from the one place it is declared."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pixelup


def test_version_matches_the_one_place_it_is_declared() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert pixelup.__version__ == declared
