"""The app's version, and where it comes from.

The frozen app and an editable install answer from distribution metadata; a run
straight from the source tree has none, and used to call itself "unknown" — in
About, in the session log, and in the sidecar beside every output.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pixelup


def _declared_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])


def test_version_matches_the_one_place_it_is_declared() -> None:
    assert pixelup.__version__ == _declared_version()


def test_a_source_tree_run_reads_pyproject_rather_than_giving_up() -> None:
    assert pixelup._version_from_source_tree() == _declared_version()


def test_the_fallback_reports_absence_instead_of_raising(monkeypatch, tmp_path) -> None:
    # Point the package somewhere with no pyproject.toml above it.
    monkeypatch.setattr(pixelup, "__file__", str(tmp_path / "pixelup" / "__init__.py"))
    assert pixelup._version_from_source_tree() is None
