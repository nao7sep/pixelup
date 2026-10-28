from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_gui_script_names_a_callable_entry_point() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module_name, attribute = pyproject["project"]["gui-scripts"]["pixelup"].split(":")

    assert callable(getattr(importlib.import_module(module_name), attribute))
