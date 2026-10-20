from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

from pixelup.i18n.languages import TAGS

ROOT = Path(__file__).resolve().parents[1]


def _installer_script() -> str:
    return (ROOT / "scripts" / "pixelup.iss").read_text(encoding="utf-8")


def test_the_gui_script_names_a_callable_entry_point() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module_name, attribute = pyproject["project"]["gui-scripts"]["pixelup"].split(":")

    assert callable(getattr(importlib.import_module(module_name), attribute))


def test_the_windows_installer_icon_exists() -> None:
    icon = re.search(r"^SetupIconFile=(.+)$", _installer_script(), flags=re.M)

    assert icon is not None
    assert (ROOT / icon.group(1).strip().replace("\\", "/")).is_file()


def test_the_windows_installer_speaks_only_interface_languages_with_built_in_messages() -> None:
    # Inno's own translations, English first as the fallback; no .isl file is
    # vendored, so an interface language Inno does not ship gets the English wizard.
    languages = re.findall(
        r'^Name: "([^"]+)"; MessagesFile: "([^"]+)"', _installer_script(), flags=re.M
    )
    tags = {tag.replace("-", "").casefold(): tag for tag in TAGS}

    assert languages[0] == ("en", "compiler:Default.isl")
    assert all(name in tags for name, _file in languages)
    assert all(file.startswith("compiler:") for _name, file in languages)
    assert not list((ROOT / "scripts").rglob("*.isl"))


def test_the_windows_installer_writes_only_innos_translated_messages() -> None:
    descriptions = re.findall(r'(?:Description|GroupDescription): "([^"]*)"', _installer_script())

    assert descriptions and all(text.startswith("{cm:") for text in descriptions)
