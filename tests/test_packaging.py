from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RUNTIME_PINS = {
    "PySide6": "6.11.2",
    "Pillow": "12.3.0",
    "pillow-heif": "1.5.0",
    "filelock": "3.32.3",
    "ncnn": "1.0.20260526",
    "numpy": "2.5.2",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "opencv-python": "5.0.0.93",
    "realesrgan": "0.3.0",
    "basicsr-fixed": "1.4.2",
    "gfpgan": "1.3.8",
}


def test_runtime_dependencies_pin_inference_stack() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert dependencies == [
        f"{package}=={version}"
        for package, version in EXPECTED_RUNTIME_PINS.items()
    ]


def test_inference_extra_is_not_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"].get("optional-dependencies", {})

    assert "inference" not in extras


def test_script_starts_gui() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["gui-scripts"]["pixelup"] == "pixelup.gui:main"


def test_source_python_range_matches_the_supported_dependency_stack() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"


def test_windows_installer_configuration() -> None:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw_line in (ROOT / "scripts" / "pixelup.iss").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1].casefold(), [])
        elif current is not None and line and not line.startswith(";"):
            current.append(line)

    setup = dict(line.split("=", 1) for line in sections["setup"])
    run = dict(
        field.strip().split(":", 1)
        for field in sections["run"][0].split(";")
        if ":" in field
    )
    flags = set(run["Flags"].split())

    assert setup["AppId"] == "{#MyAppName}"
    assert setup["DefaultDirName"] == "{autopf}\\{#MyAppName}"
    assert setup["PrivilegesRequiredOverridesAllowed"] == "dialog"
    assert "PrivilegesRequired" not in setup
    assert setup["Uninstallable"] == "yes"
    icon = ROOT / setup["SetupIconFile"].replace("\\", "/")
    assert icon.is_file()
    assert "runasoriginaluser" in flags
    assert "runascurrentuser" not in flags
    assert run["Check"].strip() == "not IsAdminInstallMode"
