from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RUNTIME_PINS = {
    "PySide6": "6.11.2",
    "Pillow": "12.3.0",
    "pillow-heif": "1.5.0",
    "filelock": "3.32.3",
    "numpy": "2.5.2",
    "torch": "2.13.0",
    "opencv-python": "5.0.0.93",
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


def test_application_license_is_packaged_on_both_platforms() -> None:
    mac_finalizer = (ROOT / "scripts" / "finalize-macos-bundle.sh").read_text(
        encoding="utf-8"
    )
    windows_packager = (ROOT / "scripts" / "package.ps1").read_text(encoding="utf-8")

    assert 'cp "$LICENSE" "$APP_BUNDLE/Contents/Resources/LICENSE.txt"' in mac_finalizer
    assert (
        "Copy-Item -LiteralPath LICENSE -Destination dist\\PixelUp\\LICENSE.txt"
        in windows_packager
    )


def test_consolidated_third_party_notices_are_built_and_packaged() -> None:
    source_notices = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    mac_packager = (ROOT / "scripts" / "package.sh").read_text(encoding="utf-8")
    mac_finalizer = (ROOT / "scripts" / "finalize-macos-bundle.sh").read_text(
        encoding="utf-8"
    )
    windows_packager = (ROOT / "scripts" / "package.ps1").read_text(encoding="utf-8")

    assert "Copyright 2018-2022 BasicSR Authors" in source_notices
    assert "Copyright (c) 2021, Xintao Wang" in source_notices
    assert "build-third-party-notices.py" in mac_packager
    assert 'cp "$NOTICES" "$APP_BUNDLE/Contents/Resources/THIRD_PARTY_NOTICES.txt"' in (
        mac_finalizer
    )
    assert "build-third-party-notices.py" in windows_packager
    assert "dist\\PixelUp\\THIRD_PARTY_NOTICES.txt" in windows_packager


def test_package_scripts_reject_editable_install_metadata() -> None:
    mac_packager = (ROOT / "scripts" / "package.sh").read_text(encoding="utf-8")
    windows_packager = (ROOT / "scripts" / "package.ps1").read_text(encoding="utf-8")

    assert '-name direct_url.json' in mac_packager
    assert '-Filter direct_url.json' in windows_packager
