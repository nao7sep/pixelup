from __future__ import annotations

import tomllib
from pathlib import Path

EXPECTED_RUNTIME_PINS = {
    "PySide6": "6.11.2",
    "Pillow": "12.3.0",
    "pillow-heif": "1.5.0",
    "filelock": "3.32.3",
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


def test_spec_derives_bundle_version_from_pyproject_ssot() -> None:
    # The frozen macOS .app must report its real version, not PyInstaller's 0.0.0
    # default, and the version must come from the pyproject.toml SSOT rather than a
    # literal in the spec (which would silently drift on the next bump). PyInstaller
    # is not installed in the test env, so this pins the wiring by source inspection:
    # the spec reads pyproject.toml into _VERSION and feeds it to both bundle keys.
    spec = Path("pixelup.spec").read_text(encoding="utf-8")

    # Version is read from the SSOT at freeze time, not hardcoded.
    assert 'tomllib.loads(' in spec
    assert '["project"]["version"]' in spec
    # Both macOS version keys are present and set from that derived _VERSION, never a literal.
    assert '"CFBundleShortVersionString": _VERSION,' in spec
    assert '"CFBundleVersion": _VERSION,' in spec

    # And there is no hardcoded copy of the current version anywhere in the spec.
    version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert version not in spec


def test_macos_bundle_finalization_is_shared_by_package_and_rebuild() -> None:
    package = Path("scripts/package.sh").read_text(encoding="utf-8")
    rebuild = Path("scripts/rebuild.command").read_text(encoding="utf-8")
    finalizer = Path("scripts/finalize-macos-bundle.sh").read_text(encoding="utf-8")

    invocation = '"$SCRIPT_DIR/finalize-macos-bundle.sh"'
    assert invocation in package
    assert invocation in rebuild
    assert 'CAR="$REPO_DIR/build/Assets.car"' in finalizer
    assert 'cp "$CAR" "$APP_BUNDLE/Contents/Resources/Assets.car"' in finalizer
    assert 'codesign --force --deep --sign - "$APP_BUNDLE"' in finalizer


def test_frozen_runtime_resources_include_only_the_windows_icon() -> None:
    spec = Path("pixelup.spec").read_text(encoding="utf-8")

    assert '("src/pixelup/resources/icon-win.png", "pixelup/resources")' in spec
    assert '("src/pixelup/resources/icon.png", "pixelup/resources")' not in spec


def test_frozen_runtime_includes_the_dynamic_https_hostname_codec() -> None:
    spec = Path("pixelup.spec").read_text(encoding="utf-8")

    assert '"encodings.idna",' in spec


def test_windows_installer_embeds_the_canonical_app_icon() -> None:
    installer = Path("scripts/pixelup.iss").read_text(encoding="utf-8")

    assert "SetupIconFile=build\\icon.ico" in installer
