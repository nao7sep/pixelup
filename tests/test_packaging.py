from __future__ import annotations

import tomllib
from pathlib import Path

EXPECTED_RUNTIME_PINS = {
    "PySide6": "6.11.0",
    "Pillow": "12.2.0",
    "pillow-heif": "1.3.0",
    "filelock": "3.29.0",
    "numpy": "2.4.4",
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "opencv-python": "4.13.0.92",
    "realesrgan": "0.3.0",
    "basicsr-fixed": "1.4.2",
    "gfpgan": "1.3.8",
}


def test_runtime_dependencies_pin_inference_stack() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert dependencies == [
        f"{package}=={version}"
        for package, version in EXPECTED_RUNTIME_PINS.items()
    ]


def test_inference_extra_is_not_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"].get("optional-dependencies", {})

    assert "inference" not in extras


def test_script_starts_gui() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["gui-scripts"]["pixelup"] == "pixelup.gui:main"
