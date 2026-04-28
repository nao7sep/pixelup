from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType

from pixelup import inference as inference_module
from pixelup.inference import (
    _INFERENCE_DEPS_HINT,
    PINNED_INFERENCE_REQUIREMENTS,
    _install_torchvision_functional_tensor_fallback,
    inference_dependency_status,
)


def test_inference_extra_pins_heavy_runtime_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    inference = pyproject["project"]["optional-dependencies"]["inference"]

    assert inference == [
        f"{package}=={version}"
        for package, version in PINNED_INFERENCE_REQUIREMENTS.items()
    ]


def test_missing_dependency_hint_points_to_inference_extra() -> None:
    assert ".[inference]" in _INFERENCE_DEPS_HINT
    assert "uv sync --extra inference" in _INFERENCE_DEPS_HINT


def test_torchvision_functional_tensor_fallback(monkeypatch) -> None:
    torchvision = ModuleType("torchvision")
    transforms = ModuleType("torchvision.transforms")
    functional = ModuleType("torchvision.transforms.functional")
    transforms.functional = functional

    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", functional)
    monkeypatch.delitem(sys.modules, "torchvision.transforms.functional_tensor", raising=False)

    _install_torchvision_functional_tensor_fallback()

    assert sys.modules["torchvision.transforms.functional_tensor"] is functional


def test_inference_dependency_status_reports_missing_and_mismatch(monkeypatch) -> None:
    versions = {
        "numpy": "2.4.4",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "opencv-python": "4.13.0.92",
        "realesrgan": "0.3.0",
        "basicsr-fixed": "1.4.1",
    }

    def fake_version(package: str) -> str:
        if package not in versions:
            raise inference_module.importlib_metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(inference_module.importlib_metadata, "version", fake_version)

    statuses = {
        status.package: status
        for status in inference_dependency_status(include_face_enhance=True)
    }

    assert statuses["numpy"].ok is True
    assert statuses["basicsr-fixed"].ok is False
    assert statuses["basicsr-fixed"].reason == "version_mismatch"
    assert statuses["gfpgan"].ok is False
    assert statuses["gfpgan"].reason == "missing"
