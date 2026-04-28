from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType

from pixelup.inference import (
    _INFERENCE_DEPS_HINT,
    _install_torchvision_functional_tensor_fallback,
)


def test_inference_extra_pins_heavy_runtime_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    inference = pyproject["project"]["optional-dependencies"]["inference"]

    assert inference == [
        "numpy==2.4.4",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "opencv-python==4.13.0.92",
        "realesrgan==0.3.0",
        "basicsr-fixed==1.4.2",
        "gfpgan==1.3.8",
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
