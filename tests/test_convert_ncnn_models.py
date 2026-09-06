from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_ncnn_models.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("convert_ncnn_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_conversion_specs_cover_every_upscaler_and_denoise_companion() -> None:
    module = _load_script()
    from pixelup.model_management import GENERAL_DENOISE_MODEL, UPSCALE_MODELS

    names = tuple(spec.name for spec in module.CONVERSION_SPECS)

    assert names == (*UPSCALE_MODELS, GENERAL_DENOISE_MODEL)
    assert len(names) == len(set(names))


def test_artifact_record_uses_file_bytes(tmp_path: Path) -> None:
    module = _load_script()
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"pixelup-ncnn")

    assert module._artifact_record(artifact) == {
        "filename": "model.bin",
        "size": 12,
        "sha256": "9307e02a1ec617d0ddb909b2cdcd613b45b53b91bb2814b42e7bbf0a215a2a72",
    }


def test_conversion_equivalence_rejects_shape_and_value_regressions() -> None:
    module = _load_script()
    expected = np.zeros((3, 8, 8), dtype=np.float32)

    with pytest.raises(RuntimeError, match="converted shape"):
        module._assert_equivalent(
            "model",
            expected,
            np.zeros((3, 4, 4), dtype=np.float32),
        )
    with pytest.raises(RuntimeError, match="exceeds FP16 tolerance"):
        module._assert_equivalent(
            "model",
            expected,
            np.ones((3, 8, 8), dtype=np.float32),
        )

    module._assert_equivalent(
        "model",
        expected,
        np.full((3, 8, 8), 0.01, dtype=np.float32),
    )
