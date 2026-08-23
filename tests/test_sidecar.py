import json
import os
import re
from pathlib import Path

import pytest
from PIL import Image

from pixelup import sidecar as sidecar_module
from pixelup.errors import PixelupError
from pixelup.paths import OutputFormat
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions


def _write_sample_sidecar(tmp_path: Path) -> tuple[Path, dict]:
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "source-realesr-general-x4v3-4x.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    output_path.write_bytes(b"image")

    published = write_sidecar(
        input_path=input_path,
        output_path=output_path,
        options=UpscaleOptions(
            input_path=input_path,
            output_arg=str(output_path),
            model="realesr-general-x4v3",
            scale=4,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            fp32=False,
            face_enhance=False,
            denoise_strength=1.0,
            alpha_mode="realesrgan",
            gpu_id=None,
            device="auto",
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            strip_metadata=False,
            target_profile=None,
            overwrite=False,
            auto_download=True,
            download_timeout=600,
            lock_timeout=600,
        ),
        result={
            "input_size": [1, 1],
            "output_size": [4, 4],
            "format": "png",
            "model": "realesr-general-x4v3",
            "scale": 4,
            "ms": 10,
        },
        warnings=[],
    )
    return published.path, json.loads(published.path.read_text())


def test_sidecar_omits_private_paths(tmp_path: Path) -> None:
    path, payload = _write_sample_sidecar(tmp_path)
    serialized = json.dumps(payload)
    assert path.name == "source-realesr-general-x4v3-4x.json"
    assert payload["input"]["filename"] == "source.png"
    assert payload["output"]["filename"] == "source-realesr-general-x4v3-4x.png"
    assert str(tmp_path) not in serialized


def test_sidecar_created_at_is_canonical_utc(tmp_path: Path) -> None:
    _, payload = _write_sample_sidecar(tmp_path)
    # Internal/serialized timestamps: ISO-8601, exactly 3 fractional digits, Z suffix.
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        payload["created_at_utc"],
    )


def test_sidecar_does_not_replace_an_existing_file(tmp_path: Path) -> None:
    sidecar, _payload = _write_sample_sidecar(tmp_path)
    original = sidecar.read_bytes()

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_exists"
    assert sidecar.read_bytes() == original


def test_sidecar_failure_cleanup_preserves_an_exact_boundary_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "source-realesr-general-x4v3-4x.json"
    real_remove = sidecar_module.remove_published_file

    def fail_flush(_descriptor: int) -> None:
        raise OSError("flush failed")

    def replace_then_cleanup(claim) -> bool:
        winner = tmp_path / "winner.tmp"
        winner.write_bytes(b"external winner")
        os.replace(winner, sidecar)
        return real_remove(claim)

    monkeypatch.setattr("pixelup.sidecar.os.fsync", fail_flush)
    monkeypatch.setattr("pixelup.sidecar.remove_published_file", replace_then_cleanup)

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_unwritable"
    assert sidecar.read_bytes() == b"external winner"


def test_sidecar_success_verification_rejects_an_exact_boundary_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "source-realesr-general-x4v3-4x.json"
    real_is_current = sidecar_module.published_file_is_current

    def replace_before_verification(claim) -> bool:
        winner = tmp_path / "winner.tmp"
        winner.write_bytes(b"external winner")
        os.replace(winner, sidecar)
        return real_is_current(claim)

    monkeypatch.setattr(
        "pixelup.sidecar.published_file_is_current", replace_before_verification
    )

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_exists"
    assert sidecar.read_bytes() == b"external winner"
