import json
from pathlib import Path

from PIL import Image

from pixelup.paths import OutputFormat
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions


def test_sidecar_omits_private_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "source-realesr-general-x4v3-4x.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    output_path.write_bytes(b"image")

    path = write_sidecar(
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

    payload = json.loads(path.read_text())
    serialized = json.dumps(payload)
    assert path.name == "source-realesr-general-x4v3-4x.json"
    assert payload["input"]["filename"] == "source.png"
    assert payload["output"]["filename"] == "source-realesr-general-x4v3-4x.png"
    assert str(tmp_path) not in serialized
