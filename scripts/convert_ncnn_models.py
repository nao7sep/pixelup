#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup.model_registry import known_model
from pixelup.models import verify_model_file


@dataclass(frozen=True, slots=True)
class ConversionSpec:
    name: str
    architecture: str
    scale: int
    depth: int


CONVERSION_SPECS = (
    ConversionSpec("realesr-general-x4v3", "srvgg", 4, 32),
    ConversionSpec("RealESRGAN_x4plus", "rrdb", 4, 23),
    ConversionSpec("RealESRNet_x4plus", "rrdb", 4, 23),
    ConversionSpec("RealESRGAN_x2plus", "rrdb", 2, 23),
    ConversionSpec("RealESRGAN_x4plus_anime_6B", "rrdb", 4, 6),
    ConversionSpec("realesr-animevideov3", "srvgg", 4, 16),
    ConversionSpec("realesr-general-wdn-x4v3", "srvgg", 4, 32),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PixelUp's pinned PyTorch models into ncnn model pairs."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing the pinned upstream .pth files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive .ncnn.param/.ncnn.bin files and a manifest.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(spec.name for spec in CONVERSION_SPECS),
        help="Optional subset of model names; the default converts every model.",
    )
    return parser.parse_args(argv)


def _model_factory(spec: ConversionSpec) -> Callable[[], Any]:
    if spec.architecture == "rrdb":
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from torch.nn import functional as torch_functional

        class ConvertibleRRDBNet(RRDBNet):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.scale == 2:
                    weights = torch.zeros(12, 3, 2, 2)
                    for channel in range(3):
                        for row in range(2):
                            for column in range(2):
                                weights[channel * 4 + row * 2 + column, channel, row, column] = 1
                    self.register_buffer(
                        "_pixel_unshuffle_weights",
                        weights,
                        persistent=False,
                    )

            def forward(self, value):
                if self.scale == 2:
                    # BasicSR's generic pixel_unshuffle traces as a five-dimensional
                    # permute, which ncnn cannot represent. A fixed one-hot 2x2
                    # strided convolution is exactly the same channel rearrangement
                    # and lowers to an ordinary ncnn Convolution layer.
                    value = torch_functional.conv2d(
                        value,
                        self._pixel_unshuffle_weights,
                        stride=2,
                    )
                feature = self.conv_first(value)
                body_feature = self.conv_body(self.body(feature))
                feature = feature + body_feature
                feature = self.lrelu(
                    self.conv_up1(
                        torch_functional.interpolate(feature, scale_factor=2, mode="nearest")
                    )
                )
                feature = self.lrelu(
                    self.conv_up2(
                        torch_functional.interpolate(feature, scale_factor=2, mode="nearest")
                    )
                )
                return self.conv_last(self.lrelu(self.conv_hr(feature)))

        return lambda: ConvertibleRRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=spec.depth,
            num_grow_ch=32,
            scale=spec.scale,
        )
    if spec.architecture == "srvgg":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        return lambda: SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=spec.depth,
            upscale=spec.scale,
            act_type="prelu",
        )
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, int | str]:
    return {
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_path(source_dir: Path, spec: ConversionSpec) -> Path:
    info = known_model(spec.name)
    if info is None:
        raise ValueError(f"No pinned source model is registered for {spec.name}")
    path = source_dir / info.filename
    verify_model_file(path, info)
    return path


def _convert_one(spec: ConversionSpec, source: Path, staging_dir: Path) -> tuple[Path, Path]:
    import pnnx
    import torch

    model = _model_factory(spec)()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()

    # Two distinct shapes make pnnx retain dynamic height and width. Both are
    # divisible by two so the x2 RRDB model's pixel-unshuffle prelude is valid.
    torch.manual_seed(0)
    inputs = (torch.rand(1, 3, 8, 8),)
    inputs2 = (torch.rand(1, 3, 12, 16),)
    build_stem = spec.name.replace("-", "_")
    prefix = staging_dir / build_stem
    pnnx.export(
        model,
        str(prefix.with_suffix(".pt")),
        inputs,
        inputs2,
        fp16=True,
        pnnxparam=str(prefix.with_suffix(".pnnx.param")),
        pnnxbin=str(prefix.with_suffix(".pnnx.bin")),
        pnnxpy=str(staging_dir / f"{build_stem}_pnnx.py"),
        pnnxonnx=str(prefix.with_suffix(".pnnx.onnx")),
        ncnnparam=str(prefix.with_suffix(".ncnn.param")),
        ncnnbin=str(prefix.with_suffix(".ncnn.bin")),
        ncnnpy=str(staging_dir / f"{build_stem}_ncnn.py"),
    )
    return prefix.with_suffix(".ncnn.param"), prefix.with_suffix(".ncnn.bin")


def convert_models(
    source_dir: Path,
    output_dir: Path,
    specs: Sequence[ConversionSpec] = CONVERSION_SPECS,
) -> dict[str, object]:
    import pnnx

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="pixelup-ncnn-models-", dir=output_dir) as temp:
        staging_dir = Path(temp)
        for spec in specs:
            source = _source_path(source_dir, spec)
            staged_param, staged_bin = _convert_one(spec, source, staging_dir)
            final_param = output_dir / f"{spec.name}.ncnn.param"
            final_bin = output_dir / f"{spec.name}.ncnn.bin"
            os.replace(staged_param, final_param)
            os.replace(staged_bin, final_bin)
            records.append(
                {
                    "name": spec.name,
                    "source": _artifact_record(source),
                    "param": _artifact_record(final_param),
                    "bin": _artifact_record(final_bin),
                }
            )

    manifest: dict[str, object] = {
        "format": 1,
        "converter": {"name": "pnnx", "version": str(pnnx.__version__), "fp16": True},
        "models": records,
    }
    manifest_path = output_dir / "ncnn-models.json"
    staged_manifest = output_dir / ".ncnn-models.json.tmp"
    staged_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staged_manifest, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = set(args.models)
    specs = tuple(spec for spec in CONVERSION_SPECS if not selected or spec.name in selected)
    convert_models(args.source_dir, args.output_dir, specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
