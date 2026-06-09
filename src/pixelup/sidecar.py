from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pixelup import __version__
from pixelup.upscale import UpscaleOptions

SCHEMA_VERSION = 1


def write_sidecar(
    *,
    input_path: Path,
    output_path: Path,
    options: UpscaleOptions,
    result: dict[str, object],
    warnings: list[str],
) -> Path:
    sidecar_path = output_path.with_suffix(".json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "app": {
            "name": "pixelup",
            "version": __version__,
        },
        "created_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "status": "success",
        "input": {
            "filename": input_path.name,
            "sha256": _sha256(input_path),
            "size_bytes": input_path.stat().st_size,
            "dimensions": result.get("input_size"),
        },
        "output": {
            "filename": output_path.name,
            "format": result.get("format"),
            "dimensions": result.get("output_size"),
        },
        "model": result.get("model"),
        "scale": result.get("scale"),
        "options": {
            "tile": options.tile,
            "tile_pad": options.tile_pad,
            "pre_pad": options.pre_pad,
            "fp32": options.fp32,
            "face_enhance": options.face_enhance,
            "denoise_strength": options.denoise_strength,
            "alpha_mode": options.alpha_mode,
            "device": options.device,
            "gpu_id": options.gpu_id,
            "output_format": options.output_format.value if options.output_format else None,
            "quality": options.quality,
            "background": options.background,
            "strip_metadata": options.strip_metadata,
            "target_profile": options.target_profile,
        },
        "warnings": warnings,
        "duration_ms": result.get("ms"),
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
