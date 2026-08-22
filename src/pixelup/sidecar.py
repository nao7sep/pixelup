from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pixelup import __version__
from pixelup.errors import ErrorCode, PixelupError
from pixelup.timestamps import utc_now_iso_ms
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
        "created_at_utc": utc_now_iso_ms(),
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
    # not recorded: this sidecar is OUTPUT metadata written beside the output image
    # at a user-chosen location, colocated with the (binary) output the app harvests
    # then forgets — not managed text the app owns and reloads as state. Output is
    # never recorded, and a sidecar beside a not-recorded output rides along into
    # exclusion (data-backup-conventions). It is regenerable from the run and would
    # bloat the text history with no recovery value.
    created = False
    try:
        # Exclusive creation is the last no-clobber gate after a potentially long
        # inference. The output reservation serializes PixelUp peers; mode="x"
        # also protects a sidecar an external process placed in the meantime.
        with sidecar_path.open("x", encoding="utf-8") as file:
            created = True
            file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS,
            "Output settings sidecar already exists.",
            hint="Retry the job to choose a new unused filename.",
            details={"sidecar": str(sidecar_path)},
        ) from exc
    except OSError as exc:
        if created:
            sidecar_path.unlink(missing_ok=True)
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not write the output settings sidecar.",
            details={"sidecar": str(sidecar_path), "reason": str(exc)},
        ) from exc
    return sidecar_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
