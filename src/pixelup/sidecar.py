from __future__ import annotations

import json
import os
from pathlib import Path

from pixelup import __version__
from pixelup.errors import ErrorCode, PixelupError
from pixelup.formats import SIDECAR_FORMAT_VERSION
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.output_reservation import (
    PublishedFile,
    close_published_file,
    published_file_is_current,
    remove_published_file,
)
from pixelup.timestamps import utc_now_iso_ms
from pixelup.upscale import UpscaleOptions


def write_sidecar(
    *,
    input_path: Path,
    output_path: Path,
    options: UpscaleOptions,
    result: dict[str, object],
    warnings: list[Message],
) -> PublishedFile:
    sidecar_path = output_path.with_suffix(".json")
    payload = {
        "format_version": SIDECAR_FORMAT_VERSION,
        "app": {
            "name": "pixelup",
            "version": __version__,
        },
        "created_at_utc": utc_now_iso_ms(),
        "status": "success",
        "input": {
            "filename": input_path.name,
            "sha256": result.get("input_sha256"),
            "size_bytes": result.get("input_size_bytes"),
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
        # The file records what happened, in English whatever the interface
        # speaks: a stored value keeps its stored form (localization-conventions).
        "warnings": [english().of(warning) for warning in warnings],
        "duration_ms": result.get("ms"),
    }
    # not recorded: this sidecar is OUTPUT metadata written beside the output image
    # at a user-chosen location, colocated with the (binary) output the app harvests
    # then forgets — not managed text the app owns and reloads as state. Output is
    # never recorded, and a sidecar beside a not-recorded output rides along into
    # exclusion (data-backup-conventions). It is regenerable from the run and would
    # bloat the text history with no recovery value.
    try:
        # Exclusive creation is the last no-clobber gate after a potentially long
        # inference. The output reservation serializes PixelUp peers; O_EXCL also
        # protects a sidecar an external process placed in the meantime.
        descriptor = os.open(sidecar_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS,
            Message("error.sidecarExists"),
            hint=Message("error.hintRetryNewName"),
            details={"sidecar": str(sidecar_path)},
        ) from exc
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.sidecarWriteFailed"),
            details={"sidecar": str(sidecar_path), "reason": str(exc)},
        ) from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as file:
            file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        remove_published_file(close_published_file(sidecar_path, descriptor))
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.sidecarWriteFailed"),
            details={"sidecar": str(sidecar_path), "reason": str(exc)},
        ) from exc
    except Exception:
        remove_published_file(close_published_file(sidecar_path, descriptor))
        raise
    claim = close_published_file(sidecar_path, descriptor)
    if not published_file_is_current(claim):
        remove_published_file(claim)
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS,
            Message("error.sidecarChanged"),
            hint=Message("error.hintRetryNewName"),
            details={"sidecar": str(sidecar_path)},
        )
    return claim
