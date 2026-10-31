from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.formats import SIDECAR_FORMAT, json_text
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.imaging import _temp_output_path, temp_file_guard
from pixelup.output_publication import publish_complete
from pixelup.output_reservation import remove_published_file
from pixelup.timestamps import utc_now_iso_ms
from pixelup.upscale import UpscaleOptions


def write_sidecar(
    *,
    input_path: Path,
    output_path: Path,
    options: UpscaleOptions,
    result: dict[str, object],
    warnings: list[Message],
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    sidecar_path = output_path.with_suffix(".json")
    content = {
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
            "output_format": options.output_format.value,
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
    stage_path = _temp_output_path(sidecar_path)
    claim = None
    try:
        with temp_file_guard(stage_path) as (file, final_mode):
            file.write(json_text(SIDECAR_FORMAT, content).encode("utf-8"))
            file.flush()
            os.fsync(file.fileno())
            if os.name != "nt":
                permissions = int(result.get("source_read_permissions", 0o444))
                os.fchmod(file.fileno(), final_mode & (permissions | 0o222))
            if should_cancel is not None and should_cancel():
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
            claim = publish_complete(stage_path, sidecar_path)
    except FileExistsError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS, Message("error.sidecarExists"),
            hint=Message("error.hintRetryNewName"), details={"sidecar": str(sidecar_path)},
        ) from exc
    except OSError as exc:
        if claim is not None:
            remove_published_file(claim)
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE, Message("error.sidecarWriteFailed"),
            details={"sidecar": str(sidecar_path), "reason": str(exc)},
        ) from exc
    if should_cancel is not None and should_cancel():
        remove_published_file(claim)
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
    return claim
