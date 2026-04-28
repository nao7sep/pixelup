from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageCms, ImageColor, UnidentifiedImageError

from pixelup.errors import ErrorCode, PixelupError
from pixelup.paths import OutputFormat


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    icc_profile: bytes | None = None
    exif: bytes | None = None


def register_image_plugins() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()


def read_image_size(path: Path) -> tuple[int, int]:
    register_image_plugins()
    try:
        with Image.open(path) as image:
            image.verify()
            return image.size
    except UnidentifiedImageError as exc:
        raise PixelupError(
            ErrorCode.INPUT_INVALID_FORMAT,
            "Input is not a readable image format.",
            details={"input": str(path)},
        ) from exc
    except PermissionError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            "Input image is not readable.",
            details={"input": str(path), "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            "Input image could not be opened.",
            details={"input": str(path), "reason": str(exc)},
        ) from exc


def image_from_bgr_array(array: Any) -> Image.Image:
    ndim = getattr(array, "ndim", None)
    if ndim == 2:
        return Image.fromarray(array)
    if ndim != 3:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Inference returned an unsupported image array.",
            details={"ndim": ndim},
        )
    channels = array.shape[2]
    if channels == 3:
        return Image.fromarray(array[:, :, ::-1].copy())
    if channels == 4:
        return Image.fromarray(array[:, :, [2, 1, 0, 3]])
    raise PixelupError(
        ErrorCode.INTERNAL_ERROR,
        "Inference returned an unsupported channel layout.",
        details={"channels": channels},
    )


def load_source_metadata(path: Path) -> SourceMetadata:
    register_image_plugins()
    try:
        with Image.open(path) as image:
            icc_profile = image.info.get("icc_profile")
            exif = image.info.get("exif")
    except (OSError, UnidentifiedImageError):
        return SourceMetadata()
    return SourceMetadata(
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
        exif=exif if isinstance(exif, bytes) else None,
    )


def save_output_image(
    image: Image.Image,
    *,
    output_path: Path,
    output_format: OutputFormat,
    quality: int,
    background: str,
    temp_dir: Path,
    source_metadata: SourceMetadata | None,
    strip_metadata: bool,
    target_profile: str | None,
) -> tuple[int, int]:
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not create the temp directory.",
            details={"path": str(temp_dir), "reason": str(exc)},
        ) from exc
    if target_profile not in {None, "srgb"}:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Display-P3 and Adobe RGB output profiles are not implemented in this phase.",
            hint="Use --target-profile srgb or omit --target-profile.",
            details={"target_profile": target_profile},
        )

    encoded = _prepare_image_for_save(
        image,
        output_format=output_format,
        background=background,
        source_metadata=source_metadata,
        strip_metadata=strip_metadata,
        target_profile=target_profile,
    )
    save_kwargs = _save_kwargs(
        output_format,
        quality=quality,
        source_metadata=source_metadata,
        strip_metadata=strip_metadata,
        target_profile=target_profile,
    )
    temp_path = _temp_output_path(temp_dir, output_format)
    try:
        encoded.save(temp_path, **save_kwargs)
        os.replace(temp_path, output_path)
    except PixelupError:
        temp_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not write the output image.",
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc
    return encoded.size


def _prepare_image_for_save(
    image: Image.Image,
    *,
    output_format: OutputFormat,
    background: str,
    source_metadata: SourceMetadata | None,
    strip_metadata: bool,
    target_profile: str | None,
) -> Image.Image:
    prepared = image
    if output_format == OutputFormat.JPG:
        prepared = _flatten_alpha(prepared, background)
    elif prepared.mode not in {"RGB", "RGBA"}:
        prepared = prepared.convert("RGBA" if "A" in prepared.getbands() else "RGB")
    needs_srgb = strip_metadata or target_profile == "srgb"
    if needs_srgb and source_metadata and source_metadata.icc_profile:
        prepared = _convert_to_srgb(prepared, source_metadata.icc_profile)
    return prepared


def _flatten_alpha(image: Image.Image, background: str) -> Image.Image:
    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return image.convert("RGB")
    try:
        background_rgb = ImageColor.getrgb(background)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "--background is not a valid Pillow color.",
            details={"background": background},
        ) from exc
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, background_rgb + (255,))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _convert_to_srgb(image: Image.Image, icc_profile: bytes) -> Image.Image:
    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        target_profile = ImageCms.createProfile("sRGB")
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        return ImageCms.profileToProfile(image.convert(mode), source_profile, target_profile)
    except (OSError, ImageCms.PyCMSError) as exc:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Could not convert the source ICC profile to sRGB.",
            details={"reason": str(exc)},
        ) from exc


def _save_kwargs(
    output_format: OutputFormat,
    *,
    quality: int,
    source_metadata: SourceMetadata | None,
    strip_metadata: bool,
    target_profile: str | None,
) -> dict[str, object]:
    if output_format == OutputFormat.JPG:
        kwargs: dict[str, object] = {"format": "JPEG", "quality": quality}
    elif output_format == OutputFormat.WEBP:
        kwargs = {"format": "WEBP", "quality": quality}
    else:
        kwargs = {"format": "PNG"}
    if strip_metadata:
        return kwargs
    if target_profile == "srgb":
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        return kwargs
    if source_metadata and source_metadata.icc_profile:
        kwargs["icc_profile"] = source_metadata.icc_profile
    exif_formats = {OutputFormat.JPG, OutputFormat.WEBP}
    if source_metadata and source_metadata.exif and output_format in exif_formats:
        kwargs["exif"] = source_metadata.exif
    return kwargs


def _temp_output_path(temp_dir: Path, output_format: OutputFormat) -> Path:
    extension = "jpg" if output_format == OutputFormat.JPG else output_format.value
    return temp_dir / f"pixelup-{os.getpid()}-{uuid4().hex}.{extension}"
