from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageColor, PngImagePlugin, UnidentifiedImageError

from pixelup.derived_metadata import (
    OutputColor,
    exif_declares_adobe_rgb,
    exif_for_output,
    xmp_for_output,
)
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.icc_profiles import profile_bytes as generated_profile_bytes
from pixelup.nanoid import nanoid
from pixelup.output_reservation import (
    PublishedFile,
    assert_output_bundle_available,
    assert_output_bundle_claims_current,
    published_file,
    remove_published_file,
)
from pixelup.paths import OutputFormat
from pixelup.session_log import log


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    icc_profile: bytes | None = None
    exif: bytes | None = None
    xmp: bytes | None = None


@dataclass(frozen=True, slots=True)
class _ColorPlan:
    source_profile: bytes
    # The profile the written pixels are in, and whether the file embeds it.
    output_profile: bytes
    embed_profile: bool
    # The PixelUp colour space the output was converted to; None when it keeps the source's.
    converted_to: OutputColor | None


PublishedCallback = Callable[[PublishedFile], None]


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
            Message("error.inputInvalidFormat"),
            details={"input": str(path)},
        ) from exc
    except PermissionError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            Message("error.inputUnreadable"),
            details={"input": str(path), "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            Message("error.inputOpenFailed"),
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
            xmp = image.info.get("xmp")
    except (OSError, UnidentifiedImageError) as exc:
        # Source metadata is optional; an unreadable header is not a failure of
        # the upscale. Note it at debug for diagnosis without alarming a run.
        log.debug("metadata.read_failed", input=str(path), reason=str(exc))
        return SourceMetadata()
    return SourceMetadata(
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
        exif=exif if isinstance(exif, bytes) else None,
        xmp=xmp if isinstance(xmp, bytes) else None,
    )


def save_output_image(
    image: Image.Image,
    *,
    output_path: Path,
    output_format: OutputFormat,
    quality: int,
    background: str,
    source_metadata: SourceMetadata | None,
    strip_metadata: bool,
    target_profile: str | None,
    on_published: PublishedCallback | None = None,
) -> tuple[int, int]:
    source_metadata = source_metadata or SourceMetadata()
    color = _plan_color(
        source_metadata, strip_metadata=strip_metadata, target_profile=target_profile
    )
    encoded = _prepare_image_for_save(
        image, output_format=output_format, background=background, color=color
    )
    save_kwargs = _save_kwargs(
        output_format,
        quality=quality,
        color=color,
        carried=None if strip_metadata else _carried_metadata(source_metadata, encoded.size, color),
    )
    temp_path = _temp_output_path(output_path)
    try:
        with temp_file_guard(temp_path):
            encoded.save(temp_path, **save_kwargs)
            _fsync_file(temp_path)
            # not recorded: this is the harvest-then-discard OUTPUT image, a binary
            # written for the user at a user-chosen location — not managed text the
            # app owns and reloads as state. Binaries are out of scope for the text
            # backup, and output is never recorded (data-backup-conventions).
            # Revalidate every member of the shared-stem bundle after encoding,
            # immediately before publication. The final path itself is still
            # claimed atomically below, so an exact-boundary winner is preserved.
            assert_output_bundle_available(output_path)
            published = _publish_image_no_clobber(temp_path, output_path)
            try:
                assert_output_bundle_claims_current(output_path, (published,))
            except PixelupError:
                remove_published_file(published)
                raise
            if on_published:
                on_published(published)
    except PixelupError:
        temp_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.outputWriteFailed"),
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc
    return encoded.size


def _publish_image_no_clobber(temp_path: Path, output_path: Path) -> PublishedFile:
    source = os.stat(temp_path)
    try:
        # A hard-link publish is atomic, no-clobber, and keeps the already-fsynced
        # staged bytes invisible at the final name until the single link operation.
        os.link(temp_path, output_path, follow_symlinks=False)
        return PublishedFile(output_path, source.st_dev, source.st_ino)
    except FileExistsError as exc:
        raise _output_exists(output_path) from exc
    except OSError as exc:
        unsupported = {
            errno.EACCES,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.EPERM,
        }
        if exc.errno not in unsupported:
            raise

    # Filesystems without hard links (notably removable exFAT volumes) still get
    # an atomic O_EXCL path claim. Bytes are copied through the claimed descriptor,
    # so deleting/replacing the pathname cannot make PixelUp overwrite the winner.
    try:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as exc:
        raise _output_exists(output_path) from exc
    published = published_file(output_path, descriptor)
    try:
        with temp_path.open("rb") as source_file, os.fdopen(descriptor, "wb") as output_file:
            shutil.copyfileobj(source_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
    except Exception:
        remove_published_file(published)
        raise
    return published


def _fsync_file(path: Path) -> None:
    # Windows maps fsync to _commit, which rejects a read-only descriptor even
    # though Unix accepts one. The staged file is ours and already complete, so
    # reopen it read/write solely for the cross-platform durability boundary.
    with path.open("r+b") as file:
        os.fsync(file.fileno())


def _output_exists(output_path: Path) -> PixelupError:
    return PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        Message("error.outputExists"),
        hint=Message("error.hintRetryNewName"),
        details={"output": str(output_path)},
    )


def _plan_color(
    source_metadata: SourceMetadata,
    *,
    strip_metadata: bool,
    target_profile: str | None,
) -> _ColorPlan:
    source_profile = _source_profile_bytes(source_metadata)
    if target_profile is not None:
        output_profile = _profile_bytes(target_profile)
        return _ColorPlan(
            source_profile,
            output_profile,
            embed_profile=True,
            converted_to=OutputColor(target_profile, _profile_description(output_profile)),
        )
    if strip_metadata:
        # A file with no profile is read as sRGB, so its pixels must be sRGB.
        return _ColorPlan(
            source_profile, _profile_bytes("srgb"), embed_profile=False, converted_to=None
        )
    return _ColorPlan(source_profile, source_profile, embed_profile=True, converted_to=None)


def _prepare_image_for_save(
    image: Image.Image,
    *,
    output_format: OutputFormat,
    background: str,
    color: _ColorPlan,
) -> Image.Image:
    prepared = image
    if output_format == OutputFormat.JPG:
        prepared = _flatten_alpha(prepared, background)
    elif prepared.mode not in {"RGB", "RGBA"}:
        prepared = prepared.convert("RGBA" if "A" in prepared.getbands() else "RGB")
    if color.output_profile != color.source_profile:
        prepared = _convert_profile(prepared, color.source_profile, color.output_profile)
    return prepared


def _flatten_alpha(image: Image.Image, background: str) -> Image.Image:
    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return image.convert("RGB")
    try:
        background_rgb = ImageColor.getrgb(background)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.backgroundInvalid"),
            details={"background": background},
        ) from exc
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, background_rgb + (255,))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _convert_profile(
    image: Image.Image,
    source_profile: bytes,
    target_profile: bytes,
) -> Image.Image:
    try:
        source = ImageCms.ImageCmsProfile(BytesIO(source_profile))
        target = ImageCms.ImageCmsProfile(BytesIO(target_profile))
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        return ImageCms.profileToProfile(image.convert(mode), source, target)
    except (OSError, ImageCms.PyCMSError) as exc:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Could not convert the image color profile.",
            details={"reason": str(exc)},
        ) from exc


def _carried_metadata(
    source_metadata: SourceMetadata,
    size: tuple[int, int],
    color: _ColorPlan,
) -> SourceMetadata:
    """The source's EXIF and XMP as the output carries them: describing the output."""
    exif = source_metadata.exif
    if exif is not None:
        try:
            exif = exif_for_output(exif, size=size, color=color.converted_to)
        except ValueError as exc:
            # A block that is not a readable TIFF structure names no dimension or colour
            # space any reader could use, so it is carried as it is.
            log.debug("metadata.exif_unreadable", reason=str(exc))
    xmp = source_metadata.xmp
    if xmp is not None:
        xmp = xmp_for_output(xmp, size=size, color=color.converted_to)
    return SourceMetadata(exif=exif, xmp=xmp)


def _save_kwargs(
    output_format: OutputFormat,
    *,
    quality: int,
    color: _ColorPlan,
    carried: SourceMetadata | None,
) -> dict[str, object]:
    if output_format == OutputFormat.JPG:
        kwargs: dict[str, object] = {"format": "JPEG", "quality": quality}
    elif output_format == OutputFormat.WEBP:
        kwargs = {"format": "WEBP", "quality": quality}
    else:
        kwargs = {"format": "PNG"}
    # Always passed, so no encoder falls back to a profile left in the image's info.
    kwargs["icc_profile"] = color.output_profile if color.embed_profile else None
    if carried is None:
        return kwargs
    if carried.exif:
        kwargs["exif"] = carried.exif
    if carried.xmp:
        if output_format == OutputFormat.PNG:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_itxt(
                "XML:com.adobe.xmp",
                carried.xmp.decode("utf-8", errors="replace"),
            )
            kwargs["pnginfo"] = pnginfo
        else:
            kwargs["xmp"] = carried.xmp
    return kwargs


def _temp_output_path(output_path: Path) -> Path:
    # Staged beside output_path, not in a central temp directory: the atomic
    # hard-link publication requires one filesystem volume, and output_path may
    # be on a USB stick or second disk rather than ~/.pixelup/temp/. The name
    # still follows the house <stem>-<discriminator>.tmp shape, derived from the
    # target's own stem.
    return output_path.parent / f"{output_path.stem}-{nanoid()}.tmp"


@contextmanager
def temp_file_guard(path: Path) -> Iterator[None]:
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _source_profile_bytes(source_metadata: SourceMetadata) -> bytes:
    if source_metadata.icc_profile:
        return source_metadata.icc_profile
    # An untagged camera file that EXIF marks as Adobe RGB holds Adobe RGB pixels.
    if source_metadata.exif and exif_declares_adobe_rgb(source_metadata.exif):
        return _profile_bytes("adobergb")
    return _profile_bytes("srgb")


def _profile_description(profile: bytes) -> str:
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))).strip()


def _profile_bytes(name: str) -> bytes:
    try:
        return generated_profile_bytes(name)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.targetProfileInvalid"),
        ) from exc
    except (OSError, ImageCms.PyCMSError) as exc:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Target ICC profile is not available.",
            details={"target_profile": name, "reason": str(exc)},
        ) from exc
