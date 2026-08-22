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

from pixelup.errors import ErrorCode, PixelupError
from pixelup.icc_profiles import profile_bytes as generated_profile_bytes
from pixelup.nanoid import nanoid
from pixelup.output_reservation import (
    assert_output_bundle_available,
    assert_output_companions_available,
)
from pixelup.paths import OutputFormat
from pixelup.session_log import log


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    icc_profile: bytes | None = None
    exif: bytes | None = None
    xmp: bytes | None = None


@dataclass(frozen=True, slots=True)
class PublishedImage:
    path: Path
    device: int
    inode: int


PublishedCallback = Callable[[PublishedImage], None]


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
                if not _published_image_is_current(published):
                    raise _output_exists(output_path)
                assert_output_companions_available(output_path)
            except PixelupError:
                remove_published_image(published)
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
            "Could not write the output image.",
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc
    return encoded.size


def remove_published_image(published: PublishedImage) -> bool:
    """Remove PixelUp's image only while its exact filesystem identity still owns the path."""
    if not _published_image_is_current(published):
        return False
    try:
        published.path.unlink()
    except OSError:
        return False
    return True


def _published_image_is_current(published: PublishedImage) -> bool:
    try:
        current = os.lstat(published.path)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if (current.st_dev, current.st_ino) != (published.device, published.inode):
        return False
    return True


def _publish_image_no_clobber(temp_path: Path, output_path: Path) -> PublishedImage:
    source = os.stat(temp_path)
    try:
        # A hard-link publish is atomic, no-clobber, and keeps the already-fsynced
        # staged bytes invisible at the final name until the single link operation.
        os.link(temp_path, output_path, follow_symlinks=False)
        return PublishedImage(output_path, source.st_dev, source.st_ino)
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
    published_stat = os.fstat(descriptor)
    published = PublishedImage(output_path, published_stat.st_dev, published_stat.st_ino)
    try:
        with temp_path.open("rb") as source_file, os.fdopen(descriptor, "wb") as output_file:
            shutil.copyfileobj(source_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
    except Exception:
        remove_published_image(published)
        raise
    return published


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _output_exists(output_path: Path) -> PixelupError:
    return PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        "Output file already exists.",
        hint="Retry the job to choose a new unused filename.",
        details={"output": str(output_path)},
    )


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
    if target_profile is not None:
        source_profile = _source_profile_bytes(source_metadata)
        prepared = _convert_profile(prepared, source_profile, _profile_bytes(target_profile))
    elif strip_metadata and source_metadata and source_metadata.icc_profile:
        prepared = _convert_profile(prepared, source_metadata.icc_profile, _profile_bytes("srgb"))
        prepared.info.pop("icc_profile", None)
    return prepared


def _flatten_alpha(image: Image.Image, background: str) -> Image.Image:
    if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
        return image.convert("RGB")
    try:
        background_rgb = ImageColor.getrgb(background)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Background is not a valid color.",
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
    if target_profile is not None:
        kwargs["icc_profile"] = _profile_bytes(target_profile)
    elif not strip_metadata:
        if source_metadata and source_metadata.icc_profile:
            kwargs["icc_profile"] = source_metadata.icc_profile
        else:
            kwargs["icc_profile"] = _profile_bytes("srgb")
    if strip_metadata:
        return kwargs
    exif_formats = {OutputFormat.JPG, OutputFormat.PNG, OutputFormat.WEBP}
    if source_metadata and source_metadata.exif and output_format in exif_formats:
        kwargs["exif"] = source_metadata.exif
    if source_metadata and source_metadata.xmp:
        if output_format == OutputFormat.PNG:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_itxt(
                "XML:com.adobe.xmp",
                source_metadata.xmp.decode("utf-8", errors="replace"),
            )
            kwargs["pnginfo"] = pnginfo
        elif output_format in {OutputFormat.JPG, OutputFormat.WEBP}:
            kwargs["xmp"] = source_metadata.xmp
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


def _source_profile_bytes(source_metadata: SourceMetadata | None) -> bytes:
    if source_metadata and source_metadata.icc_profile:
        return source_metadata.icc_profile
    return _profile_bytes("srgb")


def _profile_bytes(name: str) -> bytes:
    try:
        return generated_profile_bytes(name)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Target profile must be one of sRGB, Display P3, or Adobe RGB.",
        ) from exc
    except (OSError, ImageCms.PyCMSError) as exc:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Target ICC profile is not available.",
            details={"target_profile": name, "reason": str(exc)},
        ) from exc
