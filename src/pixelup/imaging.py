from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pixelup.errors import ErrorCode, PixelupError


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

