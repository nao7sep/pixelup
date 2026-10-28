"""What PixelUp knows about the output bundle a job publishes: an image and its JSON
sidecar sharing one stem.

Job planning gives every queued job a distinct bundle in memory (``jobs.py``), and
one PixelUp process is the supported writer, so no cross-process reservation exists.
Publication itself never replaces an existing entry: each member is claimed
atomically (hard link or ``O_EXCL``), and the check here refuses a bundle whose
name is already taken, once, before inference.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.session_log import log

BUNDLE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".json")


def assert_output_bundle_available(output_path: Path) -> None:
    occupied = next(iter(_bundle_entries(output_path)), None)
    if occupied is not None:
        raise _bundle_exists(output_path, occupied)


def remove_published_file(path: Path) -> None:
    """Remove a file this job published; one PixelUp process plans every job a
    distinct output bundle, so the name is the job's own."""
    try:
        os.unlink(path)
    except OSError:
        log.warning("output.cleanup_left_file", path=str(path))


def _bundle_entries(output_path: Path) -> list[Path]:
    stem_identity = normalized_name(output_path.stem)
    try:
        with os.scandir(output_path.parent) as entries:
            return [
                output_path.parent / entry.name
                for entry in entries
                if normalized_name(Path(entry.name).stem) == stem_identity
                and normalized_name(Path(entry.name).suffix) in BUNDLE_SUFFIXES
            ]
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.outputDirInspectFailed"),
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc


def normalized_name(value: str) -> str:
    """The same filename identity for planning and publication on every volume."""
    return unicodedata.normalize("NFC", value).casefold()


def _bundle_exists(output_path: Path, occupied: Path) -> PixelupError:
    # Directory enumeration includes broken symlinks. Publication must never replace
    # any pre-existing entry in the normalized bundle identity.
    return PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        Message("error.outputBundleExists"),
        hint=Message("error.hintRetryNewName"),
        details={"output": str(output_path), "occupied": str(occupied)},
    )
