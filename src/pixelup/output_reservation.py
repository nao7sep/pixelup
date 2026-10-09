"""What PixelUp knows about the output bundle a job publishes: an image and its JSON
sidecar sharing one stem.

Job planning gives every queued job a distinct bundle in memory (``jobs.py``), and
one PixelUp process is the supported writer, so no cross-process reservation exists.
Publication itself never replaces an existing entry: each member is claimed
atomically (hard link or ``O_EXCL``), and the checks here refuse a bundle another
writer has touched.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.session_log import log

_OUTPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_BUNDLE_SUFFIXES = frozenset((*_OUTPUT_SUFFIXES, ".json"))


@dataclass(frozen=True, slots=True)
class PublishedFile:
    path: Path
    device: int
    inode: int


def assert_output_bundle_available(output_path: Path) -> None:
    occupied = next(iter(_bundle_entries(output_path)), None)
    if occupied is not None:
        raise _bundle_exists(output_path, occupied)


def assert_output_bundle_claims_current(
    output_path: Path,
    claims: tuple[PublishedFile, ...],
) -> None:
    """Make the supplied physical files the only members of this normalized bundle."""
    for claim in claims:
        if not published_file_is_current(claim):
            raise _bundle_exists(output_path, claim.path)

    claims_by_name = {claim.path.name: claim for claim in claims}
    for occupied in _bundle_entries(output_path):
        claim = claims_by_name.get(occupied.name)
        if claim is None or not published_file_is_current(claim):
            raise _bundle_exists(output_path, occupied)


def published_file_is_current(published: PublishedFile) -> bool:
    try:
        current = os.lstat(published.path)
    except OSError:
        return False
    return (current.st_dev, current.st_ino) == (published.device, published.inode)


def remove_published_file(published: PublishedFile) -> bool:
    """Remove a file PixelUp published, if the path still names it; False leaves it in place.

    One PixelUp process plans every job a distinct output bundle, so the identity
    check immediately before the unlink is what proves the entry is still ours; an
    entry anything else put at the name is left alone.
    """
    removed = published_file_is_current(published)
    if removed:
        try:
            os.unlink(published.path)
        except OSError:
            removed = False
    if not removed:
        log.warning("output.cleanup_left_file", path=str(published.path))
    return removed


def close_published_file(path: Path, descriptor: int) -> PublishedFile:
    """The claim on ``path`` for the descriptor that created it, which is then closed.

    Called once the bytes are written, written or failed: an exFAT or FAT file id
    changes when the first bytes are (storage-path-conventions).
    """
    try:
        current = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return PublishedFile(path, current.st_dev, current.st_ino)


def _bundle_entries(output_path: Path) -> list[Path]:
    stem_identity = _text_identity(output_path.stem)
    try:
        with os.scandir(output_path.parent) as entries:
            return [
                output_path.parent / entry.name
                for entry in entries
                if _text_identity(Path(entry.name).stem) == stem_identity
                and _text_identity(Path(entry.name).suffix) in _BUNDLE_SUFFIXES
            ]
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.outputDirInspectFailed"),
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc


def _text_identity(value: str) -> str:
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
