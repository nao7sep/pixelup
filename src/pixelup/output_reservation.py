from __future__ import annotations

import hashlib
import os
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from pixelup.errors import ErrorCode, PixelupError

CancelCheck = Callable[[], bool]
WaitingCallback = Callable[[], None]
_POLL_SECONDS = 0.25
_OUTPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_BUNDLE_SUFFIXES = frozenset((*_OUTPUT_SUFFIXES, ".json"))


@dataclass(frozen=True, slots=True)
class PublishedFile:
    path: Path
    device: int
    inode: int


@contextmanager
def reserve_output_bundle(
    output_path: Path,
    temp_dir: Path,
    *,
    timeout: float,
    should_cancel: CancelCheck | None = None,
    on_waiting: WaitingCallback | None = None,
) -> Iterator[None]:
    """Serialize one output image plus its JSON sidecar across PixelUp processes."""
    if timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Output lock timeout must not be negative.")
    locks_dir = temp_dir / "output-locks"
    try:
        locks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not create the output lock directory.",
            details={"path": str(locks_dir), "reason": str(exc)},
        ) from exc

    lock = FileLock(str(locks_dir / f"{_output_lock_key(output_path)}.lock"))
    started = time.monotonic()
    while True:
        if should_cancel and should_cancel():
            raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
        remaining = timeout - (time.monotonic() - started)
        if timeout > 0 and remaining <= 0:
            raise PixelupError(
                ErrorCode.OUTPUT_EXISTS,
                "Timed out waiting to reserve the output file.",
                details={"output": str(output_path), "timeout": timeout},
            )
        try:
            lock.acquire(timeout=0 if timeout == 0 else min(_POLL_SECONDS, remaining))
            break
        except Timeout as exc:
            if timeout == 0:
                raise PixelupError(
                    ErrorCode.OUTPUT_EXISTS,
                    "The output file is already reserved by another job.",
                    details={"output": str(output_path)},
                ) from exc
            if on_waiting:
                on_waiting()
    try:
        assert_output_bundle_available(output_path)
        yield
    finally:
        lock.release()


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
    hold = published.path.with_name(f".{uuid.uuid4().hex}.pixelup-hold")
    try:
        # Rename first: this preserves whichever inode occupies the public pathname
        # at the destructive boundary. Ownership is checked only after the entry is
        # safely under our private sibling, so a replacement can never be unlinked.
        os.rename(published.path, hold)
    except OSError:
        return False

    try:
        displaced = os.lstat(hold)
    except OSError:
        return False
    if (displaced.st_dev, displaced.st_ino) != (published.device, published.inode):
        _restore_displaced_file(hold, published.path)
        return False

    try:
        hold.unlink()
    except OSError:
        return False
    return True


def _restore_displaced_file(hold: Path, destination: Path) -> None:
    """Restore a non-owned entry without replacing a later destination winner.

    A hard link preserves the exact inode and is an atomic no-clobber operation.
    On a filesystem without hard links, leaving the entry at the private hold is
    safer than copying or replacing bytes PixelUp does not own.
    """
    try:
        os.link(hold, destination, follow_symlinks=False)
    except OSError:
        return
    try:
        hold.unlink()
    except OSError:
        pass


def published_file(path: Path, descriptor: int) -> PublishedFile:
    current = os.fstat(descriptor)
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
            "Could not inspect the output directory.",
            details={"output": str(output_path), "reason": str(exc)},
        ) from exc


def _text_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _bundle_exists(output_path: Path, occupied: Path) -> PixelupError:
    # Directory enumeration includes broken symlinks. Publication must never replace
    # any pre-existing entry in the normalized bundle identity.
    return PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        "The output file or its settings sidecar already exists.",
        hint="Retry the job to choose a new unused filename.",
        details={"output": str(output_path), "occupied": str(occupied)},
    )


def _output_lock_key(output_path: Path) -> str:
    # resolve() canonicalizes symlinked parents. Case-folding is deliberately
    # conservative: PixelUp never assigns two case-only sibling names, even on a
    # case-sensitive volume, so aliases that would collide on macOS/Windows share one
    # lock. The sidecar path is the bundle identity: format variants share a stem and
    # one .json companion, so result.png and result.jpg must serialize too.
    bundle_path = output_path.with_suffix(".json")
    canonical = _text_identity(os.path.normcase(str(bundle_path.expanduser().resolve())))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
