from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from pixelup.errors import ErrorCode, PixelupError

CancelCheck = Callable[[], bool]
WaitingCallback = Callable[[], None]
_POLL_SECONDS = 0.25
_OUTPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


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
    occupied = next((path for path in _bundle_paths(output_path) if os.path.lexists(path)), None)
    if occupied is not None:
        raise _bundle_exists(output_path, occupied)


def assert_output_companions_available(output_path: Path) -> None:
    occupied = next(
        (
            path
            for path in _bundle_paths(output_path)
            if path != output_path and os.path.lexists(path)
        ),
        None,
    )
    if occupied is not None:
        raise _bundle_exists(output_path, occupied)


def _bundle_paths(output_path: Path) -> list[Path]:
    sidecar_path = output_path.with_suffix(".json")
    return [
        *(output_path.with_suffix(suffix) for suffix in _OUTPUT_SUFFIXES),
        sidecar_path,
    ]


def _bundle_exists(output_path: Path, occupied: Path) -> PixelupError:
    # lexists, unlike Path.exists, treats a broken symlink as an occupied path.
    # Publication must never replace any pre-existing directory entry.
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
    canonical = os.path.normcase(str(bundle_path.expanduser().resolve())).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
