from __future__ import annotations

import hashlib
import os
import re
import select
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from filelock import FileLock, Timeout

from pixelup.errors import ErrorCode, PixelupError
from pixelup.model_registry import (  # noqa: F401 - preserve the existing public imports
    ALL_MODELS,
    FACEXLIB_RELEASES,
    GFPGAN_RELEASES,
    KNOWN_MODELS,
    REAL_ESRGAN_RELEASES,
    ModelInfo,
    known_model,
)
from pixelup.nanoid import nanoid
from pixelup.session_log import log

DownloadCallback = Callable[[str, int, int | None], None]
WaitingCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_REPORT_BYTES = 5 * 1024 * 1024
DOWNLOAD_POLL_SECONDS = 0.25
MIN_DOWNLOAD_RATE_BYTES_PER_SECOND = 128 * 1024
LARGE_DOWNLOAD_FINALIZATION_SECONDS = 120
MAX_DOWNLOAD_REDIRECTS = 10
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(slots=True)
class _ConnectionAttempt:
    response: object | None = None
    error: BaseException | None = None
    abandoned: bool = False


def model_file(models_dir: Path, name: str) -> Path:
    info = known_model(name)
    filename = info.filename if info else f"{name}.pth"
    return models_dir / filename


def require_model_present(
    models_dir: Path,
    name: str,
) -> Path:
    path = model_file(models_dir, name)
    if not _model_file_present(path):
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{name}' is not present in the models directory.",
            hint=(
                "Turn on “Download missing models automatically” in Settings, "
                "or place the .pth file in the models directory."
            ),
            details={"model": name, "models_dir": str(models_dir), "path": str(path)},
        )
    # A present model is trusted: it was verified once when PixelUp downloaded it (or
    # placed by the user). The managed-runtime-dependencies-conventions drop
    # re-verification after acquisition, and re-hashing here would re-read the whole
    # file before every job — hundreds of MB per face-enhance image — to guard against
    # a corruption that torch's weights_only load already turns into a loud failure.
    return path


def download_model(
    models_dir: Path,
    name: str,
    *,
    download_timeout: float,
    lock_timeout: int,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, object]:
    info = known_model(name)
    if info is None or info.url is None:
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{name}' is not known to PixelUp's downloader.",
            details={"model": name},
        )
    return download_model_info(
        models_dir,
        info,
        download_timeout=download_timeout,
        lock_timeout=lock_timeout,
        on_download=on_download,
        on_waiting=on_waiting,
        should_cancel=should_cancel,
    )


def download_model_info(
    models_dir: Path,
    info: ModelInfo,
    *,
    download_timeout: float,
    lock_timeout: int,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, object]:
    if download_timeout <= 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Download timeout must be positive.")
    if lock_timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Lock timeout must be 0 or greater.")
    if info.url is None:
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{info.name}' does not have a download URL.",
            details={"model": info.name},
        )

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            "Could not create the models directory.",
            details={"models_dir": str(models_dir), "reason": str(exc)},
        ) from exc
    target = models_dir / info.filename
    if _model_file_present(target):
        return _download_result(info, target, "present")

    locks_dir = models_dir / ".locks"
    try:
        locks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            "Could not create the model lock directory.",
            details={"locks_dir": str(locks_dir), "reason": str(exc)},
        ) from exc

    lock = FileLock(str(locks_dir / f"{_lock_name(info.name)}.lock"))
    _acquire_download_lock(lock, info.name, lock_timeout, on_waiting, should_cancel)
    try:
        if _model_file_present(target):
            return _download_result(info, target, "present")
        acquisition_timeout = _acquisition_timeout_seconds(info, download_timeout)
        deadline = time.monotonic() + acquisition_timeout
        # Stage the download as a per-download-unique file INSIDE models_dir —
        # deliberately, not under a separate temp/ dir. The convention's intent
        # (a deletable staging area, unique name, verify there, then atomic
        # publish) is met: the staged file is removed on every failure path
        # below, and same-directory staging is precisely what makes os.replace
        # an atomic same-filesystem rename — a cross-volume temp/ could degrade
        # that to a copy. (Image-output staging uses temp/ in imaging.py; model
        # publish needs the same-fs guarantee.)
        temp_path = models_dir / f"{target.stem}-{nanoid()}.tmp"
        log.info(
            "model.download_started",
            model=info.name,
            url=info.url,
            timeout_seconds=acquisition_timeout,
        )
        try:
            try:
                _download_to_temp(
                    info,
                    temp_path,
                    deadline=deadline,
                    on_download=on_download,
                    should_cancel=should_cancel,
                )
                verify_model_file(
                    temp_path,
                    info,
                    deadline=deadline,
                    should_cancel=should_cancel,
                )
                _sync_staged_file(temp_path, deadline, should_cancel)
                _check_acquisition(deadline, should_cancel)
                # not recorded: model weights are large binaries, re-fetchable from
                # their source and interchangeable with it — not hand-authored text the
                # app owns as state. Binaries are out of scope for the text backup, and
                # models/ is a binary-bearing directory excluded wholesale
                # (data-backup-conventions).
                os.replace(temp_path, target)
                _sync_directory_best_effort(models_dir)
            except PixelupError as exc:
                # A cancellation is not a download failure; the job-level log records
                # it. Any other PixelupError here is a real failure (e.g. the
                # downloaded file failed verification) and gets a terminal event so
                # every model.download_started has a matching outcome.
                if exc.code != ErrorCode.JOB_CANCELLED:
                    log.warning(
                        "model.download_failed",
                        model=info.name,
                        url=info.url,
                        code=exc.code.value,
                        reason=exc.message,
                    )
                raise
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                log.warning(
                    "model.download_failed",
                    model=info.name,
                    url=info.url,
                    reason=str(exc),
                )
                raise PixelupError(
                    ErrorCode.MODEL_DOWNLOAD_FAILED,
                    f"Could not download model '{info.name}'.",
                    details={"model": info.name, "url": info.url, "reason": str(exc)},
                ) from exc
        finally:
            _remove_staged_file(temp_path, info.name)
    finally:
        lock.release()
    result = _download_result(info, target, "downloaded")
    log.info("model.download_finished", model=info.name, size_bytes=result["size_bytes"])
    return result


def verify_model_file(
    path: Path,
    info: ModelInfo | None = None,
    *,
    deadline: float | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, object]:
    _check_acquisition(deadline, should_cancel)
    if not path.is_file():
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            "Model file is missing.",
            details={"path": str(path)},
        )
    size = path.stat().st_size
    # A model PixelUp knows how to download must carry a pinned SHA-256; otherwise
    # verification would silently fall back to a size check, which cannot detect a
    # substituted same-size file. A missing pin is a registry defect, not a soft pass.
    if info is not None and info.url is not None and info.checksum_sha256 is None:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            f"Model '{info.name}' has no pinned checksum; refusing to trust it.",
            details={"model": info.name, "path": str(path)},
        )
    checksum = (
        _sha256(path, deadline=deadline, should_cancel=should_cancel)
        if info and info.checksum_sha256
        else None
    )
    ok = size > 0
    if info and info.expected_size is not None:
        ok = ok and size == info.expected_size
    if info and info.checksum_sha256 is not None:
        ok = ok and checksum == info.checksum_sha256

    result = {
        "name": info.name if info else path.stem,
        "path": str(path),
        "ok": ok,
        "size_bytes": size,
        "expected_size_bytes": info.expected_size if info else None,
        "checksum_sha256": checksum,
        "expected_checksum_sha256": info.checksum_sha256 if info else None,
    }
    if not ok:
        raise PixelupError(
            ErrorCode.MODEL_CORRUPT,
            "Model file failed verification.",
            details={"model": result},
        )
    return result


def _download_to_temp(
    info: ModelInfo,
    temp_path: Path,
    *,
    deadline: float,
    on_download: DownloadCallback | None,
    should_cancel: CancelCheck | None = None,
) -> None:
    assert info.url is not None
    bytes_done = 0
    last_reported = 0
    last_percent = -1
    with _open_download_response(info.url, deadline, should_cancel) as response:
        advertised_size = _response_size(response.headers.get("Content-Length"), None)
        if (
            info.expected_size is not None
            and advertised_size is not None
            and advertised_size > info.expected_size
        ):
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                "Model download exceeds its pinned size.",
                details={
                    "model": info.name,
                    "expected_size_bytes": info.expected_size,
                    "advertised_size_bytes": advertised_size,
                },
            )
        bytes_total = advertised_size if advertised_size is not None else info.expected_size
        with temp_path.open("wb") as output:
            while chunk := _read_download_chunk(response, deadline, should_cancel):
                next_bytes_done = bytes_done + len(chunk)
                if info.expected_size is not None and next_bytes_done > info.expected_size:
                    raise PixelupError(
                        ErrorCode.MODEL_DOWNLOAD_FAILED,
                        "Model download exceeded its pinned size while streaming.",
                        details={
                            "model": info.name,
                            "expected_size_bytes": info.expected_size,
                            "received_size_bytes": next_bytes_done,
                        },
                    )
                output.write(chunk)
                bytes_done = next_bytes_done
                if _should_report_download(bytes_done, bytes_total, last_reported, last_percent):
                    if on_download:
                        on_download(info.name, bytes_done, bytes_total)
                    last_reported = bytes_done
                    if bytes_total:
                        last_percent = bytes_done * 100 // bytes_total
    if on_download and bytes_done != last_reported:
        on_download(info.name, bytes_done, bytes_total)


def _open_download_response(
    url: str,
    deadline: float,
    should_cancel: CancelCheck | None,
):
    initial_scheme = urlsplit(url).scheme.lower()
    allow_file = initial_scheme == "file"
    _assert_safe_download_url(url, allow_file=allow_file)
    opener = build_opener(_NoRedirectHandler())
    current_url = url
    for redirects in range(MAX_DOWNLOAD_REDIRECTS + 1):
        _raise_if_cancelled(should_cancel)
        try:
            response = _open_response_cancellable(opener, current_url, deadline, should_cancel)
        except HTTPError as exc:
            if exc.code not in _REDIRECT_STATUSES:
                exc.close()
                raise
            if redirects == MAX_DOWNLOAD_REDIRECTS:
                exc.close()
                raise PixelupError(
                    ErrorCode.MODEL_DOWNLOAD_FAILED,
                    "Model download followed too many redirects.",
                    details={"url": url},
                ) from exc
            location = exc.headers.get("Location")
            exc.close()
            if not location:
                raise PixelupError(
                    ErrorCode.MODEL_DOWNLOAD_FAILED,
                    "Model download redirect did not include a Location header.",
                    details={"url": current_url},
                ) from exc
            current_url = urljoin(current_url, location)
            _assert_safe_download_url(current_url, allow_file=False)
            continue
        effective_url = response.geturl()
        try:
            _assert_safe_download_url(effective_url, allow_file=allow_file)
        except Exception:
            response.close()
            raise
        return response
    raise AssertionError("redirect loop must return or raise")


def _open_response_cancellable(
    opener,
    url: str,
    deadline: float,
    should_cancel: CancelCheck | None,
):
    """Open without letting urllib's blocking DNS/connect/TLS hide cancel or deadline.

    The daemon worker owns only connection setup. Response ownership transfers under
    the lock; if the caller has already left, the worker closes a late response.
    """
    attempt = _ConnectionAttempt()
    ready = threading.Event()
    lock = threading.Lock()
    open_timeout = _remaining_download_seconds(deadline)

    def connect() -> None:
        try:
            response = opener.open(url, timeout=open_timeout)
        except BaseException as exc:
            with lock:
                if attempt.abandoned:
                    return
                attempt.error = exc
                ready.set()
            return

        with lock:
            if attempt.abandoned:
                close_abandoned = True
            else:
                attempt.response = response
                ready.set()
                close_abandoned = False
        if close_abandoned:
            _close_response(response)

    threading.Thread(target=connect, name="pixelup-model-connect", daemon=True).start()

    try:
        while True:
            _raise_if_cancelled(should_cancel)
            remaining = _remaining_download_seconds(deadline)
            if ready.wait(min(DOWNLOAD_POLL_SECONDS, remaining)):
                break
        # Cancellation or the total deadline wins until response ownership transfers.
        _raise_if_cancelled(should_cancel)
        _remaining_download_seconds(deadline)
    except BaseException:
        with lock:
            attempt.abandoned = True
            response = attempt.response
            attempt.response = None
        if response is not None:
            _close_response(response)
        raise

    with lock:
        response = attempt.response
        error = attempt.error
        attempt.response = None
    if error is not None:
        raise error
    if response is None:
        raise RuntimeError("Connection worker completed without a response or error.")
    return response


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _read_download_chunk(response, deadline: float, should_cancel: CancelCheck | None) -> bytes:
    sock = _response_socket(response)
    if sock is None:
        _raise_if_cancelled(should_cancel)
        _remaining_download_seconds(deadline)
        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
        _raise_if_cancelled(should_cancel)
        _remaining_download_seconds(deadline)
        return chunk

    while True:
        _raise_if_cancelled(should_cancel)
        remaining = _remaining_download_seconds(deadline)
        pending = getattr(sock, "pending", None)
        if not (callable(pending) and pending() > 0):
            readable, _, _ = select.select(
                [sock],
                [],
                [],
                min(DOWNLOAD_POLL_SECONDS, remaining),
            )
            if not readable:
                continue

        remaining = _remaining_download_seconds(deadline)
        set_timeout = getattr(sock, "settimeout", None)
        if callable(set_timeout):
            set_timeout(min(DOWNLOAD_POLL_SECONDS, remaining))
        reader = getattr(response, "read1", None) or response.read
        try:
            chunk = reader(DOWNLOAD_CHUNK_BYTES)
        except TimeoutError:
            # A readable raw socket may still hold only part of a TLS record.
            # Return to the polling loop so cancellation and the total deadline
            # remain responsive without treating slow progress as a failure.
            continue
        _raise_if_cancelled(should_cancel)
        _remaining_download_seconds(deadline)
        return chunk


def _response_socket(response):
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    return getattr(raw, "_sock", None)


def _assert_safe_download_url(url: str, *, allow_file: bool) -> None:
    scheme = urlsplit(url).scheme.lower()
    if scheme == "https" or (allow_file and scheme == "file"):
        return
    raise PixelupError(
        ErrorCode.MODEL_DOWNLOAD_FAILED,
        "Refusing an insecure model download URL; HTTPS is required.",
        details={"url": url},
    )


def _remaining_download_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Model acquisition exceeded its total timeout.")
    return remaining


def _raise_if_cancelled(should_cancel: CancelCheck | None) -> None:
    if should_cancel and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")


def _check_acquisition(deadline: float | None, should_cancel: CancelCheck | None) -> None:
    _raise_if_cancelled(should_cancel)
    if deadline is not None:
        _remaining_download_seconds(deadline)


def _acquisition_timeout_seconds(info: ModelInfo, minimum_timeout: float) -> float:
    """Use the caller's floor, extending it for pinned large artifacts.

    128 KiB/s is intentionally conservative for a healthy but slow connection. The
    same resulting bound covers transport, verification, durable staging, and publish.
    """
    if info.expected_size is None:
        return float(minimum_timeout)
    size_bound = info.expected_size / MIN_DOWNLOAD_RATE_BYTES_PER_SECOND
    if size_bound <= minimum_timeout:
        return float(minimum_timeout)
    return size_bound + LARGE_DOWNLOAD_FINALIZATION_SECONDS


def _sync_staged_file(
    path: Path, deadline: float, should_cancel: CancelCheck | None
) -> None:
    _check_acquisition(deadline, should_cancel)
    # Open with write access because Windows FlushFileBuffers, which backs fsync,
    # requires it even though the completed staged bytes are no longer modified.
    with path.open("r+b") as file:
        os.fsync(file.fileno())
    _check_acquisition(deadline, should_cancel)


def _sync_directory_best_effort(path: Path) -> None:
    """Persist the rename on platforms that permit directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows cannot fsync directories and macOS may treat it as a no-op.
            pass
    finally:
        os.close(descriptor)


def _remove_staged_file(path: Path, model: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("model.temp_cleanup_failed", model=model, path=str(path), reason=str(exc))


def _acquire_download_lock(
    lock: FileLock,
    model: str,
    lock_timeout: int,
    on_waiting: WaitingCallback | None,
    should_cancel: CancelCheck | None = None,
) -> None:
    start = time.monotonic()
    if lock_timeout == 0:
        try:
            lock.acquire(timeout=0)
            return
        except Timeout as exc:
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                f"Timed out waiting for model download lock for '{model}'.",
                details={"model": model, "lock_timeout": lock_timeout},
            ) from exc
    while True:
        if should_cancel and should_cancel():
            raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
        elapsed = time.monotonic() - start
        remaining = lock_timeout - elapsed
        if remaining <= 0:
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                f"Timed out waiting for model download lock for '{model}'.",
                details={"model": model, "lock_timeout": lock_timeout},
            )
        try:
            lock.acquire(timeout=min(1.0, remaining))
            return
        except Timeout:
            if on_waiting:
                on_waiting(model, time.monotonic() - start)


def _model_file_present(path: Path) -> bool:
    # Presence is trust: a non-empty file at the target path was verified once at
    # download (or placed by the user), so the download is skipped and the file is not
    # re-hashed — see require_model_present. A zero-byte leftover does not count.
    return path.is_file() and path.stat().st_size > 0


def _download_result(info: ModelInfo, path: Path, status: str) -> dict[str, object]:
    return {
        "name": info.name,
        "status": status,
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def _lock_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _response_size(content_length: str | None, expected_size: int | None) -> int | None:
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            pass
    return expected_size


def _should_report_download(
    bytes_done: int,
    bytes_total: int | None,
    last_reported: int,
    last_percent: int,
) -> bool:
    if bytes_total and bytes_done >= bytes_total:
        return True
    if bytes_done - last_reported >= DOWNLOAD_REPORT_BYTES:
        return True
    if bytes_total:
        percent = bytes_done * 100 // bytes_total
        return percent > last_percent
    return False


def _sha256(
    path: Path,
    *,
    deadline: float | None = None,
    should_cancel: CancelCheck | None = None,
) -> str:
    _check_acquisition(deadline, should_cancel)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            _check_acquisition(deadline, should_cancel)
            digest.update(chunk)
    _check_acquisition(deadline, should_cancel)
    return digest.hexdigest()
