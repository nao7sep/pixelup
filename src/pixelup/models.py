from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import time
from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.model_registry import ModelInfo, model_info
from pixelup.session_log import log

DownloadCallback = Callable[[str, int, int | None], None]
CancelCheck = Callable[[], bool]

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_REPORT_BYTES = 5 * 1024 * 1024
MIN_DOWNLOAD_RATE_BYTES_PER_SECOND = 128 * 1024
LARGE_DOWNLOAD_FINALIZATION_SECONDS = 120
MAX_DOWNLOAD_REDIRECTS = 10
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def model_file(models_dir: Path, name: str) -> Path:
    return models_dir / model_info(name).filename


def require_model_present(
    models_dir: Path,
    name: str,
) -> Path:
    path = model_file(models_dir, name)
    if not _model_file_present(path):
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            Message.of("error.modelMissing", model=name),
            hint=Message("error.hintInstallModel"),
            details={"model": name, "models_dir": str(models_dir), "path": str(path)},
        )
    # A present model is trusted: it was verified once when PixelUp downloaded it (or
    # placed by the user). The managed-runtime-dependencies-conventions drop
    # re-verification after acquisition, and re-hashing here would re-read the whole
    # file before every job — hundreds of MB per face-enhance image — to guard against
    # a corruption that torch's weights_only load already turns into a loud failure.
    return path


def model_is_ready(models_dir: Path, name: str) -> bool:
    """Whether the artifact-local cache has a usable non-empty model file."""
    return _model_file_present(model_file(models_dir, name))


def download_model(
    models_dir: Path,
    name: str,
    *,
    download_timeout: float,
    on_download: DownloadCallback | None = None,
    should_cancel: CancelCheck | None = None,
    force: bool = False,
) -> dict[str, object]:
    return download_model_info(
        models_dir,
        model_info(name),
        download_timeout=download_timeout,
        on_download=on_download,
        should_cancel=should_cancel,
        force=force,
    )


def download_model_info(
    models_dir: Path,
    info: ModelInfo,
    *,
    download_timeout: float,
    on_download: DownloadCallback | None = None,
    should_cancel: CancelCheck | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download, verify and atomically publish one model file.

    Startup admits one writer per data root, and ``ModelManager`` refuses an
    artifact already running or queued, so no per-download lock is needed. Apps
    with independent data roots can still share a separately configured models
    folder: a duplicate download wastes bandwidth but publishes only a verified
    file atomically.
    """
    if download_timeout <= 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.downloadTimeoutInvalid"))
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            Message("error.modelsDirCreateFailed"),
            details={"models_dir": str(models_dir), "reason": str(exc)},
        ) from exc
    target = models_dir / info.filename
    if not force and _model_file_present(target):
        return _download_result(info, target, "present")

    acquisition_timeout = _acquisition_timeout_seconds(info, download_timeout)
    deadline = time.monotonic() + acquisition_timeout
    log.info(
        "model.download_started",
        model=info.name,
        url=info.url,
        timeout_seconds=acquisition_timeout,
    )
    temp_path: Path | None = None
    try:
        try:
            # Stage the download as a per-download-unique file INSIDE models_dir —
            # deliberately, not under a separate temp/ dir. The convention's intent
            # (a deletable staging area, unique name, verify there, then atomic
            # publish) is met: the staged file is removed on every failure path
            # below, and same-directory staging is precisely what makes os.replace
            # an atomic same-filesystem rename — a cross-volume temp/ could degrade
            # that to a copy.
            descriptor, temp_name = tempfile.mkstemp(
                dir=models_dir, prefix=f"{target.stem}-", suffix=".tmp"
            )
            os.close(descriptor)
            temp_path = Path(temp_name)
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
        except (HTTPError, URLError, TimeoutError, OSError, HTTPException) as exc:
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                Message.of("error.modelDownloadFailed", model=info.name),
                details={"model": info.name, "url": info.url, "reason": str(exc)},
            ) from exc
    finally:
        if temp_path is not None:
            _remove_staged_file(temp_path, info.name)
    result = _download_result(info, target, "downloaded")
    log.info("model.download_finished", model=info.name, size_bytes=result["size_bytes"])
    return result


def verify_model_file(
    path: Path,
    info: ModelInfo,
    *,
    deadline: float | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, object]:
    _check_acquisition(deadline, should_cancel)
    if not path.is_file():
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            Message("error.modelFileMissing"),
            details={"path": str(path)},
        )
    size = path.stat().st_size
    checksum = _sha256(path, deadline=deadline, should_cancel=should_cancel)
    ok = size == info.expected_size and checksum == info.checksum_sha256

    result = {
        "name": info.name,
        "path": str(path),
        "ok": ok,
        "size_bytes": size,
        "expected_size_bytes": info.expected_size,
        "checksum_sha256": checksum,
        "expected_checksum_sha256": info.checksum_sha256,
    }
    if not ok:
        raise PixelupError(
            ErrorCode.MODEL_CORRUPT,
            Message("error.modelVerificationFailed"),
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
    bytes_done = 0
    last_reported = 0
    last_percent = -1
    with _open_download_response(info.url, deadline, should_cancel) as response:
        advertised_size = _advertised_size(response.headers.get("Content-Length"))
        if advertised_size is not None and advertised_size > info.expected_size:
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                Message("error.modelTooLarge"),
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
                if next_bytes_done > info.expected_size:
                    raise PixelupError(
                        ErrorCode.MODEL_DOWNLOAD_FAILED,
                        Message("error.modelTooLargeStreaming"),
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
                    Message("error.tooManyRedirects"),
                    details={"url": url},
                ) from exc
            location = exc.headers.get("Location")
            exc.close()
            if not location:
                raise PixelupError(
                    ErrorCode.MODEL_DOWNLOAD_FAILED,
                    Message("error.redirectNoLocation"),
                    details={"url": current_url},
                ) from exc
            current_url = urljoin(current_url, location)
            _assert_safe_download_url(current_url, allow_file=False)
            continue
        try:
            _assert_safe_download_url(response.geturl(), allow_file=allow_file)
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
    """Keep setup owned by the installation worker until the call returns.

    urllib's timeout cannot interrupt every DNS/platform call. Cancellation remains
    requested until physical return; a late response is closed before settlement.
    """
    _check_acquisition(deadline, should_cancel)
    try:
        response = opener.open(url, timeout=_remaining_download_seconds(deadline))
    except Exception as exc:
        try:
            _raise_if_cancelled(should_cancel)
        except PixelupError:
            if isinstance(exc, HTTPError):
                exc.close()
            raise
        raise
    try:
        _check_acquisition(deadline, should_cancel)
    except BaseException:
        response.close()
        raise
    return response


def _read_download_chunk(response, deadline: float, should_cancel: CancelCheck | None) -> bytes:
    _check_acquisition(deadline, should_cancel)
    sock = _response_socket(response)
    if sock is not None:
        # Buffered HTTP/TLS streams can become unusable after a socket timeout.
        # Give this read the remaining acquisition budget and never retry it.
        sock.settimeout(_remaining_download_seconds(deadline))
    reader = getattr(response, "read1", None) or response.read
    try:
        chunk = reader(DOWNLOAD_CHUNK_BYTES)
    except Exception:
        _raise_if_cancelled(should_cancel)
        raise
    _check_acquisition(deadline, should_cancel)
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
        Message("error.insecureUrl"),
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


def _remove_staged_file(path: Path, model: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("model.temp_cleanup_failed", model=model, path=str(path), reason=str(exc))


def _model_file_present(path: Path) -> bool:
    # Presence is trust: a non-empty file at the target path was verified once at
    # download (or placed by the user), so the download is skipped and the file is not
    # re-hashed — see require_model_present. A zero-byte leftover does not count.
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0


def _download_result(info: ModelInfo, path: Path, status: str) -> dict[str, object]:
    return {
        "name": info.name,
        "status": status,
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def _advertised_size(content_length: str | None) -> int | None:
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            pass
    return None


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
