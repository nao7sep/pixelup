from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from filelock import FileLock, Timeout

from pixelup.errors import ErrorCode, PixelupError
from pixelup.nanoid import nanoid
from pixelup.session_log import log

DownloadCallback = Callable[[str, int, int | None], None]
WaitingCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
GFPGAN_RELEASES = "https://github.com/TencentARC/GFPGAN/releases/download"
FACEXLIB_RELEASES = "https://github.com/xinntao/facexlib/releases/download"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_REPORT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    alias: str | None
    filename: str
    url: str | None
    expected_size: int | None = None
    checksum_sha256: str | None = None
    listed: bool = True


# Each model is pinned to an immutable upstream release artifact (the URL's tag)
# and to a SHA-256 of that artifact's bytes. The hashes were computed from the
# official releases (xinntao/Real-ESRGAN, TencentARC/GFPGAN, xinntao/facexlib) and
# confirmed byte-identical to what those pinned URLs serve; that is the trust
# anchor, since these old releases publish no upstream checksum of their own. A
# download is verified against its hash before it is cached (see verify_model_file),
# so a corrupted or substituted same-size file never reaches the cache; a file already
# on disk is then trusted and not re-hashed on use.
# These projects froze in 2022; each entry is already the latest of its model.
ALL_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        "RealESRGAN_x4plus",
        "x4plus",
        "RealESRGAN_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        67040989,
        checksum_sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
    ModelInfo(
        "RealESRNet_x4plus",
        "x4plusnet",
        "RealESRNet_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.1/RealESRNet_x4plus.pth",
        67040989,
        checksum_sha256="a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
    ),
    ModelInfo(
        "RealESRGAN_x2plus",
        "x2plus",
        "RealESRGAN_x2plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        67061725,
        checksum_sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    ),
    ModelInfo(
        "RealESRGAN_x4plus_anime_6B",
        "anime",
        "RealESRGAN_x4plus_anime_6B.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        17938799,
        checksum_sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
    ModelInfo(
        "realesr-animevideov3",
        "animevideo",
        "realesr-animevideov3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        2504012,
        checksum_sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
    ),
    ModelInfo(
        "realesr-general-x4v3",
        "general",
        "realesr-general-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        4885111,
        checksum_sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    ),
    ModelInfo(
        "realesr-general-wdn-x4v3",
        None,
        "realesr-general-wdn-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        4885111,
        checksum_sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        listed=False,
    ),
    ModelInfo(
        "GFPGANv1.4",
        "gfpgan",
        "GFPGANv1.4.pth",
        f"{GFPGAN_RELEASES}/v1.3.4/GFPGANv1.4.pth",
        348632874,
        checksum_sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
    ),
    ModelInfo(
        "facexlib-detection-retinaface-resnet50",
        None,
        "detection_Resnet50_Final.pth",
        f"{FACEXLIB_RELEASES}/v0.1.0/detection_Resnet50_Final.pth",
        109497761,
        checksum_sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        listed=False,
    ),
    ModelInfo(
        "facexlib-parsing-parsenet",
        None,
        "parsing_parsenet.pth",
        f"{FACEXLIB_RELEASES}/v0.2.2/parsing_parsenet.pth",
        85331193,
        checksum_sha256="3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2",
        listed=False,
    ),
)

KNOWN_MODELS: tuple[ModelInfo, ...] = tuple(model for model in ALL_MODELS if model.listed)
_MODEL_BY_NAME = {model.name: model for model in ALL_MODELS}


def known_model(name: str) -> ModelInfo | None:
    return _MODEL_BY_NAME.get(name)


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
                "Enable automatic model downloads, or place the .pth file in the models directory."
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
    download_timeout: int,
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
    download_timeout: int,
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
        locks_dir = models_dir / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.MODEL_DOWNLOAD_FAILED,
            "Could not create the models directory.",
            details={"models_dir": str(models_dir), "reason": str(exc)},
        ) from exc
    target = models_dir / info.filename
    if _model_file_present(target):
        return _download_result(info, target, "present")

    lock = FileLock(str(locks_dir / f"{_lock_name(info.name)}.lock"))
    _acquire_download_lock(lock, info.name, lock_timeout, on_waiting, should_cancel)
    try:
        if _model_file_present(target):
            return _download_result(info, target, "present")
        # Stage the download as a per-download-unique file INSIDE models_dir —
        # deliberately, not under a separate temp/ dir. The convention's intent
        # (a deletable staging area, unique name, verify there, then atomic
        # publish) is met: the staged file is removed on every failure path
        # below, and same-directory staging is precisely what makes os.replace
        # an atomic same-filesystem rename — a cross-volume temp/ could degrade
        # that to a copy. (Image-output staging uses temp/ in imaging.py; model
        # publish needs the same-fs guarantee.)
        temp_path = models_dir / f"{target.stem}-{nanoid()}.tmp"
        log.info("model.download_started", model=info.name, url=info.url)
        try:
            _download_to_temp(
                info,
                temp_path,
                download_timeout=download_timeout,
                on_download=on_download,
                should_cancel=should_cancel,
            )
            verify_model_file(temp_path, info)
            # not recorded: model weights are large binaries, re-fetchable from
            # their source and interchangeable with it — not hand-authored text the
            # app owns as state. Binaries are out of scope for the text backup, and
            # models/ is a binary-bearing directory excluded wholesale
            # (data-backup-conventions).
            os.replace(temp_path, target)
        except PixelupError as exc:
            temp_path.unlink(missing_ok=True)
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
            temp_path.unlink(missing_ok=True)
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
        lock.release()
    result = _download_result(info, target, "downloaded")
    log.info("model.download_finished", model=info.name, size_bytes=result["size_bytes"])
    return result


def verify_model_file(path: Path, info: ModelInfo | None = None) -> dict[str, object]:
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
    checksum = _sha256(path) if info and info.checksum_sha256 else None
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
    download_timeout: int,
    on_download: DownloadCallback | None,
    should_cancel: CancelCheck | None = None,
) -> None:
    assert info.url is not None
    bytes_done = 0
    last_reported = 0
    last_percent = -1
    with urlopen(info.url, timeout=download_timeout) as response:
        bytes_total = _response_size(response.headers.get("Content-Length"), info.expected_size)
        with temp_path.open("wb") as output:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                if should_cancel and should_cancel():
                    raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
                output.write(chunk)
                bytes_done += len(chunk)
                if _should_report_download(bytes_done, bytes_total, last_reported, last_percent):
                    if on_download:
                        on_download(info.name, bytes_done, bytes_total)
                    last_reported = bytes_done
                    if bytes_total:
                        last_percent = bytes_done * 100 // bytes_total
    if on_download and bytes_done != last_reported:
        on_download(info.name, bytes_done, bytes_total)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
