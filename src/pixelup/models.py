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
from uuid import uuid4

from filelock import FileLock, Timeout

from pixelup.errors import ErrorCode, PixelupError

DownloadCallback = Callable[[str, int, int | None], None]
WaitingCallback = Callable[[str, float], None]

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
GFPGAN_RELEASES = "https://github.com/TencentARC/GFPGAN/releases/download"
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


ALL_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        "RealESRGAN_x4plus",
        "x4plus",
        "RealESRGAN_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        67040989,
    ),
    ModelInfo(
        "RealESRNet_x4plus",
        "x4plusnet",
        "RealESRNet_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.1/RealESRNet_x4plus.pth",
        67040989,
    ),
    ModelInfo(
        "RealESRGAN_x2plus",
        "x2plus",
        "RealESRGAN_x2plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        67061725,
    ),
    ModelInfo(
        "RealESRGAN_x4plus_anime_6B",
        "anime",
        "RealESRGAN_x4plus_anime_6B.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        17938799,
    ),
    ModelInfo(
        "realesr-animevideov3",
        "animevideo",
        "realesr-animevideov3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        2504012,
    ),
    ModelInfo(
        "realesr-general-x4v3",
        "general",
        "realesr-general-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        4885111,
    ),
    ModelInfo(
        "realesr-general-wdn-x4v3",
        None,
        "realesr-general-wdn-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        4885111,
        listed=False,
    ),
    ModelInfo(
        "GFPGANv1.4",
        "gfpgan",
        "GFPGANv1.4.pth",
        f"{GFPGAN_RELEASES}/v1.3.4/GFPGANv1.4.pth",
        348632874,
    ),
)

KNOWN_MODELS: tuple[ModelInfo, ...] = tuple(model for model in ALL_MODELS if model.listed)
_MODEL_BY_NAME = {model.name: model for model in ALL_MODELS}
_ALIAS_BY_NAME = {model.name: model.alias for model in ALL_MODELS if model.alias}


def model_short_name(name: str) -> str:
    return _ALIAS_BY_NAME.get(name, name)


def all_model_names(*, include_unlisted: bool = False) -> list[str]:
    models = ALL_MODELS if include_unlisted else KNOWN_MODELS
    return [model.name for model in models]


def known_model(name: str) -> ModelInfo | None:
    return _MODEL_BY_NAME.get(name)


def model_file(models_dir: Path, name: str) -> Path:
    info = known_model(name)
    filename = info.filename if info else f"{name}.pth"
    return models_dir / filename


def model_present(models_dir: Path, name: str) -> bool:
    path = model_file(models_dir, name)
    return path.is_file() and path.stat().st_size > 0


def model_record(models_dir: Path, info: ModelInfo) -> dict[str, object]:
    path = models_dir / info.filename
    present = path.is_file()
    return {
        "name": info.name,
        "alias": info.alias,
        "present": present,
        "size_bytes": path.stat().st_size if present else None,
    }


def list_model_records(models_dir: Path, names: list[str] | None = None) -> list[dict[str, object]]:
    if not names:
        return [model_record(models_dir, info) for info in KNOWN_MODELS]
    records: list[dict[str, object]] = []
    for name in names:
        info = known_model(name)
        if info is None:
            path = model_file(models_dir, name)
            present = path.is_file()
            records.append(
                {
                    "name": name,
                    "alias": None,
                    "present": present,
                    "size_bytes": path.stat().st_size if present else None,
                }
            )
        else:
            records.append(model_record(models_dir, info))
    return records


def require_model_present(
    models_dir: Path,
    name: str,
) -> Path:
    path = model_file(models_dir, name)
    if not path.is_file():
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{name}' is not present in the models directory.",
            hint=(
                "Run 'pixelup models download MODEL', use --auto-download, "
                "or place the .pth file there."
            ),
            details={"model": name, "models_dir": str(models_dir), "path": str(path)},
        )
    verify_model_file(path, known_model(name))
    return path


def download_models(
    models_dir: Path,
    names: list[str],
    *,
    download_timeout: int,
    lock_timeout: int,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
) -> list[dict[str, object]]:
    return [
        download_model(
            models_dir,
            name,
            download_timeout=download_timeout,
            lock_timeout=lock_timeout,
            on_download=on_download,
            on_waiting=on_waiting,
        )
        for name in names
    ]


def download_model(
    models_dir: Path,
    name: str,
    *,
    download_timeout: int,
    lock_timeout: int,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
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
    )


def download_model_info(
    models_dir: Path,
    info: ModelInfo,
    *,
    download_timeout: int,
    lock_timeout: int,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
) -> dict[str, object]:
    if download_timeout <= 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--download-timeout must be positive.")
    if lock_timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--lock-timeout must be 0 or greater.")
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
    if _model_file_is_valid(target, info):
        return _download_result(info, target, "present")

    lock = FileLock(str(locks_dir / f"{_lock_name(info.name)}.lock"))
    _acquire_download_lock(lock, info.name, lock_timeout, on_waiting)
    try:
        if _model_file_is_valid(target, info):
            return _download_result(info, target, "present")
        temp_path = models_dir / f".{info.filename}.{os.getpid()}.{uuid4().hex}.tmp"
        try:
            _download_to_temp(
                info,
                temp_path,
                download_timeout=download_timeout,
                on_download=on_download,
            )
            verify_model_file(temp_path, info)
            os.replace(temp_path, target)
        except PixelupError:
            temp_path.unlink(missing_ok=True)
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise PixelupError(
                ErrorCode.MODEL_DOWNLOAD_FAILED,
                f"Could not download model '{info.name}'.",
                details={"model": info.name, "url": info.url, "reason": str(exc)},
            ) from exc
    finally:
        lock.release()
    return _download_result(info, target, "downloaded")


def verify_present_models(models_dir: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for info in ALL_MODELS:
        path = models_dir / info.filename
        if not path.is_file():
            continue
        results.append(verify_model_file(path, info))
    return results


def verify_model_file(path: Path, info: ModelInfo | None = None) -> dict[str, object]:
    if not path.is_file():
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            "Model file is missing.",
            details={"path": str(path)},
        )
    size = path.stat().st_size
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
) -> None:
    assert info.url is not None
    bytes_done = 0
    last_reported = 0
    last_percent = -1
    with urlopen(info.url, timeout=download_timeout) as response:
        bytes_total = _response_size(response.headers.get("Content-Length"), info.expected_size)
        with temp_path.open("wb") as output:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
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


def _model_file_is_valid(path: Path, info: ModelInfo) -> bool:
    if not path.is_file():
        return False
    try:
        verify_model_file(path, info)
    except PixelupError:
        return False
    return True


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
