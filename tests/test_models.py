import hashlib
import re
from pathlib import Path

import pytest
from filelock import FileLock

from pixelup.errors import PixelupError
from pixelup.models import ModelInfo, download_model_info, require_model_present, verify_model_file


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_require_model_present_accepts_custom_model_file(tmp_path: Path) -> None:
    custom_file = tmp_path / "custom-model.pth"
    custom_file.write_bytes(b"weights")

    assert require_model_present(tmp_path, "custom-model") == custom_file


def test_require_model_present_returns_known_model_without_rehashing(tmp_path: Path) -> None:
    # A known model carries a pinned checksum, but require_model_present must not
    # re-hash it at use-time (the convention drops re-verification after acquisition):
    # a present file is returned even when its bytes do not match the pin.
    model_path = tmp_path / "RealESRGAN_x4plus.pth"
    model_path.write_bytes(b"not the real weights")

    assert require_model_present(tmp_path, "RealESRGAN_x4plus") == model_path


def test_require_model_present_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(PixelupError) as excinfo:
        require_model_present(tmp_path, "RealESRGAN_x4plus")

    assert excinfo.value.code == "model_not_found"


def test_require_model_present_treats_empty_file_as_missing(tmp_path: Path) -> None:
    # A zero-byte leftover is not a usable model: presence requires a non-empty file,
    # so it reads as missing rather than being returned as present.
    (tmp_path / "RealESRGAN_x4plus.pth").write_bytes(b"")

    with pytest.raises(PixelupError) as excinfo:
        require_model_present(tmp_path, "RealESRGAN_x4plus")

    assert excinfo.value.code == "model_not_found"


def test_verify_model_file_rejects_wrong_size(tmp_path: Path) -> None:
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(b"small")
    info = ModelInfo("model", None, "model.pth", None, expected_size=10)

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(model_file, info)

    assert excinfo.value.code == "model_corrupt"


def test_verify_model_file_rejects_checksum_mismatch(tmp_path: Path) -> None:
    # Correct size, wrong bytes: a substituted same-size file must be caught by the
    # SHA-256 gate, which a size-only check could not detect.
    content = b"the real weights"
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(content)
    info = ModelInfo(
        "model",
        None,
        "model.pth",
        None,
        expected_size=len(content),
        checksum_sha256=_sha256_hex(b"a different blob of the same length!!"),
    )

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(model_file, info)

    assert excinfo.value.code == "model_corrupt"


def test_verify_model_file_requires_pinned_checksum_for_downloadable(tmp_path: Path) -> None:
    # A model PixelUp knows how to download (it has a url) must carry a pinned
    # checksum; without one, verification must not fall back to a size-only pass.
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(b"weights")
    info = ModelInfo(
        "downloadable",
        None,
        "model.pth",
        "https://example.com/model.pth",
        expected_size=7,
        checksum_sha256=None,
    )

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(model_file, info)

    assert excinfo.value.code == "internal_error"


def test_download_model_info_uses_temp_file_and_validates_size(tmp_path: Path) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
        checksum_sha256=_sha256_hex(content),
    )
    events: list[tuple[str, int, int | None]] = []

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
        on_download=lambda model, done, total: events.append((model, done, total)),
    )

    assert result["status"] == "downloaded"
    assert (models_dir / "local-model.pth").read_bytes() == b"downloaded weights"
    assert events[-1] == ("local-model", source.stat().st_size, source.stat().st_size)


def test_download_model_info_temp_file_uses_stem_nanoid_shape(tmp_path: Path) -> None:
    # The staged download's temp name is <stem>-<nanoid>.tmp (target's stem, no
    # leading dot, no pid segment), same directory as the eventual target.
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
        checksum_sha256=_sha256_hex(content),
    )
    captured: list[str] = []

    def on_download(model: str, done: int, total: int | None) -> None:
        captured.extend(path.name for path in models_dir.glob("*.tmp"))

    download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
        on_download=on_download,
    )

    assert captured
    assert re.fullmatch(r"local-model-[0-9a-f]{32}\.tmp", captured[-1])


def test_download_model_info_skips_present_file_without_rehashing(tmp_path: Path) -> None:
    # A non-empty file already at the target is trusted and the download skipped — even
    # when its bytes match neither the pinned size nor the checksum. Presence is trust;
    # the convention drops re-verification after acquisition, so no re-hash gates the
    # skip (a file is verified once, when PixelUp downloads it).
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "local-model.pth"
    target.write_bytes(b"whatever is already on disk")
    source = tmp_path / "source.pth"
    source.write_bytes(b"different")
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=999,  # deliberately wrong
        checksum_sha256=_sha256_hex(b"a totally different blob"),  # deliberately non-matching
    )

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
    )

    assert result["status"] == "present"
    assert target.read_bytes() == b"whatever is already on disk"


def test_download_model_info_removes_temp_on_verification_failure(tmp_path: Path) -> None:
    # A fully downloaded temp file whose bytes fail verification (wrong pinned
    # checksum) must be cleaned up: download_model_info raises model_corrupt, no
    # target file is published, and no leftover .tmp remains in models_dir. This is
    # the Deliver/fail-clean invariant on the most security-relevant path — a
    # substituted or corrupt download must never reach the cache.
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=len(content),
        checksum_sha256=_sha256_hex(b"a different blob of the same length"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(
            models_dir,
            info,
            download_timeout=10,
            lock_timeout=1,
        )

    assert excinfo.value.code == "model_corrupt"
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_cancels_while_waiting_for_lock(tmp_path: Path) -> None:
    # When the download lock is held by another process and the caller signals
    # cancellation, the wait must abort with JOB_CANCELLED instead of blocking
    # for the full lock_timeout.
    source = tmp_path / "source.pth"
    source.write_bytes(b"downloaded weights")
    models_dir = tmp_path / "models"
    locks_dir = models_dir / ".locks"
    locks_dir.mkdir(parents=True)
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
    )
    cancel_after = 2
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] >= cancel_after

    lock = FileLock(str(locks_dir / "local-model.lock"))
    with lock.acquire(timeout=1):
        with pytest.raises(PixelupError) as excinfo:
            download_model_info(
                models_dir,
                info,
                download_timeout=10,
                lock_timeout=600,
                should_cancel=should_cancel,
            )

    assert excinfo.value.code == "job_cancelled"
    assert not (models_dir / "local-model.pth").exists()


def test_download_model_info_lock_timeout_leaves_existing_state(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"downloaded weights")
    models_dir = tmp_path / "models"
    locks_dir = models_dir / ".locks"
    locks_dir.mkdir(parents=True)
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
    )

    lock = FileLock(str(locks_dir / "local-model.lock"))
    with lock.acquire(timeout=1):
        with pytest.raises(PixelupError) as excinfo:
            download_model_info(
                models_dir,
                info,
                download_timeout=10,
                lock_timeout=0,
            )

    assert excinfo.value.code == "model_download_failed"
    assert excinfo.value.details == {"model": "local-model", "lock_timeout": 0}
    assert not (models_dir / "local-model.pth").exists()
