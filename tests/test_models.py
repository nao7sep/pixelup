import hashlib
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


def test_download_model_info_skips_valid_existing_file(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = b"existing"
    target = models_dir / "local-model.pth"
    target.write_bytes(existing)
    source = tmp_path / "source.pth"
    source.write_bytes(b"different")
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=target.stat().st_size,
        checksum_sha256=_sha256_hex(existing),
    )

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
    )

    assert result["status"] == "present"
    assert target.read_bytes() == b"existing"


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
