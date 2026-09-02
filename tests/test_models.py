import hashlib
import re
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from filelock import FileLock

import pixelup.models as models_module
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
    info = ModelInfo("model", "model.pth", None, expected_size=10)

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
    assert re.fullmatch(r"local-model-[A-Za-z0-9_-]{21}\.tmp", captured[-1])


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


def test_forced_download_replaces_present_file_only_after_verification(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "local-model.pth"
    target.write_bytes(b"old verified model")
    source = tmp_path / "source.pth"
    source.write_bytes(b"new verified model")
    info = ModelInfo(
        "local-model",
        target.name,
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
        checksum_sha256=_sha256_hex(source.read_bytes()),
    )

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
        force=True,
    )

    assert result["status"] == "downloaded"
    assert target.read_bytes() == b"new verified model"


def test_failed_forced_download_preserves_the_present_file(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "local-model.pth"
    target.write_bytes(b"old verified model")
    source = tmp_path / "source.pth"
    source.write_bytes(b"untrusted replacement")
    info = ModelInfo(
        "local-model",
        target.name,
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
        checksum_sha256=_sha256_hex(b"different trusted bytes"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(
            models_dir,
            info,
            download_timeout=10,
            lock_timeout=1,
            force=True,
        )

    assert excinfo.value.code == "model_corrupt"
    assert target.read_bytes() == b"old verified model"
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_does_not_require_lock_directory_for_present_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "local-model.pth"
    target.write_bytes(b"already present")
    info = ModelInfo("local-model", target.name, "https://example.com/model.pth")
    original_mkdir = Path.mkdir

    def reject_lock_directory(path: Path, *args: object, **kwargs: object) -> None:
        if path == models_dir / ".locks":
            raise OSError("read-only models directory")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reject_lock_directory)

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
    )

    assert result["status"] == "present"
    assert not (models_dir / ".locks").exists()


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


def test_download_model_info_refuses_https_redirect_to_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedirectingOpener:
        calls = 0

        def open(self, url: str, *, timeout: float):
            self.calls += 1
            headers = Message()
            headers["Location"] = "http://example.com/model.pth"
            raise HTTPError(url, 302, "Found", headers, BytesIO())

    opener = RedirectingOpener()
    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: opener)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/start",
        expected_size=7,
        checksum_sha256=_sha256_hex(b"weights"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10, lock_timeout=1)

    assert excinfo.value.code == "model_download_failed"
    assert "HTTPS is required" in excinfo.value.user_message
    assert opener.calls == 1
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_enforces_total_download_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    observed_timeouts: list[float] = []

    class SlowResponse:
        headers = {"Content-Length": "1"}
        fp = None

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://example.com/model.pth"

        def read(self, _size: int) -> bytes:
            clock[0] = 2.0
            return b"x"

    class SlowOpener:
        def open(self, _url: str, *, timeout: float) -> SlowResponse:
            observed_timeouts.append(timeout)
            return SlowResponse()

    monkeypatch.setattr(models_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: SlowOpener())
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=1, lock_timeout=1)

    assert excinfo.value.code == "model_download_failed"
    assert "total timeout" in str(excinfo.value.details["reason"])
    assert observed_timeouts == [1.0]
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_rejects_advertised_size_above_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedResponse:
        headers = {"Content-Length": "2"}
        fp = None

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://example.com/model.pth"

        def read(self, _size: int) -> bytes:
            raise AssertionError("an advertised oversize response must not be read")

    class OversizedOpener:
        def open(self, _url: str, *, timeout: float) -> OversizedResponse:
            assert timeout > 0
            return OversizedResponse()

    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: OversizedOpener())
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10, lock_timeout=1)

    assert excinfo.value.code == "model_download_failed"
    assert excinfo.value.details["advertised_size_bytes"] == 2
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_aborts_stream_above_pin_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedStreamResponse:
        headers: dict[str, str] = {}
        fp = None

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://example.com/model.pth"

        def read(self, _size: int) -> bytes:
            return b"xx"

    class OversizedStreamOpener:
        def open(self, _url: str, *, timeout: float) -> OversizedStreamResponse:
            assert timeout > 0
            return OversizedStreamResponse()

    monkeypatch.setattr(
        models_module,
        "build_opener",
        lambda *_handlers: OversizedStreamOpener(),
    )
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10, lock_timeout=1)

    assert excinfo.value.code == "model_download_failed"
    assert excinfo.value.details["received_size_bytes"] == 2
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_cancels_during_blocking_connection_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = models_module.threading.Event()
    release = models_module.threading.Event()
    closed = models_module.threading.Event()

    class LateResponse:
        def close(self) -> None:
            closed.set()

    class BlockingOpener:
        def open(self, _url: str, *, timeout: float) -> LateResponse:
            assert timeout > 0
            started.set()
            release.wait(5)
            return LateResponse()

    monkeypatch.setattr(models_module, "DOWNLOAD_POLL_SECONDS", 0.01)
    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: BlockingOpener())
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    try:
        with pytest.raises(PixelupError) as excinfo:
            download_model_info(
                models_dir,
                info,
                download_timeout=10,
                lock_timeout=1,
                should_cancel=started.is_set,
            )
    finally:
        release.set()

    assert excinfo.value.code == "job_cancelled"
    assert closed.wait(1), "a response returned after cancellation must be closed"
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_download_model_info_deadline_abandons_blocking_connection_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = models_module.threading.Event()
    closed = models_module.threading.Event()

    class LateResponse:
        def close(self) -> None:
            closed.set()

    class BlockingOpener:
        def open(self, _url: str, *, timeout: float) -> LateResponse:
            assert 0 < timeout <= 0.051
            release.wait(5)
            return LateResponse()

    monkeypatch.setattr(models_module, "DOWNLOAD_POLL_SECONDS", 0.01)
    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: BlockingOpener())
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    try:
        with pytest.raises(PixelupError) as excinfo:
            download_model_info(models_dir, info, download_timeout=0.05, lock_timeout=1)
    finally:
        release.set()

    assert excinfo.value.code == "model_download_failed"
    assert "total timeout" in str(excinfo.value.details["reason"])
    assert closed.wait(1), "a response returned after the deadline must be closed"
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_verify_model_file_honors_cancellation_during_hash(tmp_path: Path) -> None:
    content = b"x" * (2 * 1024 * 1024)
    path = tmp_path / "model.pth"
    path.write_bytes(content)
    info = ModelInfo(
        "local-model",
        "model.pth",
        None,
        expected_size=len(content),
        checksum_sha256=_sha256_hex(content),
    )
    calls = 0

    def should_cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(path, info, should_cancel=should_cancel)

    assert excinfo.value.code == "job_cancelled"


def test_download_read_poll_honors_cancellation_while_socket_is_stalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StalledSocket:
        def pending(self) -> int:
            return 0

    class Raw:
        _sock = StalledSocket()

    class File:
        raw = Raw()

    class StalledResponse:
        fp = File()

        def read(self, _size: int) -> bytes:
            raise AssertionError("a stalled socket must not be read before it is ready")

    polls = 0

    def should_cancel() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    monkeypatch.setattr(models_module.select, "select", lambda *_args: ([], [], []))

    with pytest.raises(PixelupError) as excinfo:
        models_module._read_download_chunk(
            StalledResponse(),
            deadline=models_module.time.monotonic() + 60,
            should_cancel=should_cancel,
        )

    assert excinfo.value.code == "job_cancelled"


def test_download_read_timeout_returns_to_cancellation_poll() -> None:
    timeouts: list[float] = []

    class PartialTlsSocket:
        def pending(self) -> int:
            return 1

        def settimeout(self, timeout: float) -> None:
            timeouts.append(timeout)

    class Raw:
        _sock = PartialTlsSocket()

    class File:
        raw = Raw()

    class PartialTlsResponse:
        fp = File()
        reads = 0

        def read1(self, _size: int) -> bytes:
            self.reads += 1
            raise TimeoutError("partial TLS record")

        def read(self, _size: int) -> bytes:
            raise AssertionError("read1 should be used when available")

    cancellation_checks = 0

    def should_cancel() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 2

    response = PartialTlsResponse()
    with pytest.raises(PixelupError) as excinfo:
        models_module._read_download_chunk(
            response,
            deadline=models_module.time.monotonic() + 60,
            should_cancel=should_cancel,
        )

    assert excinfo.value.code == "job_cancelled"
    assert response.reads == 1
    assert timeouts == [models_module.DOWNLOAD_POLL_SECONDS]


def test_download_model_info_checks_cancellation_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=len(content),
        checksum_sha256=_sha256_hex(content),
    )
    cancelled = False
    original_verify = models_module.verify_model_file

    def verify_then_cancel(
        path: Path,
        model_info: ModelInfo | None = None,
        *,
        deadline: float | None = None,
        should_cancel: models_module.CancelCheck | None = None,
    ) -> dict[str, object]:
        nonlocal cancelled
        assert deadline is not None
        result = original_verify(
            path,
            model_info,
            deadline=deadline,
            should_cancel=should_cancel,
        )
        cancelled = True
        return result

    monkeypatch.setattr(models_module, "verify_model_file", verify_then_cancel)

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(
            models_dir,
            info,
            download_timeout=10,
            lock_timeout=1,
            should_cancel=lambda: cancelled,
        )

    assert excinfo.value.code == "job_cancelled"
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_large_model_extends_the_whole_acquisition_timeout() -> None:
    small = ModelInfo("small", "small.pth", None, expected_size=1)
    large = ModelInfo("large", "large.pth", None, expected_size=348_632_874)

    assert models_module._acquisition_timeout_seconds(small, 600) == 600
    assert models_module._acquisition_timeout_seconds(large, 600) > 2_600


def test_verification_honors_the_acquisition_deadline(tmp_path: Path) -> None:
    content = b"weights"
    path = tmp_path / "model.pth"
    path.write_bytes(content)
    info = ModelInfo(
        "model",
        "model.pth",
        None,
        expected_size=len(content),
        checksum_sha256=_sha256_hex(content),
    )

    with pytest.raises(TimeoutError, match="acquisition exceeded its total timeout"):
        verify_model_file(path, info, deadline=models_module.time.monotonic() - 1)


def test_download_syncs_stage_before_publish_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    target = models_dir / "local-model.pth"
    info = ModelInfo(
        "local-model",
        target.name,
        source.resolve().as_uri(),
        expected_size=len(content),
        checksum_sha256=_sha256_hex(content),
    )
    order: list[str] = []
    real_replace = models_module.os.replace

    def sync_stage(
        path: Path,
        deadline: float,
        should_cancel: models_module.CancelCheck | None,
    ) -> None:
        assert path.is_file()
        assert deadline > models_module.time.monotonic()
        assert should_cancel is None
        order.append("sync-stage")

    def replace(source_path: Path, target_path: Path) -> None:
        order.append("publish")
        real_replace(source_path, target_path)

    def sync_directory(path: Path) -> None:
        assert path == models_dir
        assert target.is_file()
        order.append("sync-directory")

    monkeypatch.setattr(models_module, "_sync_staged_file", sync_stage)
    monkeypatch.setattr(models_module.os, "replace", replace)
    monkeypatch.setattr(models_module, "_sync_directory_best_effort", sync_directory)

    download_model_info(models_dir, info, download_timeout=10, lock_timeout=1)

    assert order == ["sync-stage", "publish", "sync-directory"]


def test_sync_staged_file_calls_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.tmp"
    path.write_bytes(b"weights")
    descriptors: list[int] = []
    monkeypatch.setattr(models_module.os, "fsync", descriptors.append)

    models_module._sync_staged_file(
        path,
        models_module.time.monotonic() + 60,
        should_cancel=None,
    )

    assert len(descriptors) == 1


def test_download_model_info_cleans_temp_after_unexpected_progress_failure(tmp_path: Path) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=len(content),
        checksum_sha256=_sha256_hex(content),
    )

    def fail_progress(_model: str, _done: int, _total: int | None) -> None:
        raise RuntimeError("progress sink failed")

    with pytest.raises(RuntimeError, match="progress sink failed"):
        download_model_info(
            models_dir,
            info,
            download_timeout=10,
            lock_timeout=1,
            on_download=fail_progress,
        )

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
