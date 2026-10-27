import hashlib
import re
import socket
import ssl
import threading
import time
from email.message import Message
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

import pixelup.models as models_module
from pixelup.errors import PixelupError
from pixelup.model_registry import GENERAL_MODEL, ModelInfo, model_info
from pixelup.models import download_model_info, require_model_present, verify_model_file


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _info(
    name: str,
    filename: str,
    url: str = "https://example.com/model.pth",
    *,
    expected_size: int,
    checksum_sha256: str = _sha256_hex(b""),
) -> ModelInfo:
    return ModelInfo(
        name=name,
        filename=filename,
        url=url,
        expected_size=expected_size,
        checksum_sha256=checksum_sha256,
        architecture=model_info(GENERAL_MODEL).architecture,
        upscaler=True,
    )


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
    info = _info("model", "model.pth", expected_size=10)

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(model_file, info)

    assert excinfo.value.code == "model_corrupt"


def test_verify_model_file_rejects_checksum_mismatch(tmp_path: Path) -> None:
    # Correct size, wrong bytes: a substituted same-size file must be caught by the
    # SHA-256 gate, which a size-only check could not detect.
    content = b"the real weights"
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(content)
    info = _info(
        "model",
        "model.pth",
        expected_size=len(content),
        checksum_sha256=_sha256_hex(b"a different blob of the same length!!"),
    )

    with pytest.raises(PixelupError) as excinfo:
        verify_model_file(model_file, info)

    assert excinfo.value.code == "model_corrupt"


def test_download_model_info_uses_temp_file_and_validates_size(tmp_path: Path) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = _info(
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
        on_download=lambda model, done, total: events.append((model, done, total)),
    )

    assert result["status"] == "downloaded"
    assert (models_dir / "local-model.pth").read_bytes() == b"downloaded weights"
    assert events[-1] == ("local-model", source.stat().st_size, source.stat().st_size)


def test_download_model_info_temp_file_uses_the_stem_shape_beside_the_target(
    tmp_path: Path,
) -> None:
    # The staged download's temp name is <stem>-<random>.tmp (target's stem, no
    # leading dot), created exclusively in the same directory as the eventual target,
    # and nothing else is made beside it.
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = _info(
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
        on_download=on_download,
    )

    assert captured
    assert re.fullmatch(r"local-model-[a-z0-9_]+\.tmp", captured[-1])
    assert sorted(path.name for path in models_dir.iterdir()) == ["local-model.pth"]


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
    info = _info(
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
    info = _info(
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
    info = _info(
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
            force=True,
        )

    assert excinfo.value.code == "model_corrupt"
    assert target.read_bytes() == b"old verified model"
    assert not list(models_dir.glob("*.tmp"))


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
    info = _info(
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
    info = _info(
        "local-model",
        "local-model.pth",
        "https://example.com/start",
        expected_size=7,
        checksum_sha256=_sha256_hex(b"weights"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10)

    assert excinfo.value.code == "model_download_failed"
    assert "HTTPS is required" in str(excinfo.value)
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
    info = _info(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=1)

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
    info = _info(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10)

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
    info = _info(
        "local-model",
        "local-model.pth",
        "https://example.com/model.pth",
        expected_size=1,
        checksum_sha256=_sha256_hex(b"x"),
    )

    with pytest.raises(PixelupError) as excinfo:
        download_model_info(models_dir, info, download_timeout=10)

    assert excinfo.value.code == "model_download_failed"
    assert excinfo.value.details["received_size_bytes"] == 2
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


@pytest.mark.parametrize("cancel", [True, False])
def test_connection_keeps_ownership_until_late_response_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    started, release, closed, settled = (threading.Event() for _ in range(4))
    errors: list[PixelupError] = []
    clock = [0.0]

    class LateResponse:
        def close(self) -> None:
            closed.set()

    class BlockingOpener:
        def open(self, _url: str, *, timeout: float) -> LateResponse:
            assert timeout == 10
            started.set()
            assert release.wait(5)
            return LateResponse()

    monkeypatch.setattr(models_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: BlockingOpener())
    models_dir = tmp_path / "models"
    info = _info("local-model", "local-model.pth", expected_size=1)

    def acquire() -> None:
        try:
            download_model_info(
                models_dir, info, download_timeout=10,
                should_cancel=lambda: cancel and started.is_set(),
            )
        except PixelupError as exc:
            errors.append(exc)
        finally:
            settled.set()

    worker = threading.Thread(target=acquire)
    worker.start()
    try:
        assert started.wait(5)
        clock[0] = 11
        assert not settled.is_set()
        assert not closed.is_set()
        assert list(models_dir.glob("*.tmp"))
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert closed.is_set()
    assert len(errors) == 1
    assert errors[0].code == ("job_cancelled" if cancel else "model_download_failed")
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_verify_model_file_honors_cancellation_during_hash(tmp_path: Path) -> None:
    content = b"x" * (2 * 1024 * 1024)
    path = tmp_path / "model.pth"
    path.write_bytes(content)
    info = _info(
        "local-model",
        "model.pth",
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


def test_fragmented_real_http_response_can_progress_beyond_old_poll_timeout() -> None:
    client, server = socket.socketpair()
    first_sent, continue_body = threading.Event(), threading.Event()

    def send() -> None:
        try:
            server.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nabc")
            first_sent.set()
            assert continue_body.wait(5)
            # A healthy fragment may arrive later than the removed 250 ms poll.
            time.sleep(0.3)
            server.sendall(b"def")
        finally:
            server.close()

    sender = threading.Thread(target=send)
    sender.start()
    response = HTTPResponse(client)
    try:
        assert first_sent.wait(5)
        response.begin()
        deadline = time.monotonic() + 5
        first = models_module._read_download_chunk(response, deadline, None)
        assert first == b"abc"
        continue_body.set()
        assert models_module._read_download_chunk(response, deadline, None) == b"def"
        assert models_module._read_download_chunk(response, deadline, None) == b""
    finally:
        continue_body.set()
        response.close()
        client.close()
        sender.join(5)
    assert not sender.is_alive()


@pytest.mark.skipif(not ssl.HAS_PSK, reason="the local TLS fixture requires PSK support")
def test_fragmented_tls_record_keeps_real_http_response_readable_below_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Public fixture key: this socketpair never leaves the test process.
    key = b"pixelup-public-tls-fragmentation-test-key"
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.minimum_version = ssl.TLSVersion.TLSv1_2
    server_context.maximum_version = ssl.TLSVersion.TLSv1_2
    server_context.set_ciphers("PSK-AES128-GCM-SHA256")
    server_context.set_psk_server_callback(
        lambda identity: key if identity == "pixelup-test" else b""
    )
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.check_hostname = False
    client_context.verify_mode = ssl.CERT_NONE
    client_context.minimum_version = ssl.TLSVersion.TLSv1_2
    client_context.maximum_version = ssl.TLSVersion.TLSv1_2
    client_context.set_ciphers("PSK-AES128-GCM-SHA256")
    client_context.set_psk_client_callback(lambda _hint: ("pixelup-test", key))
    raw_client, server = socket.socketpair()
    raw_client.settimeout(5)
    server.settimeout(5)
    client = client_context.wrap_socket(raw_client, do_handshake_on_connect=False)
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    encrypted_server = server_context.wrap_bio(incoming, outgoing, server_side=True)
    headers_sent, send_body, first_fragment, body_read_started = (
        threading.Event() for _ in range(4)
    )
    server_errors = []
    body = b"complete body from one fragmented TLS record"

    def flush_encrypted() -> None:
        while outgoing.pending:
            server.sendall(outgoing.read())

    def send() -> None:
        try:
            while True:
                try:
                    encrypted_server.do_handshake()
                except ssl.SSLWantReadError:
                    flush_encrypted()
                    received = server.recv(65536)
                    if not received:
                        raise EOFError("TLS client closed during the fixture handshake") from None
                    incoming.write(received)
                else:
                    flush_encrypted()
                    break
            encrypted_server.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii")
            )
            flush_encrypted()
            headers_sent.set()
            assert send_body.wait(5), "the client never finished reading the HTTP headers"
            assert encrypted_server.write(body) == len(body)
            record = outgoing.read()
            assert record[0] == 23  # TLS application data, with a five-byte record header.
            assert len(record) == 5 + int.from_bytes(record[3:5], "big")
            server.sendall(record[:7])
            first_fragment.set()
            assert body_read_started.wait(5), "the client never attempted its body read"
            # Start this delay only once the real TLS read has begun. A complete
            # record arrives after the removed 250 ms timeout but within its budget.
            time.sleep(0.35)
            server.sendall(record[7:])
        except BaseException as exc:
            server_errors.append(exc)
        finally:
            server.close()

    sender = threading.Thread(target=send)
    response = None
    sender.start()
    try:
        client.do_handshake()
        assert headers_sent.wait(5)
        response = HTTPResponse(client)
        response.begin()
        real_recv_into = client.recv_into

        def observed_recv_into(buffer, nbytes=0, flags=0):
            body_read_started.set()
            return real_recv_into(buffer, nbytes, flags)

        monkeypatch.setattr(client, "recv_into", observed_recv_into)
        send_body.set()
        assert first_fragment.wait(5)
        deadline = time.monotonic() + 5
        assert models_module._read_download_chunk(response, deadline, None) == body
        assert models_module._read_download_chunk(response, deadline, None) == b""
    finally:
        send_body.set()
        body_read_started.set()
        if response is not None:
            response.close()
        client.close()
        sender.join(5)
    assert not sender.is_alive()
    assert not server_errors, server_errors


def test_download_read_timeout_is_terminal_and_uses_remaining_deadline() -> None:
    timeouts: list[float] = []

    class PartialTlsSocket:
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

    response = PartialTlsResponse()
    with pytest.raises(TimeoutError, match="partial TLS record"):
        models_module._read_download_chunk(
            response, deadline=time.monotonic() + 60, should_cancel=None,
        )
    assert response.reads == 1
    assert len(timeouts) == 1
    assert 59 < timeouts[0] <= 60


def test_download_model_info_checks_cancellation_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"downloaded weights"
    source = tmp_path / "source.pth"
    source.write_bytes(content)
    models_dir = tmp_path / "models"
    info = _info(
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
        info: ModelInfo,
        *,
        deadline: float | None = None,
        should_cancel: models_module.CancelCheck | None = None,
    ) -> dict[str, object]:
        nonlocal cancelled
        assert deadline is not None
        result = original_verify(
            path,
            info,
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
            should_cancel=lambda: cancelled,
        )

    assert excinfo.value.code == "job_cancelled"
    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_large_model_extends_the_whole_acquisition_timeout() -> None:
    small = _info("small", "small.pth", expected_size=1)
    large = _info("large", "large.pth", expected_size=348_632_874)

    assert models_module._acquisition_timeout_seconds(small, 600) == 600
    assert models_module._acquisition_timeout_seconds(large, 600) > 2_600


def test_verification_honors_the_acquisition_deadline(tmp_path: Path) -> None:
    content = b"weights"
    path = tmp_path / "model.pth"
    path.write_bytes(content)
    info = _info(
        "model",
        "model.pth",
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
    info = _info(
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

    monkeypatch.setattr(models_module, "_sync_staged_file", sync_stage)
    monkeypatch.setattr(models_module.os, "replace", replace)

    download_model_info(models_dir, info, download_timeout=10)

    assert order == ["sync-stage", "publish"]


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
    info = _info(
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
            on_download=fail_progress,
        )

    assert not (models_dir / "local-model.pth").exists()
    assert not list(models_dir.glob("*.tmp"))


def test_model_readiness_distinguishes_missing_from_filesystem_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = GENERAL_MODEL
    assert not models_module.model_is_ready(tmp_path, name)
    failure = PermissionError(13, "denied", str(tmp_path))

    def unreadable(_path: Path):
        raise failure

    monkeypatch.setattr(Path, "stat", unreadable)
    with pytest.raises(PermissionError) as caught:
        models_module.model_is_ready(tmp_path, name)
    assert caught.value is failure


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_body_read_closes_response_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    closed = False
    cancellation_requested = False

    class Response:
        headers = {}

        def geturl(self):
            return "https://example.com/model.pth"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            nonlocal closed
            closed = True

        def read(self, _size):
            nonlocal cancellation_requested
            cancellation_requested = cancel
            raise TimeoutError("body read stopped")

    class Opener:
        def open(self, _url, *, timeout):
            return Response()

    monkeypatch.setattr(models_module, "build_opener", lambda *_handlers: Opener())
    info = _info("local-model", "local-model.pth", expected_size=1)
    with pytest.raises(PixelupError) as caught:
        download_model_info(
            tmp_path, info, download_timeout=10,
            should_cancel=lambda: cancellation_requested,
        )
    assert caught.value.code == ("job_cancelled" if cancel else "model_download_failed")
    assert closed
    assert not (tmp_path / info.filename).exists()
    assert not list(tmp_path.glob("*.tmp"))
