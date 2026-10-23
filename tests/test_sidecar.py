import errno
import json
import os
import re
import threading
from pathlib import Path

import pytest
from PIL import Image

from pixelup import sidecar as sidecar_module
from pixelup.errors import PixelupError
from pixelup.output_cleanup import output_cleanup
from pixelup.paths import OutputFormat
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions


def _write_sample_sidecar(
    tmp_path: Path, *, read_permissions: int = 0o444
) -> tuple[Path, dict]:
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "source-realesr-general-x4v3-4x.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    output_path.write_bytes(b"image")

    published = write_sidecar(
        input_path=input_path,
        output_path=output_path,
        options=UpscaleOptions(
            input_path=input_path,
            output_path=output_path,
            model="realesr-general-x4v3",
            scale=4,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            fp32=False,
            denoise_strength=1.0,
            alpha_mode="realesrgan",
            gpu_id=None,
            device="auto",
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            strip_metadata=False,
            target_profile=None,
        ),
        result={
            "input_size": [1, 1],
            "input_sha256": "ab" * 32,
            "input_size_bytes": 69,
            "output_size": [4, 4],
            "format": "png",
            "model": "realesr-general-x4v3",
            "scale": 4,
            "ms": 10,
            "source_read_permissions": read_permissions,
        },
        warnings=[],
    )
    return published.path, json.loads(published.path.read_text())


def test_sidecar_omits_private_paths(tmp_path: Path) -> None:
    path, payload = _write_sample_sidecar(tmp_path)
    serialized = json.dumps(payload)
    assert path.name == "source-realesr-general-x4v3-4x.json"
    assert payload["input"]["filename"] == "source.png"
    assert payload["output"]["filename"] == "source-realesr-general-x4v3-4x.png"
    assert str(tmp_path) not in serialized


def test_sidecar_carries_its_format_version(tmp_path: Path) -> None:
    _path, payload = _write_sample_sidecar(tmp_path)
    assert payload["format_version"] == 1


def test_sidecar_records_the_input_facts_read_when_the_job_started(tmp_path: Path) -> None:
    _, payload = _write_sample_sidecar(tmp_path)
    # The input on disk is a different file by now; the sidecar keeps what the job read.
    assert payload["input"]["sha256"] == "ab" * 32
    assert payload["input"]["size_bytes"] == 69


def test_sidecar_created_at_is_canonical_utc(tmp_path: Path) -> None:
    _, payload = _write_sample_sidecar(tmp_path)
    # Internal/serialized timestamps: ISO-8601, exactly 3 fractional digits, Z suffix.
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        payload["created_at_utc"],
    )


def test_sidecar_does_not_replace_an_existing_file(tmp_path: Path) -> None:
    sidecar, _payload = _write_sample_sidecar(tmp_path)
    original = sidecar.read_bytes()

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_exists"
    assert sidecar.read_bytes() == original


def test_sidecar_write_failure_without_hard_links_leaves_no_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disk_full(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    def unsupported_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr("pixelup.sidecar.os.fsync", disk_full)
    monkeypatch.setattr("pixelup.output_reservation.os.link", unsupported_link)

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_unwritable"
    assert not (tmp_path / "source-realesr-general-x4v3-4x.json").exists()


def test_sidecar_fsync_failure_happens_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "source-realesr-general-x4v3-4x.json"

    def fail_flush(_descriptor: int) -> None:
        assert not sidecar.exists()
        raise OSError("flush failed")

    monkeypatch.setattr("pixelup.sidecar.os.fsync", fail_flush)
    monkeypatch.setattr(
        sidecar_module, "publish_complete",
        lambda *_args: pytest.fail("must not publish unsynced bytes"),
    )
    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)
    assert excinfo.value.code == "output_unwritable"
    assert not sidecar.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_sidecar_success_verification_rejects_an_exact_boundary_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "source-realesr-general-x4v3-4x.json"
    real_is_current = sidecar_module.published_file_is_current

    def replace_before_verification(claim) -> bool:
        winner = tmp_path / "winner.tmp"
        winner.write_bytes(b"external winner")
        os.replace(winner, sidecar)
        return real_is_current(claim)

    monkeypatch.setattr(
        "pixelup.sidecar.published_file_is_current", replace_before_verification
    )

    with pytest.raises(PixelupError) as excinfo:
        _write_sample_sidecar(tmp_path)

    assert excinfo.value.code == "output_exists"
    assert sidecar.read_bytes() == b"external winner"


def test_quit_removes_a_sidecar_still_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A forced exit must not leave a half-written sidecar: while its bytes are
    # written it belongs to the quit like a partial image does.
    entered = threading.Event()
    release = threading.Event()
    real_fsync = sidecar_module.os.fsync
    errors: list[BaseException] = []

    def held_fsync(descriptor: int) -> None:
        entered.set()
        assert release.wait(5)
        real_fsync(descriptor)

    monkeypatch.setattr(sidecar_module.os, "fsync", held_fsync)

    def write() -> None:
        try:
            _write_sample_sidecar(tmp_path)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=write)
    worker.start()
    try:
        assert entered.wait(5)
        output_cleanup.begin_shutdown()
        assert output_cleanup._settled.wait(5)
        if os.name != "nt":
            assert list(tmp_path.glob("*.json")) == []
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert list(tmp_path.glob("*.json")) == []



@pytest.mark.parametrize("hard_links", [True, False])
def test_reader_at_publication_observes_the_complete_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    if not hard_links:
        def unsupported(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")
        monkeypatch.setattr("pixelup.output_publication.os.link", unsupported)
    publish = sidecar_module.publish_complete
    observed: list[dict] = []

    def publish_and_read(stage: Path, target: Path):
        assert not target.exists()
        staged_bytes = stage.read_bytes()
        claim = publish(stage, target)
        # This observer runs before the sidecar writer acknowledges publication.
        assert target.read_bytes() == staged_bytes
        observed.append(json.loads(target.read_bytes()))
        return claim

    monkeypatch.setattr(sidecar_module, "publish_complete", publish_and_read)
    _path, payload = _write_sample_sidecar(tmp_path)
    assert observed == [payload]
    assert payload["input"]["sha256"] == "ab" * 32
    assert payload["status"] == "success"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary permission bits")
@pytest.mark.parametrize("read_permissions", [0o400, 0o440, 0o404, 0o444])
def test_sidecar_does_not_broaden_source_read_permissions(
    tmp_path: Path, read_permissions: int
) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.write_bytes(b"")
    ordinary_reads = ordinary.stat().st_mode & 0o444

    path, _payload = _write_sample_sidecar(tmp_path, read_permissions=read_permissions)

    assert path.stat().st_mode & 0o444 == ordinary_reads & read_permissions
    assert path.stat().st_mode & 0o7111 == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary permission bits")
def test_sidecar_staging_is_private_before_any_json_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fdopen = sidecar_module.os.fdopen
    observed: list[int] = []

    class ObservedWriter:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def write(self, data):
            status = os.fstat(self.file.fileno())
            assert status.st_size == 0
            assert status.st_mode & 0o077 == 0
            observed.append(len(data))
            return self.file.write(data)

    monkeypatch.setattr(
        sidecar_module.os, "fdopen", lambda *a, **kw: ObservedWriter(fdopen(*a, **kw))
    )
    _write_sample_sidecar(tmp_path)
    assert len(observed) == 1
    assert observed[0] > 0
