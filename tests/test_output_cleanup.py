from __future__ import annotations

import ctypes
import errno
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from pixelup import imaging
from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import SourceMetadata, save_output_image
from pixelup.output_cleanup import _open_windows_output_handle, output_cleanup
from pixelup.paths import OutputFormat


@pytest.mark.parametrize(
    ("path", "flags", "native_path", "access", "disposition"),
    [
        (
            r"C:\photos\out.tmp",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            r"\\?\C:\photos\out.tmp",
            0x40000000,
            1,
        ),
        (r"\\server\share\out.tmp", os.O_RDONLY, r"\\?\UNC\server\share\out.tmp", 0x80000000, 3),
    ],
)
def test_windows_owned_handles_use_delete_sharing_and_transfer_once(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    flags: int,
    native_path: str,
    access: int,
    disposition: int,
) -> None:
    opened = []
    transferred = []
    closed = []

    class NativeCall:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    kernel = SimpleNamespace(
        CreateFileW=NativeCall(lambda *args: opened.append(args) or 123),
        CloseHandle=NativeCall(lambda handle: closed.append(handle) or 1),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel, raising=False)
    monkeypatch.setattr(os, "O_BINARY", 0x8000, raising=False)
    monkeypatch.setattr(os, "O_NOINHERIT", 0x80, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(
            open_osfhandle=lambda handle, mode: transferred.append((handle, mode)) or 456,
        ),
    )
    fake_path = SimpleNamespace(absolute=lambda: path)
    assert _open_windows_output_handle(fake_path, flags) == 456
    assert opened == [(native_path, access, 0x7, None, disposition, 0x80, None)]
    assert transferred == [(123, 0x8080)]
    assert closed == []  # descriptor owns the transferred handle


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary output mode contract")
@pytest.mark.parametrize("publication", ["hardlink", "exclusive_copy"])
def test_completed_output_keeps_ordinary_creation_mode_and_staging_is_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    reference = tmp_path / "ordinary.txt"
    reference.write_bytes(b"reference")
    expected_mode = reference.stat().st_mode & 0o777
    output = tmp_path / "out.png"
    original_save = Image.Image.save
    staged_modes = []

    def save(image, stream, **kwargs):
        staged_modes.append(os.fstat(stream.fileno()).st_mode & 0o777)
        original_save(image, stream, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)
    if publication == "exclusive_copy":
        monkeypatch.setattr(
            imaging.os,
            "link",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError(errno.EPERM, "unsupported")),
        )
    save_output_image(
        Image.new("RGB", (8, 4)),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )
    assert staged_modes == [expected_mode & 0o600]
    assert output.stat().st_mode & 0o777 == expected_mode


def test_staging_close_failure_preserves_primary_and_still_removes_owned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / "out-owned.tmp"
    primary = ValueError("encode failed")
    original_close = os.close
    closed = []

    def close(descriptor):
        original_close(descriptor)
        closed.append(descriptor)
        raise OSError(errno.EIO, "close cleanup failed")

    monkeypatch.setattr(imaging.os, "close", close)
    with pytest.raises(ValueError) as failure:
        with imaging.temp_file_guard(stage) as (stream, _mode):
            stream.write(b"partial")
            raise primary
    assert failure.value is primary
    assert len(closed) == 1
    assert any("close cleanup failed" in note for note in primary.__notes__)
    assert not stage.exists()


@pytest.mark.parametrize("stage", ["encode", "exclusive_copy"])
def test_quit_removes_only_active_owned_staging_and_partial_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    output = tmp_path / "out.png"
    foreign = tmp_path / "foreign-work.tmp"
    foreign.write_bytes(b"keep")
    entered = threading.Event()
    release = threading.Event()
    errors = []
    original_save = Image.Image.save
    original_copy = imaging._copy_staged

    def held_encode(image, stream, **kwargs):
        stream.write(b"partial")
        stream.flush()
        entered.set()
        assert release.wait(5)
        original_save(image, stream, **kwargs)

    def held_copy(source, target):
        target.write(b"partial")
        target.flush()
        entered.set()
        assert release.wait(5)
        original_copy(source, target)

    if stage == "encode":
        monkeypatch.setattr(Image.Image, "save", held_encode)
    else:
        monkeypatch.setattr(
            imaging.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(errno.EPERM, "unsupported")),
        )
        monkeypatch.setattr(imaging, "_copy_staged", held_copy)

    def save():
        try:
            save_output_image(
                Image.new("RGB", (8, 4)),
                output_path=output,
                output_format=OutputFormat.PNG,
                quality=95,
                background="white",
                source_metadata=SourceMetadata(),
                strip_metadata=True,
                target_profile=None,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=save)
    worker.start()
    try:
        assert entered.wait(5)
        output_cleanup.begin_shutdown()
        assert output_cleanup._settled.wait(5)
        if os.name != "nt":
            # POSIX removes directory entries now; Windows marks the shared
            # handles delete-pending until close (including process hard exit).
            assert not output.exists()
            assert list(tmp_path.glob("out-*.tmp")) == []
        assert foreign.read_bytes() == b"keep"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], PixelupError)
    assert errors[0].code == ErrorCode.JOB_CANCELLED
    assert not output.exists()
    assert list(tmp_path.glob("out-*.tmp")) == []


def test_publication_treats_einval_from_link_as_no_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows reports CreateHardLink on exFAT as EINVAL; publication falls back to
    # an exclusive claim instead of failing every output on such a drive.
    output = tmp_path / "out.png"
    monkeypatch.setattr(
        imaging.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(errno.EINVAL, "invalid")),
    )

    size = save_output_image(
        Image.new("RGB", (2, 2), "white"),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )

    assert size == (2, 2)
    assert output.is_file()
    assert list(tmp_path.glob("out-*.tmp")) == []
