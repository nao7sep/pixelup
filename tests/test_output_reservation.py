from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.output_reservation import _output_lock_key, reserve_output_bundle


def _hold_reservation(
    output: Path,
    temp_dir: Path,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with reserve_output_bundle(output, temp_dir, timeout=5):
        ready.set()
        release.wait(5)


def test_symlinked_parent_and_direct_path_share_one_reservation(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(output_dir, target_is_directory=True)
    direct = output_dir / "result.png"
    alias = alias_dir / "result.png"

    assert _output_lock_key(direct) == _output_lock_key(alias)
    with reserve_output_bundle(direct, tmp_path / "temp", timeout=1):
        with pytest.raises(PixelupError) as excinfo:
            with reserve_output_bundle(alias, tmp_path / "temp", timeout=0):
                raise AssertionError("a physical output alias acquired a second lock")

    assert excinfo.value.code == "output_exists"


def test_format_variants_share_the_sidecar_reservation(tmp_path: Path) -> None:
    png = tmp_path / "result.png"
    jpg = tmp_path / "result.jpg"

    assert _output_lock_key(png) == _output_lock_key(jpg)
    with reserve_output_bundle(png, tmp_path / "temp", timeout=1):
        with pytest.raises(PixelupError):
            with reserve_output_bundle(jpg, tmp_path / "temp", timeout=0):
                raise AssertionError("two format variants acquired the shared sidecar")


def test_other_format_remnant_blocks_the_bundle(tmp_path: Path) -> None:
    (tmp_path / "result.jpg").write_bytes(b"existing")

    with pytest.raises(PixelupError) as excinfo:
        with reserve_output_bundle(tmp_path / "result.png", tmp_path / "temp", timeout=1):
            raise AssertionError("a bundle with an image remnant was reserved")

    assert excinfo.value.code == "output_exists"


def test_existing_sidecar_blocks_reservation_before_work_starts(tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PixelupError) as excinfo:
        with reserve_output_bundle(output, tmp_path / "temp", timeout=1):
            raise AssertionError("an occupied output bundle was reserved")

    assert excinfo.value.code == "output_exists"


def test_reservation_serializes_a_separate_process_through_a_symlink_alias(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(output_dir, target_is_directory=True)
    direct = output_dir / "result.png"
    alias = alias_dir / "result.png"
    temp_dir = tmp_path / "temp"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_reservation,
        args=(direct, temp_dir, ready, release),
    )
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(PixelupError) as excinfo:
            with reserve_output_bundle(alias, temp_dir, timeout=0):
                raise AssertionError("a second process acquired the physical alias")
        assert excinfo.value.code == "output_exists"
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
