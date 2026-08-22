from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.output_reservation import (
    PublishedFile,
    _output_lock_key,
    remove_published_file,
    reserve_output_bundle,
)


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


@pytest.mark.parametrize(
    "output_name, occupied_name",
    [
        ("result.png", "result.JPG"),
        ("result.png", "RESULT.JSON"),
        ("r\u00e9sult.png", "re\u0301sult.webp"),
    ],
)
def test_actual_directory_entries_use_one_casefolded_nfc_bundle_identity(
    tmp_path: Path,
    output_name: str,
    occupied_name: str,
) -> None:
    occupied = tmp_path / occupied_name
    occupied.write_bytes(b"existing")

    with pytest.raises(PixelupError) as excinfo:
        with reserve_output_bundle(tmp_path / output_name, tmp_path / "temp", timeout=1):
            raise AssertionError("a normalized same-stem companion was missed")

    assert excinfo.value.code == "output_exists"
    assert occupied.read_bytes() == b"existing"


def test_case_variant_broken_symlink_occupies_the_normalized_bundle(tmp_path: Path) -> None:
    occupied = tmp_path / "RESULT.JPG"
    occupied.symlink_to(tmp_path / "missing.jpg")

    with pytest.raises(PixelupError):
        with reserve_output_bundle(tmp_path / "result.png", tmp_path / "temp", timeout=1):
            raise AssertionError("a broken symlink entry was missed")

    assert occupied.is_symlink()


@pytest.mark.parametrize("name", ["result.png", "result.json"])
def test_claimed_cleanup_restores_an_exact_boundary_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    path = tmp_path / name
    path.write_bytes(b"pixelup")
    identity = os.lstat(path)
    claim = PublishedFile(path, identity.st_dev, identity.st_ino)
    winner = tmp_path / "winner.tmp"
    winner.write_bytes(b"external winner")
    real_rename = os.rename
    replaced = False

    def replace_at_move(source: Path, hold: Path) -> None:
        nonlocal replaced
        if not replaced and Path(source) == path:
            replaced = True
            os.replace(winner, path)
        real_rename(source, hold)

    monkeypatch.setattr("pixelup.output_reservation.os.rename", replace_at_move)

    assert remove_published_file(claim) is False
    assert path.read_bytes() == b"external winner"
    assert list(tmp_path.glob("*.pixelup-hold")) == []


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
