from __future__ import annotations

from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.output_reservation import assert_output_bundle_available, remove_published_file


def test_other_format_remnant_blocks_the_bundle(tmp_path: Path) -> None:
    (tmp_path / "result.jpg").write_bytes(b"existing")

    with pytest.raises(PixelupError) as excinfo:
        assert_output_bundle_available(tmp_path / "result.png")

    assert excinfo.value.code == "output_exists"


def test_existing_sidecar_blocks_the_bundle_before_work_starts(tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PixelupError) as excinfo:
        assert_output_bundle_available(output)

    assert excinfo.value.code == "output_exists"


def test_a_free_bundle_is_available(tmp_path: Path) -> None:
    (tmp_path / "result-2.png").write_bytes(b"a different stem")

    assert_output_bundle_available(tmp_path / "result.png")


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
        assert_output_bundle_available(tmp_path / output_name)

    assert excinfo.value.code == "output_exists"
    assert occupied.read_bytes() == b"existing"


def test_case_variant_broken_symlink_occupies_the_normalized_bundle(
    tmp_path: Path, file_symlink_capability: None
) -> None:
    occupied = tmp_path / "RESULT.JPG"
    occupied.symlink_to(tmp_path / "missing.jpg")

    with pytest.raises(PixelupError):
        assert_output_bundle_available(tmp_path / "result.png")

    assert occupied.is_symlink()


def test_cleanup_removes_its_file_and_reports_one_already_gone(tmp_path: Path) -> None:
    path = tmp_path / "result.png"
    path.write_bytes(b"pixelup")

    remove_published_file(path)
    remove_published_file(path)

    assert list(tmp_path.iterdir()) == []
