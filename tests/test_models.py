from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.models import (
    ModelInfo,
    all_model_names,
    download_model_info,
    list_model_records,
    model_short_name,
    require_model_present,
    verify_model_file,
    verify_present_models,
)


def test_model_short_aliases() -> None:
    assert model_short_name("RealESRGAN_x4plus") == "x4plus"
    assert model_short_name("RealESRGAN_x4plus_anime_6B") == "anime"
    assert model_short_name("custom-model") == "custom-model"


def test_all_model_names_can_include_unlisted_companion_models() -> None:
    assert "realesr-general-wdn-x4v3" not in all_model_names()
    assert "realesr-general-wdn-x4v3" in all_model_names(include_unlisted=True)


def test_model_records_report_presence(tmp_path: Path) -> None:
    model_file = tmp_path / "RealESRGAN_x4plus.pth"
    model_file.write_bytes(b"weights")

    records = list_model_records(tmp_path, ["RealESRGAN_x4plus"])

    assert records == [
        {
            "name": "RealESRGAN_x4plus",
            "alias": "x4plus",
            "present": True,
            "size_bytes": len(b"weights"),
        }
    ]

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


def test_verify_present_models_includes_unlisted_companion_models(tmp_path: Path) -> None:
    (tmp_path / "realesr-general-wdn-x4v3.pth").write_bytes(b"small")

    with pytest.raises(PixelupError) as excinfo:
        verify_present_models(tmp_path)

    assert excinfo.value.code == "model_corrupt"
    assert excinfo.value.details["model"]["name"] == "realesr-general-wdn-x4v3"


def test_download_model_info_uses_temp_file_and_validates_size(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"downloaded weights")
    models_dir = tmp_path / "models"
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=source.stat().st_size,
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
    target = models_dir / "local-model.pth"
    target.write_bytes(b"existing")
    source = tmp_path / "source.pth"
    source.write_bytes(b"different")
    info = ModelInfo(
        "local-model",
        None,
        "local-model.pth",
        source.resolve().as_uri(),
        expected_size=target.stat().st_size,
    )

    result = download_model_info(
        models_dir,
        info,
        download_timeout=10,
        lock_timeout=1,
    )

    assert result["status"] == "present"
    assert target.read_bytes() == b"existing"
