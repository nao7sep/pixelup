from pathlib import Path

from pixelup.models import list_model_records, model_short_name, require_model_present


def test_model_short_aliases() -> None:
    assert model_short_name("RealESRGAN_x4plus") == "x4plus"
    assert model_short_name("RealESRGAN_x4plus_anime_6B") == "anime"
    assert model_short_name("custom-model") == "custom-model"


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
    assert require_model_present(tmp_path, "RealESRGAN_x4plus") == model_file

