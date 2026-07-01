import json
from pathlib import Path

import pytest

from pixelup.app_config import AppConfig, ensure_app_config, load_app_config, save_app_config
from pixelup.paths import OutputFormat


def test_app_config_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        max_concurrent_jobs=3,
        output_format=OutputFormat.WEBP,
        quality=82,
        tile=512,
        device="cpu",
        auto_download=False,
    )

    save_app_config(config, path)

    assert load_app_config(path) == config


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    assert load_app_config(tmp_path / "missing.json") == AppConfig()


def test_ensure_app_config_writes_defaults_on_first_run(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert not path.exists()

    created = ensure_app_config(path)

    assert created is True
    assert path.exists()
    # Written through save_app_config, so it round-trips back to the defaults.
    assert load_app_config(path) == AppConfig()


def test_ensure_app_config_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_app_config(AppConfig(quality=55, tile=256), path)
    before = path.read_text(encoding="utf-8")

    created = ensure_app_config(path)

    assert created is False
    # Absence is the single trigger, so an existing file is left byte-for-byte as it was.
    assert path.read_text(encoding="utf-8") == before
    assert load_app_config(path).quality == 55


def test_app_config_round_trips_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(font_family="Courier New, monospace")

    save_app_config(config, path)

    assert load_app_config(path).font_family == "Courier New, monospace"


def test_load_app_config_normalizes_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"font_family": "  Arial  "}), encoding="utf-8")

    assert load_app_config(path).font_family == "Arial"


def test_load_app_config_falls_back_on_unusable_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    path.write_text(json.dumps({"font_family": "   "}), encoding="utf-8")
    assert load_app_config(path).font_family == AppConfig().font_family

    path.write_text(json.dumps({"font_family": 42}), encoding="utf-8")
    assert load_app_config(path).font_family == AppConfig().font_family


def test_invalid_config_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_app_config(path)


def test_load_app_config_clamps_out_of_range_values(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"max_concurrent_jobs": 0, "quality": 250, "tile": -5}),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.max_concurrent_jobs == 1
    assert config.quality == 100
    assert config.tile == 0


def test_load_app_config_clamps_low_quality(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quality": -10}), encoding="utf-8")

    assert load_app_config(path).quality == 0


def test_load_app_config_coerces_numeric_strings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"max_concurrent_jobs": "4", "tile": "256"}),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.max_concurrent_jobs == 4
    assert config.tile == 256


def test_load_app_config_clamps_upper_bounds(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"max_concurrent_jobs": 99, "tile": 9999}),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.max_concurrent_jobs == 8
    assert config.tile == 4096


def test_load_app_config_coerces_unknown_device_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device": "gpu"}), encoding="utf-8")

    assert load_app_config(path).device == AppConfig().device


def test_load_app_config_lowercases_device(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device": "CPU"}), encoding="utf-8")

    assert load_app_config(path).device == "cpu"


def test_load_app_config_coerces_unknown_output_format_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_format": "gif"}), encoding="utf-8")

    assert load_app_config(path).output_format == AppConfig().output_format


def test_load_app_config_lowercases_output_format(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_format": "WEBP"}), encoding="utf-8")

    assert load_app_config(path).output_format == OutputFormat.WEBP


def test_load_app_config_falls_back_on_non_numeric_value(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quality": "high", "tile": None}), encoding="utf-8")

    config = load_app_config(path)

    assert config.quality == AppConfig().quality
    assert config.tile == AppConfig().tile
