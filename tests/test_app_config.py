import json
from pathlib import Path

import pytest

from pixelup.app_config import (
    AppConfig,
    ensure_app_config,
    load_app_config,
    load_app_config_result,
    save_app_config,
)
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


def test_corrupt_config_is_quarantined_then_reset(tmp_path: Path) -> None:
    # A present-but-corrupt config.json must never crash startup: the load path
    # quarantines the unreadable file aside (bytes preserved) and resets to defaults,
    # rather than raising or silently discarding the original (storage-path conventions).
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    result = load_app_config_result(path)

    # Defaults were loaded, so the app can proceed on a fresh, valid config.
    assert result.config == AppConfig()
    # The corrupt original was quarantined, not discarded: a <stem>-<ms-utc>.invalid
    # sibling now holds the exact original bytes, and config.json itself no longer does.
    assert result.quarantined_to is not None
    assert result.quarantined_to.parent == tmp_path
    assert result.quarantined_to.name.startswith("config-")
    assert result.quarantined_to.suffix == ".invalid"
    assert result.quarantined_to.read_text(encoding="utf-8") == "{ this is not valid json"
    # config.json was reset on disk through the normal save path, so the next load is clean.
    assert path.exists()
    assert load_app_config_result(path).quarantined_to is None
    assert load_app_config(path) == AppConfig()


def test_non_object_config_is_quarantined_then_reset(tmp_path: Path) -> None:
    # A syntactically valid but wrong-shaped config (a JSON array, not an object) is
    # corrupt just the same: quarantined-then-reset, never raised.
    path = tmp_path / "config.json"
    path.write_text("[]\n", encoding="utf-8")

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.quarantined_to is not None
    assert result.quarantined_to.suffix == ".invalid"
    assert result.quarantined_to.read_text(encoding="utf-8") == "[]\n"
    assert load_app_config(path) == AppConfig()


def test_missing_config_is_not_treated_as_corrupt(tmp_path: Path) -> None:
    # The normal first run: no file, defaults, and nothing quarantined.
    result = load_app_config_result(tmp_path / "missing.json")

    assert result.config == AppConfig()
    assert result.quarantined_to is None


def test_corrupt_config_lets_window_open(
    qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end pin: a corrupt config.json under a redirected PIXELUP_HOME must let
    # MainWindow construct (the frozen-app failure the finding is about was the window
    # never opening), running on freshly reset defaults and surfacing the reset to the
    # user rather than crashing.
    from PySide6.QtWidgets import QApplication

    from pixelup.gui import MainWindow
    from pixelup.runner import JobRunner
    from pixelup.session_log import configure_session_logging

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PIXELUP_HOME", str(home))
    (home / "config.json").write_text("{ broken", encoding="utf-8")

    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    notices: list[str] = []
    monkeypatch.setattr(
        "pixelup.gui.warn_config_reset",
        lambda parent, name: notices.append(name),
    )
    log_file = home / "logs" / "session.log"
    configure_session_logging(log_file)

    window = MainWindow(log_file=log_file)
    try:
        # The window opened on defaults instead of crashing on the corrupt file.
        assert window.config == AppConfig()
        assert window._config_quarantined_to is not None
        # The corrupt original was quarantined next to the (now reset) config.json.
        assert window._config_quarantined_to.suffix == ".invalid"
        assert window._config_quarantined_to.read_text(encoding="utf-8") == "{ broken"
        assert (home / "config.json").exists()
        # The deferred non-fatal notice fires once the event loop turns.
        QApplication.processEvents()
        assert notices == [window._config_quarantined_to.name]
    finally:
        window._session_shutdown = True
        window.close()
        window.deleteLater()
        QApplication.processEvents()


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
