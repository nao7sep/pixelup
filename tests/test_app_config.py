import json
from pathlib import Path

import pytest

from pixelup.app_config import (
    AppConfig,
    config_log_payload,
    ensure_app_config,
    load_app_config,
    load_app_config_result,
    save_app_config,
)
from pixelup.jobs import JobSettings, job_settings_log_payload
from pixelup.parameters import DEFAULT_SCALE
from pixelup.paths import OutputFormat


def test_app_config_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        max_concurrent_jobs=3,
        auto_download=False,
    )

    save_app_config(config, path)

    assert load_app_config(path) == config


def test_app_config_round_trips_the_parameters_panel(tmp_path: Path) -> None:
    # The panel is persisted user intent: what the user left in the Parameters group
    # must come back verbatim on the next launch. Every field is set away from its
    # built-in, so a field silently dropped by the serializer or the loader shows up
    # as a value that snapped back to its default.
    path = tmp_path / "config.json"
    parameters = JobSettings(
        scale=2,
        face_enhance=True,
        denoise_strength=0.25,
        alpha_mode="bicubic",
        device="cpu",
        output_format=OutputFormat.WEBP,
        quality=82,
        tile=512,
        strip_metadata=True,
        target_profile="p3",
    )
    config = AppConfig(parameters=parameters)

    save_app_config(config, path)
    loaded = load_app_config(path)

    assert loaded.parameters == parameters
    assert loaded.parameters.scale == 2
    assert loaded == config
    # Nothing survived by accident: the round-tripped panel differs from the built-ins
    # in every field, so this could not pass on a loader that just returns defaults.
    assert loaded.parameters != JobSettings()


def test_app_config_round_trips_a_deliberate_zero_tile(tmp_path: Path) -> None:
    # 0 is no longer the default but is still a real choice, and it is also falsy —
    # so it is exactly the value a truthiness bug in the loader would quietly replace
    # with 256. It must survive the round trip.
    path = tmp_path / "config.json"
    save_app_config(AppConfig(parameters=JobSettings(tile=0)), path)

    assert load_app_config(path).parameters.tile == 0


def test_fresh_config_carries_the_built_in_parameters(tmp_path: Path) -> None:
    # One source of the built-ins: a config with no parameters of its own is
    # JobSettings(), not a separately-written set of defaults.
    assert AppConfig().parameters == JobSettings()
    assert load_app_config(tmp_path / "missing.json").parameters == JobSettings()


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    assert load_app_config(tmp_path / "missing.json") == AppConfig()


def test_model_downloads_are_off_until_the_user_turns_them_on() -> None:
    # The managed-runtime-dependencies conventions: nothing downloads on its own.
    # A fresh install downloads only after the user flips the Settings toggle.
    assert AppConfig().auto_download is False


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
    save_app_config(AppConfig(parameters=JobSettings(quality=55)), path)
    before = path.read_text(encoding="utf-8")

    created = ensure_app_config(path)

    assert created is False
    # Absence is the single trigger, so an existing file is left byte-for-byte as it was.
    assert path.read_text(encoding="utf-8") == before
    assert load_app_config(path).parameters.quality == 55


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


def _write(path: Path, **parameters: object) -> None:
    path.write_text(json.dumps({"parameters": parameters}), encoding="utf-8")


def test_load_app_config_clamps_concurrent_jobs(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    path.write_text(json.dumps({"max_concurrent_jobs": 0}), encoding="utf-8")
    assert load_app_config(path).max_concurrent_jobs == 1

    path.write_text(json.dumps({"max_concurrent_jobs": 99}), encoding="utf-8")
    assert load_app_config(path).max_concurrent_jobs == 8

    path.write_text(json.dumps({"max_concurrent_jobs": "4"}), encoding="utf-8")
    assert load_app_config(path).max_concurrent_jobs == 4


def test_load_parameters_clamps_out_of_range_ranges(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, quality=250, denoise_strength=9.5)

    parameters = load_app_config(path).parameters

    assert parameters.quality == 100
    assert parameters.denoise_strength == 1.0


def test_load_parameters_falls_back_rather_than_clamping_a_stray_tile(tmp_path: Path) -> None:
    # Tile is enumerated now, not a range, and the difference is not academic: when it
    # was clamped, a hand-edited -5 snapped to the NEAREST END — 0 — which is the
    # whole-image pass, the one value documented as able to hard-crash the machine. A
    # stray value is not "near" a choice, so it falls back to the built-in like scale.
    path = tmp_path / "config.json"
    _write(path, tile=-5)

    assert load_app_config(path).parameters.tile == JobSettings().tile


def test_load_parameters_clamps_lower_bounds(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, quality=-10, denoise_strength=-1.0)

    parameters = load_app_config(path).parameters

    assert parameters.quality == 0
    assert parameters.denoise_strength == 0.0


def test_load_parameters_falls_back_on_a_tile_that_is_not_a_choice(tmp_path: Path) -> None:
    # 9999 used to clamp to 4096 — a size the panel can no longer show, and one that
    # for a typical photo is a single tile, i.e. the whole-image pass under another
    # name. It falls back to the built-in instead.
    path = tmp_path / "config.json"
    _write(path, tile=9999)

    assert load_app_config(path).parameters.tile == JobSettings().tile


def test_load_parameters_keeps_a_tile_that_is_a_real_choice(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, tile=0)

    # "Whole image" is a genuine choice, not a stray value to fall back from.
    assert load_app_config(path).parameters.tile == 0


def test_load_parameters_coerces_numeric_strings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, tile="512", quality="80", denoise_strength="0.25")

    parameters = load_app_config(path).parameters

    assert parameters.tile == 512
    assert parameters.quality == 80
    assert parameters.denoise_strength == 0.25


def test_load_parameters_reads_a_persisted_scale(tmp_path: Path) -> None:
    # The other selectable scale survives a load on its own, straight from the JSON the
    # user's file actually holds.
    path = tmp_path / "config.json"
    _write(path, scale=2)

    assert load_app_config(path).parameters.scale == 2


def test_load_parameters_coerces_numeric_string_scale(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, scale="2")

    assert load_app_config(path).parameters.scale == 2


@pytest.mark.parametrize("bad", [3, 0, -4, 8, "4x", None, "", [], {}, 2.5, "2.5", True])
def test_load_parameters_falls_back_to_the_built_in_scale(tmp_path: Path, bad: object) -> None:
    # Scale is enumerated, not a range: an unselectable value is not clamped toward a
    # neighbour the user never chose, it falls back to the built-in. 2.5 and "2.5" are
    # the pins that int() truncation cannot quietly manufacture a selectable 2.
    path = tmp_path / "config.json"
    _write(path, scale=bad)

    assert load_app_config(path).parameters.scale == DEFAULT_SCALE


def test_load_parameters_reads_an_integral_float_scale(tmp_path: Path) -> None:
    # 2.0 is unambiguously the 2x choice — JSON has one number type, so a hand-edited
    # or re-serialized file can carry it — and it comes back as the canonical int.
    path = tmp_path / "config.json"
    _write(path, scale=2.0)

    scale = load_app_config(path).parameters.scale
    assert scale == 2
    assert isinstance(scale, int)


def test_load_parameters_keeps_the_panel_when_only_scale_is_bad(tmp_path: Path) -> None:
    # The per-field lenient-load contract, checked on the new field: one unreadable
    # scale must cost the user the scale, not the other nine parameters.
    path = tmp_path / "config.json"
    _write(path, scale=99, quality=42, tile=512, device="cpu")

    parameters = load_app_config(path).parameters

    assert parameters.scale == DEFAULT_SCALE
    assert parameters.quality == 42
    assert parameters.tile == 512
    assert parameters.device == "cpu"


def test_load_parameters_coerces_unknown_device_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, device="gpu")

    assert load_app_config(path).parameters.device == JobSettings().device


def test_load_parameters_lowercases_device(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, device="CPU")

    assert load_app_config(path).parameters.device == "cpu"


def test_load_parameters_coerces_unknown_output_format_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, output_format="gif")

    assert load_app_config(path).parameters.output_format == JobSettings().output_format


def test_load_parameters_lowercases_output_format(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, output_format="WEBP")

    assert load_app_config(path).parameters.output_format == OutputFormat.WEBP


def test_load_parameters_coerces_unknown_alpha_mode_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, alpha_mode="nearest")

    assert load_app_config(path).parameters.alpha_mode == JobSettings().alpha_mode


def test_load_parameters_accepts_known_alpha_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, alpha_mode="bicubic")

    assert load_app_config(path).parameters.alpha_mode == "bicubic"


def test_load_parameters_coerces_unknown_target_profile_to_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, target_profile="cmyk")

    assert load_app_config(path).parameters.target_profile is None


def test_load_parameters_accepts_null_and_known_target_profile(tmp_path: Path) -> None:
    path = tmp_path / "config.json"

    # null is itself a valid profile ("Default"), not a bad value to fall back from.
    _write(path, target_profile=None)
    assert load_app_config(path).parameters.target_profile is None

    _write(path, target_profile="adobergb")
    assert load_app_config(path).parameters.target_profile == "adobergb"


def test_load_parameters_falls_back_on_non_numeric_value(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, quality="high", tile=None, denoise_strength="loud")

    parameters = load_app_config(path).parameters

    assert parameters.quality == JobSettings().quality
    assert parameters.tile == JobSettings().tile
    assert parameters.denoise_strength == JobSettings().denoise_strength


def test_load_parameters_keeps_good_fields_beside_a_bad_one(tmp_path: Path) -> None:
    # Per-field fallback, not all-or-nothing: one unreadable field must not cost the
    # user the rest of the panel.
    path = tmp_path / "config.json"
    _write(path, tile="nonsense", quality=60, device="cpu")

    parameters = load_app_config(path).parameters

    assert parameters.tile == JobSettings().tile
    assert parameters.quality == 60
    assert parameters.device == "cpu"


def test_load_parameters_falls_back_when_not_an_object(tmp_path: Path) -> None:
    # A parameters key of the wrong shape is a field that cannot be read, not
    # whole-file corruption: the panel falls back, the load does not quarantine.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"parameters": "nope", "max_concurrent_jobs": 3}), encoding="utf-8")

    result = load_app_config_result(path)

    assert result.quarantined_to is None
    assert result.config.parameters == JobSettings()
    assert result.config.max_concurrent_jobs == 3


def test_old_flat_keys_are_inert(tmp_path: Path) -> None:
    # Pre-release, deliberately un-migrated: the four keys AppConfig used to hold flat
    # are simply not read any more, and must not leak back in as parameters.
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"output_format": "webp", "quality": 10, "tile": 1024, "device": "cpu"}),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.parameters == JobSettings()


def test_config_log_payload_shape() -> None:
    config = AppConfig(
        max_concurrent_jobs=2,
        auto_download=False,
        parameters=JobSettings(quality=70, tile=128, device="cpu"),
    )

    assert config_log_payload(config) == {
        "max_concurrent_jobs": 2,
        "auto_download": False,
        "font_family": AppConfig().font_family,
        "parameters": job_settings_log_payload(config.parameters),
    }
