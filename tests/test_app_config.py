import json
import threading
from dataclasses import fields, replace
from pathlib import Path

import pytest
from filelock import FileLock

from pixelup.app_config import (
    AppConfig,
    config_log_payload,
    load_app_config,
    load_app_config_result,
    save_app_config_merged,
)
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.jobs import JobSettings, job_settings_log_payload
from pixelup.paths import OutputFormat


def test_app_config_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        max_concurrent_jobs=3,
    )

    save_app_config_merged(config, AppConfig(), path)

    assert load_app_config(path) == config


def test_app_config_round_trips_the_parameters_panel(tmp_path: Path) -> None:
    # The panel is persisted user intent: what the user left in the Parameters group
    # must come back verbatim on the next launch. Every field is set away from its
    # built-in, so a field silently dropped by the serializer or the loader shows up
    # as a value that snapped back to its default.
    path = tmp_path / "config.json"
    parameters = JobSettings(
        scale=2,
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

    save_app_config_merged(config, AppConfig(), path)
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
    save_app_config_merged(AppConfig(parameters=JobSettings(tile=0)), AppConfig(), path)

    assert load_app_config(path).parameters.tile == 0


def test_fresh_config_carries_the_built_in_parameters(tmp_path: Path) -> None:
    # One source of the built-ins: a config with no parameters of its own is
    # JobSettings(), not a separately-written set of defaults.
    assert AppConfig().parameters == JobSettings()
    assert load_app_config(tmp_path / "missing.json").parameters == JobSettings()


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    assert load_app_config(path) == AppConfig()
    assert not path.exists()


def test_obsolete_auto_download_key_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"format_version": 1, "auto_download": True}), encoding="utf-8")

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.quarantined_to is None


def test_unchanged_defaults_write_no_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert save_app_config_merged(AppConfig(), AppConfig(), path) == AppConfig()
    assert not path.exists()


def test_one_changed_set_writes_only_its_key(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    candidate = AppConfig(parameters=JobSettings(quality=55))
    save_app_config_merged(candidate, AppConfig(), path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored) == {"format_version", "parameters"}
    assert stored["format_version"] == 1
    assert set(stored["parameters"]) == {item.name for item in fields(JobSettings)}
    assert load_app_config(path) == candidate


def test_one_stored_set_uses_built_ins_for_every_other_set(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"format_version": 1, "font_family": "Menlo"}', encoding="utf-8")
    assert load_app_config(path) == AppConfig(font_family="Menlo")


def test_next_edit_drops_unknown_keys_and_writes_the_format_version(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"format_version": 1, "version": 99, "future": true, "language": "ja"}', encoding="utf-8"
    )
    previous = load_app_config(path)
    save_app_config_merged(replace(previous, max_concurrent_jobs=3), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "format_version": 1, "language": "ja", "max_concurrent_jobs": 3,
    }


def test_saving_a_set_equal_to_its_built_in_deletes_only_that_set(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    previous = AppConfig(language="ja", parameters=JobSettings(quality=55))
    save_app_config_merged(previous, AppConfig(), path)
    reset = save_app_config_merged(replace(previous, parameters=JobSettings()), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"format_version": 1, "language": "ja"}
    assert reset == AppConfig(language="ja")


def test_the_last_set_back_at_its_built_in_leaves_an_empty_map(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    previous = AppConfig(parameters=JobSettings(quality=55))
    save_app_config_merged(previous, AppConfig(), path)
    save_app_config_merged(AppConfig(), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"format_version": 1}


def test_any_save_drops_a_stored_copy_equal_to_its_built_in(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"format_version": 1, "parameters": _parameter_map()}), encoding="utf-8"
    )
    previous = load_app_config(path)
    save_app_config_merged(replace(previous, language="ja"), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"format_version": 1, "language": "ja"}


def test_any_save_drops_a_set_that_failed_its_check(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"format_version": 1, "max_concurrent_jobs": 99, "language": "ja"}', encoding="utf-8"
    )
    previous = load_app_config(path)
    save_app_config_merged(replace(previous, font_family="Menlo"), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "format_version": 1, "font_family": "Menlo", "language": "ja",
    }


def test_app_config_round_trips_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(font_family="Courier New, monospace")

    save_app_config_merged(config, AppConfig(), path)

    assert load_app_config(path).font_family == "Courier New, monospace"


def test_load_app_config_cleans_font_family_as_a_single_line(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"format_version": 1, "font_family": "  Arial,\n Menlo  "}), encoding="utf-8"
    )

    assert load_app_config(path).font_family == "Arial, Menlo"


def test_whitespace_font_family_reads_as_the_built_in(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"format_version": 1, "font_family": "   "}), encoding="utf-8")

    assert load_app_config(path) == AppConfig()


def test_load_app_config_preserves_the_legacy_font_stack(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"format_version": 1, "font_family": "Helvetica Neue, Segoe UI, Roboto, Arial"}),
        encoding="utf-8",
    )

    assert load_app_config(path).font_family == "Helvetica Neue, Segoe UI, Roboto, Arial"


def test_load_app_config_ignores_unusable_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"format_version": 1, "font_family": 42}), encoding="utf-8")

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.quarantined_to is None
    assert path.exists()


def test_corrupt_config_is_quarantined_and_reads_as_built_ins(tmp_path: Path) -> None:
    # A present-but-corrupt config.json must never crash startup: the load path
    # quarantines the unreadable file aside (bytes preserved) and runs on the built-ins,
    # rather than raising or silently discarding the original (store-recovery-conventions).
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    result = load_app_config_result(path)

    # The built-ins were loaded, so the app can proceed.
    assert result.config == AppConfig()
    # The corrupt original was quarantined, not discarded: a <stem>-<ms-utc>.invalid
    # sibling now holds the exact original bytes, and config.json itself no longer does.
    assert result.quarantined_to is not None
    assert result.quarantined_to.parent == tmp_path
    assert result.quarantined_to.name.startswith("config-")
    assert result.quarantined_to.suffix == ".invalid"
    assert result.quarantined_to.read_text(encoding="utf-8") == "{ this is not valid json"
    # Recovery continues as a fresh install, without materializing defaults.
    assert not path.exists()
    assert load_app_config_result(path).quarantined_to is None
    assert load_app_config(path) == AppConfig()


def test_non_object_config_is_quarantined_and_reads_as_built_ins(tmp_path: Path) -> None:
    # A syntactically valid but wrong-shaped config (a JSON array, not an object) is
    # corrupt just the same: quarantined, never raised.
    path = tmp_path / "config.json"
    path.write_text("[]\n", encoding="utf-8")

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.quarantined_to is not None
    assert result.quarantined_to.suffix == ".invalid"
    assert result.quarantined_to.read_text(encoding="utf-8") == "[]\n"
    assert load_app_config(path) == AppConfig()


def test_a_failed_quarantine_propagates_and_leaves_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    def _fail(target: Path) -> Path:
        raise OSError("rename failed")

    monkeypatch.setattr("pixelup.app_config.quarantine_corrupt_file", _fail)

    with pytest.raises(OSError, match="rename failed"):
        load_app_config_result(path)
    assert path.read_text(encoding="utf-8") == "{ this is not valid json"


def test_a_config_without_a_format_version_is_quarantined(tmp_path: Path) -> None:
    # Nothing infers a version from the file's shape (store-recovery-conventions).
    path = tmp_path / "config.json"
    text = '{"language": "ja"}'
    path.write_text(text, encoding="utf-8")
    result = load_app_config_result(path)
    assert result.config == AppConfig()
    assert result.quarantined_to is not None
    assert result.quarantined_to.read_text(encoding="utf-8") == text
    assert not path.exists()


@pytest.mark.parametrize("marker", ['"1"', "0", "-1", "1.5", "true", "null"])
def test_a_format_version_that_is_not_a_version_is_quarantined(
    tmp_path: Path, marker: str
) -> None:
    path = tmp_path / "config.json"
    text = f'{{"format_version": {marker}, "language": "ja"}}'
    path.write_text(text, encoding="utf-8")
    result = load_app_config_result(path)
    assert result.config == AppConfig()
    assert result.quarantined_to is not None
    assert result.quarantined_to.read_text(encoding="utf-8") == text


def test_a_newer_config_is_left_in_place_and_reads_as_built_ins(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    text = '{"format_version": 2, "language": "ja"}'
    path.write_text(text, encoding="utf-8")

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.newer_format == 2
    assert result.quarantined_to is None
    assert path.read_text(encoding="utf-8") == text
    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


def test_a_save_over_a_newer_config_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    text = '{"format_version": 2, "future": {"anything": true}}'
    path.write_text(text, encoding="utf-8")
    previous = load_app_config(path)

    with pytest.raises(PixelupError) as raised:
        save_app_config_merged(replace(previous, language="ja"), previous, path)

    assert raised.value.code == ErrorCode.STORE_NEWER_FORMAT
    assert raised.value.message.key == "error.configNewer"
    assert path.read_text(encoding="utf-8") == text


def test_missing_config_is_not_treated_as_corrupt(tmp_path: Path) -> None:
    # The normal first run: no file, defaults, and nothing quarantined.
    result = load_app_config_result(tmp_path / "missing.json")

    assert result.config == AppConfig()
    assert result.quarantined_to is None


def test_corrupt_config_lets_window_open(
    qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end pin: a corrupt config.json under a redirected PIXELUP_DATA_DIR must let
    # MainWindow construct (the frozen-app failure the finding is about was the window
    # never opening), running on the built-ins and telling the user rather than
    # crashing.
    from PySide6.QtWidgets import QApplication

    from pixelup.gui import MainWindow
    from pixelup.runner import JobRunner
    from pixelup.session_log import configure_session_logging

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(home))
    (home / "config.json").write_text("{ broken", encoding="utf-8")

    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    notices: list[str] = []
    monkeypatch.setattr(
        "pixelup.gui.warn_config_reset",
        lambda _parent: notices.append("shown"),
    )
    log_file = configure_session_logging()

    window = MainWindow(log_file=log_file)
    try:
        # The window opened on defaults instead of crashing on the corrupt file.
        assert window.config == AppConfig()
        assert window._config_quarantined_to is not None
        # The corrupt original was quarantined next to where config.json was.
        assert window._config_quarantined_to.suffix == ".invalid"
        assert window._config_quarantined_to.read_text(encoding="utf-8") == "{ broken"
        assert not (home / "config.json").exists()
        # The deferred non-fatal notice fires once the event loop turns.
        QApplication.processEvents()
        assert notices == ["shown"]
    finally:
        window._session_shutdown = True
        window.close()
        window.deleteLater()
        QApplication.processEvents()


def _parameter_map(**updates: object) -> dict[str, object]:
    defaults = JobSettings()
    return {
        "scale": defaults.scale,
        "denoise_strength": defaults.denoise_strength,
        "alpha_mode": defaults.alpha_mode,
        "device": defaults.device,
        "output_format": defaults.output_format.value,
        "quality": defaults.quality,
        "tile": defaults.tile,
        "strip_metadata": defaults.strip_metadata,
        "target_profile": defaults.target_profile,
        **updates,
    }


def _write(path: Path, **parameters: object) -> None:
    path.write_text(
        json.dumps({"format_version": 1, "parameters": _parameter_map(**parameters)}),
        encoding="utf-8",
    )


def test_a_newer_config_lets_the_window_open_and_is_never_written(
    qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QApplication

    from pixelup.gui import MainWindow
    from pixelup.runner import JobRunner
    from pixelup.session_log import configure_session_logging

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(home))
    text = '{"format_version": 2, "language": "ja"}'
    (home / "config.json").write_text(text, encoding="utf-8")

    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    notices: list[str] = []
    monkeypatch.setattr("pixelup.gui.warn_config_newer", lambda _parent: notices.append("shown"))
    monkeypatch.setattr("pixelup.gui.warn_config_reset", lambda _parent: notices.append("reset"))
    log_file = configure_session_logging()

    window = MainWindow(log_file=log_file)
    try:
        assert window.config == AppConfig()
        assert window._config_quarantined_to is None
        QApplication.processEvents()
        assert notices == ["shown"]

        window.quality.setValue(window.quality.value() - 1)
        assert window._flush_parameters_save() is False
        assert window.parameters_result.message == Message("error.configNewer")
        assert (home / "config.json").read_text(encoding="utf-8") == text
    finally:
        window._session_shutdown = True
        window.close()
        window.deleteLater()


def test_invalid_utf8_config_is_quarantined_and_reads_as_built_ins(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = b'{"font_family": "\xff"}'
    path.write_bytes(original)

    result = load_app_config_result(path)

    assert result.config == AppConfig()
    assert result.quarantined_to is not None
    assert result.quarantined_to.read_bytes() == original
    assert load_app_config(path) == AppConfig()


def test_config_read_oserror_propagates_without_touching_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    original = b'{"auto_download": false}'
    path.write_bytes(original)

    def fail_read(_path: Path, *args: object, **kwargs: object) -> str:
        raise OSError("read failed")

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(OSError, match="read failed"):
        load_app_config_result(path)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.invalid"))


@pytest.mark.parametrize(
    "data",
    [
        {"max_concurrent_jobs": 0},
        {"max_concurrent_jobs": 99},
        {"max_concurrent_jobs": "4"},
        {"max_concurrent_jobs": True},
        {"language": "JA"},
        {"language": 7},
        {"parameters": "nope"},
        {"parameters": _parameter_map(strip_metadata="false")},
        {"parameters": _parameter_map(quality=250)},
        {"parameters": _parameter_map(quality="80")},
        {"parameters": _parameter_map(denoise_strength=9.5)},
        {"parameters": _parameter_map(denoise_strength="0.25")},
        {"parameters": _parameter_map(denoise_strength=10**1000)},
        {"parameters": _parameter_map(scale=2.0)},
        {"parameters": _parameter_map(scale="2")},
        {"parameters": _parameter_map(scale=3)},
        {"parameters": _parameter_map(tile=9999)},
        {"parameters": _parameter_map(tile="512")},
        {"parameters": _parameter_map(device="CPU")},
        {"parameters": _parameter_map(device="gpu")},
        {"parameters": _parameter_map(output_format="WEBP")},
        {"parameters": _parameter_map(output_format="gif")},
        {"parameters": _parameter_map(alpha_mode="nearest")},
        {"parameters": _parameter_map(target_profile="cmyk")},
    ],
)
def test_present_malformed_set_falls_back_without_costing_other_sets(
    tmp_path: Path, data: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"format_version": 1, "font_family": "Menlo", **data}), encoding="utf-8"
    )
    original = path.read_bytes()
    result = load_app_config_result(path)
    assert result.config == AppConfig(font_family="Menlo")
    assert result.quarantined_to is None
    assert path.read_bytes() == original
    warnings = [record for record in caplog.records if record.message == "config.invalid_set"]
    assert len(warnings) == 1
    assert warnings[0].fields["key"] == next(iter(data))


def test_absent_and_unknown_fields_do_not_make_the_config_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "max_concurrent_jobs": 4,
                "future_setting": {"future": True},
                "parameters": _parameter_map(scale=2, future_parameter="ignored"),
            }
        ),
        encoding="utf-8",
    )

    result = load_app_config_result(path)

    assert result.quarantined_to is None
    assert result.config.max_concurrent_jobs == 4
    assert result.config.parameters.scale == 2
    assert result.config.parameters.quality == JobSettings().quality


def test_obsolete_face_enhance_field_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "parameters": _parameter_map(face_enhance=True, strip_metadata=False),
            }
        ),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.parameters.strip_metadata is False
    assert not hasattr(config.parameters, "face_enhance")


def test_valid_parameter_boundaries_and_choices_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(
        path,
        quality=0,
        denoise_strength=1.0,
        scale=2,
        tile=0,
        device="cpu",
        output_format="webp",
        alpha_mode="bicubic",
        target_profile="adobergb",
    )

    parameters = load_app_config(path).parameters

    assert parameters.quality == 0
    assert parameters.denoise_strength == 1.0
    assert parameters.scale == 2
    assert parameters.tile == 0
    assert parameters.device == "cpu"
    assert parameters.output_format == OutputFormat.WEBP
    assert parameters.alpha_mode == "bicubic"
    assert parameters.target_profile == "adobergb"


def test_null_target_profile_is_a_valid_choice(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, target_profile=None)

    assert load_app_config(path).parameters.target_profile is None


def test_old_flat_keys_are_inert(tmp_path: Path) -> None:
    # Pre-release, deliberately un-migrated: the four keys AppConfig used to hold flat
    # are simply not read any more, and must not leak back in as parameters.
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "output_format": "webp",
                "quality": 10,
                "tile": 1024,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.parameters == JobSettings()


def test_save_app_config_merged_keeps_a_sibling_windows_untouched_field(
    tmp_path: Path,
) -> None:
    # Two windows open on the same config.json (PU-3). Window A saves a settings
    # change; window B, still holding its older snapshot, then saves an unrelated
    # Parameters-panel edit. B's save must not carry A's field back to its stale value.
    path = tmp_path / "config.json"
    opened = AppConfig()
    save_app_config_merged(opened, AppConfig(), path)

    save_app_config_merged(replace(opened, max_concurrent_jobs=4), AppConfig(), path)  # window A

    b_candidate = replace(opened, parameters=JobSettings(quality=42))  # window B
    merged = save_app_config_merged(b_candidate, opened, path)

    assert merged.max_concurrent_jobs == 4
    assert merged.parameters.quality == 42
    assert load_app_config(path) == merged


def test_save_app_config_merged_is_a_no_op_when_candidate_matches_previous(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    opened = AppConfig(max_concurrent_jobs=3)
    save_app_config_merged(opened, AppConfig(), path)
    written_at = path.stat().st_mtime_ns

    merged = save_app_config_merged(opened, opened, path)

    assert merged == opened
    assert path.stat().st_mtime_ns == written_at


def test_save_app_config_merged_times_out_behind_another_holder(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_app_config_merged(AppConfig(), AppConfig(), path)
    lock = FileLock(str(path.with_name(f"{path.name}.lock")))
    lock.acquire()
    try:
        with pytest.raises(PixelupError) as excinfo:
            save_app_config_merged(
                AppConfig(max_concurrent_jobs=2),
                AppConfig(),
                path,
            )
        assert excinfo.value.code == "internal_error"
    finally:
        lock.release()


def test_save_app_config_merged_serializes_two_concurrent_savers(tmp_path: Path) -> None:
    # Not a race in the fix itself: both callers' edits land, applied one after the
    # other rather than one silently overwriting the other's.
    path = tmp_path / "config.json"
    opened = AppConfig()
    save_app_config_merged(opened, AppConfig(), path)
    ready = threading.Barrier(2)
    results: list[AppConfig] = []
    errors: list[Exception] = []

    def save(candidate: AppConfig) -> None:
        try:
            ready.wait(timeout=2)
            results.append(save_app_config_merged(candidate, opened, path))
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` for the assertion.
            errors.append(exc)

    thread_a = threading.Thread(
        target=save, args=(replace(opened, max_concurrent_jobs=5),)
    )
    thread_b = threading.Thread(
        target=save, args=(replace(opened, parameters=JobSettings(quality=17)),)
    )
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)

    assert not errors
    final = load_app_config(path)
    assert final.max_concurrent_jobs == 5
    assert final.parameters.quality == 17


def test_config_log_payload_shape() -> None:
    config = AppConfig(
        max_concurrent_jobs=2,
        parameters=JobSettings(quality=70, tile=128, device="cpu"),
    )

    assert config_log_payload(config) == {
        "max_concurrent_jobs": 2,
        "font_family": AppConfig().font_family,
        "language": "system",
        "parameters": job_settings_log_payload(config.parameters),
    }


def test_partial_parameters_read_as_the_built_in_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"format_version": 1, "language": "ja", "parameters": {"scale": 2}}', encoding="utf-8"
    )
    result = load_app_config_result(path)
    assert result.config == AppConfig(language="ja")
    assert result.quarantined_to is None
    warnings = [record for record in caplog.records if record.message == "config.invalid_set"]
    assert len(warnings) == 1
    assert warnings[0].fields["key"] == "parameters"


def test_edit_writes_the_untouched_user_set_whole_from_memory(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    stored_parameters = _parameter_map(quality=42, future_parameter="dropped")
    path.write_text(
        json.dumps({"format_version": 1, "parameters": stored_parameters}), encoding="utf-8"
    )
    previous = load_app_config(path)
    save_app_config_merged(replace(previous, language="ja"), previous, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "format_version": 1, "parameters": _parameter_map(quality=42), "language": "ja",
    }
