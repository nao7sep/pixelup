import json
import os
import stat
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
from config_support import header as _header
from config_support import load_app_config
from config_support import stored_content as _content

from pixelup import __version__
from pixelup.app_config import (
    AppConfig,
    ConfigSaveResult,
    config_log_payload,
    load_app_config_result,
    save_app_config,
)
from pixelup.formats import RefusedFile, RefusedFilesError
from pixelup.jobs import JobSettings, job_settings_log_payload
from pixelup.paths import OutputFormat


def _save(candidate: AppConfig, previous: AppConfig, path: Path) -> ConfigSaveResult:
    """One save as a launch makes it: from the file state read at load."""
    return save_app_config(candidate, previous, load_app_config_result(path.parent).files)


def test_app_config_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(font_family="Menlo")

    _save(config, AppConfig(), path)

    assert load_app_config(path.parent) == config


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

    _save(config, AppConfig(), path)
    loaded = load_app_config(path.parent)

    assert loaded.parameters == parameters
    assert loaded.parameters.scale == 2
    assert loaded == config
    # Nothing survived by accident: the round-tripped panel differs from the built-ins
    # in every field, so this could not pass on a loader that just returns defaults.
    assert loaded.parameters != JobSettings()


def test_app_config_round_trips_a_non_default_tile(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _save(AppConfig(parameters=JobSettings(tile=2048)), AppConfig(), path)

    assert load_app_config(path.parent).parameters.tile == 2048


def test_fresh_config_carries_the_built_in_parameters(tmp_path: Path) -> None:
    # One source of the built-ins: a config with no parameters of its own is
    # JobSettings(), not a separately-written set of defaults.
    assert AppConfig().parameters == JobSettings()
    assert load_app_config(tmp_path).parameters == JobSettings()


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    assert load_app_config(path.parent) == AppConfig()
    assert not path.exists()


def test_unchanged_defaults_write_no_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert _save(AppConfig(), AppConfig(), path).config == AppConfig()
    assert not path.exists()


def test_one_changed_set_writes_only_its_key(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    candidate = AppConfig(parameters=JobSettings(quality=55))
    _save(candidate, AppConfig(), path)
    stored = _content(path)
    assert set(stored) == {"parameters"}
    assert set(stored["parameters"]) == {item.name for item in fields(JobSettings)}
    assert not (tmp_path / "config.json").exists()
    assert load_app_config(path.parent) == candidate


def test_preferences_and_parameters_are_stored_in_their_own_files(tmp_path: Path) -> None:
    candidate = AppConfig(language="ja", parameters=JobSettings(quality=55))
    _save(candidate, AppConfig(), tmp_path / "config.json")
    assert _content(tmp_path / "config.json") == {
        "language": "ja",
    }
    assert _content(tmp_path / "parameters.json") == {
        "parameters": _parameter_map(quality=55),
    }
    assert load_app_config(tmp_path) == candidate


def test_each_save_writes_only_the_file_whose_sets_changed(tmp_path: Path) -> None:
    # A Settings save leaves parameters.json alone and a Parameters save or reset
    # leaves config.json alone, so each file's history follows its own editor.
    opened = AppConfig(language="ja", parameters=JobSettings(quality=55))
    first = _save(opened, AppConfig(), tmp_path / "config.json")
    old = 1_000_000_000
    for name in ("config.json", "parameters.json"):
        os.utime(tmp_path / name, (old, old))

    reset = replace(opened, parameters=JobSettings())
    second = save_app_config(reset, opened, first.files)
    assert (tmp_path / "config.json").stat().st_mtime == old
    assert _content(tmp_path / "parameters.json") == {}

    os.utime(tmp_path / "parameters.json", (old, old))
    save_app_config(replace(reset, font_family="Menlo"), reset, second.files)
    assert (tmp_path / "parameters.json").stat().st_mtime == old


def _refused(root: Path) -> tuple[RefusedFile, ...]:
    """Load ``root`` expecting a refusal, and check no settings file was changed."""
    before = {path.name: path.read_bytes() for path in root.glob("*.json")}
    with pytest.raises(RefusedFilesError) as raised:
        load_app_config_result(root)
    assert {path.name: path.read_bytes() for path in root.glob("*.json")} == before
    return raised.value.files


@pytest.mark.parametrize("name", ["config.json", "parameters.json"])
def test_a_newer_settings_file_stops_the_launch_untouched(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text(
        json.dumps({**_header(path.stem, 2), "future": {"anything": True}}), encoding="utf-8"
    )
    assert _refused(tmp_path) == (RefusedFile(path, newer=True),)


def test_every_refused_settings_file_is_named(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{ broken", encoding="utf-8")
    (tmp_path / "parameters.json").write_text(
        json.dumps(_header("parameters", 2)), encoding="utf-8"
    )
    assert _refused(tmp_path) == (
        RefusedFile(tmp_path / "config.json", newer=False),
        RefusedFile(tmp_path / "parameters.json", newer=True),
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS replacement modes")
def test_a_changed_save_keeps_the_file_mode_with_a_fresh_modified_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "parameters.json"
    _save(AppConfig(parameters=JobSettings(quality=55)), AppConfig(), path)
    os.chmod(path, 0o640)
    old = 1_000_000_000
    os.utime(path, (old, old))

    previous = load_app_config(path.parent)
    _save(replace(previous, parameters=JobSettings(quality=60)), previous, path)

    assert load_app_config(path.parent).parameters.quality == 60
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    assert os.stat(path).st_mtime > old
    assert list(tmp_path.glob("*.tmp")) == []


def test_one_stored_set_uses_built_ins_for_every_other_set(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_header("config"), "font_family": "Menlo"}), encoding="utf-8")
    assert load_app_config(path.parent) == AppConfig(font_family="Menlo")


def test_next_edit_keeps_unknown_keys_and_writes_the_header(tmp_path: Path) -> None:
    # A key this build does not know (an older one, or one a newer PixelUp added) is
    # not disposable: an unrelated save writes it back unchanged (developer decision).
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({**_header("config"), "version": 99, "future": {"a": [1, 2]}, "language": "ja"}),
        encoding="utf-8",
    )
    previous = load_app_config(path.parent)
    _save(replace(previous, font_family="Menlo"), previous, path)
    assert _content(path) == {
        "version": 99,
        "future": {"a": [1, 2]},
        "language": "ja",
        "font_family": "Menlo",
    }


def test_saving_a_set_equal_to_its_built_in_deletes_only_that_set(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    previous = AppConfig(language="ja", font_family="Menlo")
    _save(previous, AppConfig(), path)
    reset = _save(replace(previous, font_family=AppConfig().font_family), previous, path).config
    assert _content(path) == {"language": "ja"}
    assert reset == AppConfig(language="ja")


def test_the_last_set_back_at_its_built_in_leaves_an_empty_map(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    previous = AppConfig(parameters=JobSettings(quality=55))
    _save(previous, AppConfig(), path)
    _save(AppConfig(), previous, path)
    assert _content(path) == {}


def test_any_save_drops_a_stored_copy_equal_to_its_built_in(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    path.write_text(
        json.dumps({**_header("parameters"), "parameters": _parameter_map()}), encoding="utf-8"
    )
    previous = load_app_config(path.parent)
    _save(replace(previous, parameters=JobSettings(quality=60)), previous, path)
    _save(replace(previous, parameters=JobSettings()), previous, path)
    assert _content(path) == {}


def test_an_unrelated_save_keeps_a_set_that_failed_its_check(tmp_path: Path) -> None:
    # The rejected set runs on its built-in, but its stored value is the user's and
    # stays in the file, unchanged, until the user saves that same set (developer
    # decision).
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({**_header("config"), "language": "JA", "future": "kept"}),
        encoding="utf-8",
    )
    loaded = load_app_config_result(path.parent)
    assert loaded.rejected == ("language",)
    assert loaded.config.language == "system"

    first = save_app_config(
        replace(loaded.config, display_zone="Asia/Tokyo"), loaded.config, loaded.files
    )
    second = save_app_config(
        replace(first.config, font_family="Menlo"), first.config, first.files
    )

    assert second.config.language == "system"
    assert _content(path) == {
        "language": "JA",
        "future": "kept",
        "display_zone": "Asia/Tokyo",
        "font_family": "Menlo",
    }


def test_saving_a_set_that_failed_its_check_replaces_it(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    path.write_text(
        json.dumps({**_header("parameters"), "parameters": _parameter_map(quality=250)}),
        encoding="utf-8",
    )
    loaded = load_app_config_result(path.parent)
    edited = replace(loaded.config, parameters=JobSettings(quality=60))

    result = save_app_config(edited, loaded.config, loaded.files)

    assert _content(path) == {
        "parameters": _parameter_map(quality=60),
    }
    assert result.files.parameters.unapplied == frozenset()
    # The next unrelated save no longer has anything to carry for that set.
    later = save_app_config(replace(edited, language="ja"), edited, result.files)
    assert _content(path)["parameters"] == _parameter_map(
        quality=60
    )
    assert later.files.parameters.unapplied == frozenset()


def test_a_set_that_failed_its_check_is_kept_byte_for_byte_across_a_save(tmp_path: Path) -> None:
    # Written back exactly as stored: float spelling, key order inside the set and
    # unknown members all survive, so nothing about the user's copy is reinterpreted.
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {**_header("config"), "display_zone": {"offset": 9.0, "z": None, "zone": "Asia/Tokyo"}}
        ),
        encoding="utf-8",
    )
    loaded = load_app_config_result(path.parent)

    save_app_config(replace(loaded.config, language="ja"), loaded.config, loaded.files)

    stored = _content(path)
    assert json.dumps(stored["display_zone"]) == '{"offset": 9.0, "z": null, "zone": "Asia/Tokyo"}'


def test_a_save_replaces_hand_edits_made_after_load(tmp_path: Path) -> None:
    # PixelUp owns config.json while it runs (developer decision): its save writes
    # from the state read at launch, so an edit made to the file meanwhile, even one
    # that broke it, is replaced by the user's current settings.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_header("config"), "language": "ja"}), encoding="utf-8")
    loaded = load_app_config_result(path.parent)
    path.write_text("{ broken by hand", encoding="utf-8")

    save_app_config(replace(loaded.config, font_family="Menlo"), loaded.config, loaded.files)

    assert _content(path) == {
        "language": "ja",
        "font_family": "Menlo",
    }
    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


def test_app_config_round_trips_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(font_family="Courier New, monospace")

    _save(config, AppConfig(), path)

    assert load_app_config(path.parent).font_family == "Courier New, monospace"


def test_load_app_config_cleans_font_family_as_a_single_line(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({**_header("config"), "font_family": "  Arial,\n Menlo  "}), encoding="utf-8"
    )

    assert load_app_config(path.parent).font_family == "Arial, Menlo"


def test_whitespace_font_family_reads_as_the_built_in(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_header("config"), "font_family": "   "}), encoding="utf-8")

    assert load_app_config(path.parent) == AppConfig()


def test_load_app_config_preserves_a_font_stack(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({**_header("config"), "font_family": "Helvetica Neue, Segoe UI, Roboto, Arial"}),
        encoding="utf-8",
    )

    assert load_app_config(path.parent).font_family == "Helvetica Neue, Segoe UI, Roboto, Arial"


def test_load_app_config_ignores_unusable_font_family(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**_header("config"), "font_family": 42}), encoding="utf-8")

    result = load_app_config_result(path.parent)

    assert result.config == AppConfig()
    assert path.exists()


@pytest.mark.parametrize(
    "text",
    ["{ this is not valid json", "[]\n", '{"language": "ja"}'],
    ids=["not-json", "not-an-object", "no-header"],
)
def test_an_unreadable_config_stops_the_launch_untouched(tmp_path: Path, text: str) -> None:
    # Nothing infers a format from the file's shape (file-identity-conventions), and
    # the file is neither set aside nor replaced (store-recovery-conventions).
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    assert _refused(tmp_path) == (RefusedFile(path, newer=False),)
    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


@pytest.mark.parametrize(
    "header",
    [
        {"format": 1},
        {"format": "pixelup.config"},
        {"format": None},
        {"format": {"owner": "pixelup", "kind": "config"}},
        {"format": {"owner": "pixelup", "version": 1}},
        {"format": {"kind": "config", "version": 1}},
        {"format": {"owner": 7, "kind": "config", "version": 1}},
        {"format_version": 1},
        *(
            {"format": {"owner": "pixelup", "kind": "config", "version": version}}
            for version in ("1", 0, -1, 1.5, True, None)
        ),
    ],
)
def test_a_missing_or_malformed_header_is_unreadable(
    tmp_path: Path, header: dict[str, object]
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**header, "language": "ja"}), encoding="utf-8")
    assert _refused(tmp_path) == (RefusedFile(path, newer=False),)


@pytest.mark.parametrize(
    ("name", "owner", "kind"),
    [
        ("config.json", "picpatch", "config"),
        ("config.json", "pixelup", "parameters"),
        ("parameters.json", "pixelup", "config"),
        ("config.json", "Pixelup", "config"),
    ],
)
def test_a_file_of_another_owner_or_kind_is_unreadable(
    tmp_path: Path, name: str, owner: str, kind: str
) -> None:
    # Even when its version is newer: a file of another format is unreadable, not
    # newer, whatever its shape.
    path = tmp_path / name
    header = {"format": {"owner": owner, "kind": kind, "version": 2}}
    text = json.dumps({**header, "language": "ja", "parameters": _parameter_map(quality=1)})
    path.write_text(text, encoding="utf-8")
    assert _refused(tmp_path) == (RefusedFile(path, newer=False),)


@pytest.mark.parametrize("writer", [None, "anything", {"name": "picpatch", "version": 9}, 7])
def test_the_writer_is_never_read(tmp_path: Path, writer: object) -> None:
    path = tmp_path / "config.json"
    data = {"format": {"owner": "pixelup", "kind": "config", "version": 1}, "language": "ja"}
    if writer is not None:
        data["writer"] = writer
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_app_config_result(tmp_path)
    assert result.config == AppConfig(language="ja")
    assert result.files.config.unapplied == frozenset()

    save_app_config(replace(result.config, font_family="Menlo"), result.config, result.files)
    assert json.loads(path.read_text(encoding="utf-8"))["writer"] == _header("config")["writer"]


def test_a_saved_file_starts_with_its_header(tmp_path: Path) -> None:
    _save(AppConfig(language="ja"), AppConfig(), tmp_path / "config.json")
    stored = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert list(stored)[:2] == ["format", "writer"]
    assert stored["format"] == {"owner": "pixelup", "kind": "config", "version": 1}
    assert stored["writer"] == {"name": "pixelup", "version": __version__}


def test_missing_config_is_not_treated_as_corrupt(tmp_path: Path) -> None:
    # The normal first run: no file, defaults, nothing refused.
    result = load_app_config_result(tmp_path)

    assert result.config == AppConfig()


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
    path = path.with_name("parameters.json")
    path.write_text(
        json.dumps({**_header("parameters"), "parameters": _parameter_map(**parameters)}),
        encoding="utf-8",
    )


def test_invalid_utf8_config_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b'{"font_family": "\xff"}')
    assert _refused(tmp_path) == (RefusedFile(path, newer=False),)


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
        load_app_config_result(path.parent)

    assert path.read_bytes() == original
    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


@pytest.mark.parametrize(
    "data",
    [
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
    (tmp_path / "config.json").write_text(
        json.dumps({**_header("config"), "font_family": "Menlo"}), encoding="utf-8"
    )
    path = tmp_path / ("parameters.json" if "parameters" in data else "config.json")
    stored = _content(path) if path.exists() else {}
    path.write_text(json.dumps({**_header(path.stem), **stored, **data}), encoding="utf-8")
    original = path.read_bytes()
    result = load_app_config_result(path.parent)
    assert result.config == AppConfig(font_family="Menlo")
    assert result.rejected == (next(iter(data)),)
    assert path.read_bytes() == original
    warnings = [record for record in caplog.records if record.message == "config.invalid_set"]
    assert len(warnings) == 1
    assert warnings[0].fields["key"] == next(iter(data))


def test_absent_and_unknown_fields_do_not_make_the_config_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    path.write_text(
        json.dumps(
            {
                **_header("parameters"),
                "future_setting": {"future": True},
                "parameters": _parameter_map(scale=2, future_parameter="ignored"),
            }
        ),
        encoding="utf-8",
    )

    result = load_app_config_result(path.parent)

    assert result.config.parameters.scale == 2
    assert result.config.parameters.quality == JobSettings().quality


def test_valid_parameter_boundaries_and_choices_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(
        path,
        quality=0,
        denoise_strength=1.0,
        scale=2,
        tile=128,
        device="cpu",
        output_format="webp",
        alpha_mode="bicubic",
        target_profile="adobergb",
    )

    parameters = load_app_config(path.parent).parameters

    assert parameters.quality == 0
    assert parameters.denoise_strength == 1.0
    assert parameters.scale == 2
    assert parameters.tile == 128
    assert parameters.device == "cpu"
    assert parameters.output_format == OutputFormat.WEBP
    assert parameters.alpha_mode == "bicubic"
    assert parameters.target_profile == "adobergb"


def test_null_target_profile_is_a_valid_choice(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write(path, target_profile=None)

    assert load_app_config(path.parent).parameters.target_profile is None


def test_a_save_that_changes_nothing_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    opened = AppConfig(font_family="Menlo")
    first = _save(opened, AppConfig(), path)
    written_at = path.stat().st_mtime_ns

    saved = save_app_config(opened, opened, first.files).config

    assert saved == opened
    assert path.stat().st_mtime_ns == written_at


def test_config_log_payload_shape() -> None:
    config = AppConfig(
        parameters=JobSettings(quality=70, tile=128, device="cpu"),
    )

    assert config_log_payload(config) == {
        "font_family": AppConfig().font_family,
        "language": "system",
        "display_zone": "system",
        "parameters": job_settings_log_payload(config.parameters),
        "models_folder": "",
    }


def test_partial_parameters_read_as_the_built_in_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({**_header("config"), "language": "ja"}), encoding="utf-8"
    )
    path = tmp_path / "parameters.json"
    path.write_text(
        json.dumps({**_header("parameters"), "parameters": {"scale": 2}}), encoding="utf-8"
    )
    result = load_app_config_result(path.parent)
    assert result.config == AppConfig(language="ja")
    assert result.rejected == ("parameters",)
    warnings = [record for record in caplog.records if record.message == "config.invalid_set"]
    assert len(warnings) == 1
    assert warnings[0].fields["key"] == "parameters"


def test_edit_writes_the_untouched_user_set_whole_from_memory(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({**_header("config"), "font_family": "  Arial,\n Menlo  "}), encoding="utf-8"
    )
    previous = load_app_config(path.parent)
    _save(replace(previous, language="ja"), previous, path)
    assert _content(path) == {
        "font_family": "Arial, Menlo",
        "language": "ja",
    }


def test_a_parameters_save_writes_the_set_whole_from_memory(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    stored_parameters = _parameter_map(quality=42, future_parameter="dropped")
    path.write_text(
        json.dumps({**_header("parameters"), "parameters": stored_parameters}), encoding="utf-8"
    )
    previous = load_app_config(path.parent)
    _save(replace(previous, parameters=JobSettings(quality=43)), previous, path)
    assert _content(path) == {
        "parameters": _parameter_map(quality=43),
    }


def test_models_folder_is_saved_reset_and_preserved_when_rejected(tmp_path):
    from dataclasses import replace

    from pixelup.app_config import AppConfig, load_app_config_result, save_app_config

    path = tmp_path / "config.json"
    initial = load_app_config_result(path.parent)
    chosen = replace(initial.config, models_folder=str(tmp_path / "external models"))
    saved = save_app_config(chosen, initial.config, initial.files)
    assert load_app_config_result(path.parent).config.models_folder == chosen.models_folder
    save_app_config(AppConfig(), chosen, saved.files)
    assert "models_folder" not in _content(path)
    path.write_text(json.dumps({**_header("config"), "models_folder": 123}))
    rejected = load_app_config_result(path.parent)
    assert rejected.rejected == ("models_folder",)
    save_app_config(replace(rejected.config, language="ja"), rejected.config, rejected.files)
    assert _content(path)["models_folder"] == 123


def test_display_zone_is_saved_and_preserved_when_this_computer_lacks_it(tmp_path):
    from dataclasses import replace

    path = tmp_path / "config.json"
    initial = load_app_config_result(path.parent)
    assert initial.config.display_zone == "system"
    chosen = replace(initial.config, display_zone="Asia/Tokyo")
    save_app_config(chosen, initial.config, initial.files)
    assert _content(path)["display_zone"] == "Asia/Tokyo"
    assert load_app_config_result(path.parent).config.display_zone == "Asia/Tokyo"

    path.write_text(json.dumps({**_header("config"), "display_zone": "Mars/Olympus_Mons"}))
    rejected = load_app_config_result(path.parent)
    assert rejected.rejected == ("display_zone",)
    assert rejected.config.display_zone == "system"
    save_app_config(replace(rejected.config, language="ja"), rejected.config, rejected.files)
    assert _content(path)["display_zone"] == "Mars/Olympus_Mons"


def test_successful_save_returns_the_saved_state(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    initial = load_app_config_result(path.parent)
    candidate = replace(initial.config, language="ja")

    saved = save_app_config(candidate, initial.config, initial.files)

    assert saved.config == candidate
    assert saved.files.config.exists
    assert not saved.files.parameters.exists
    assert load_app_config(path.parent) == candidate
