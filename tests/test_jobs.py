from __future__ import annotations

from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest

from pixelup.devices import DEFAULT_DEVICE
from pixelup.jobs import (
    DEFAULT_SCALE,
    DEFAULT_TILE,
    MIN_TILE,
    SCALE_CHOICES,
    SCALE_VALUES,
    Job,
    JobSettings,
    coerce_output_format,
    create_jobs,
    job_log_payload,
    job_settings_log_payload,
    job_status_summary,
    options_for_job,
    retry_failed_jobs,
    settings_for_model,
)
from pixelup.paths import OutputFormat


def test_coerce_output_format_accepts_string_values() -> None:
    assert coerce_output_format("png") == OutputFormat.PNG
    assert coerce_output_format("jpg") == OutputFormat.JPG


def test_coerce_output_format_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError):
        coerce_output_format(123)


def test_settings_for_model_forces_denoise_for_non_general_model() -> None:
    adjusted = settings_for_model(JobSettings(denoise_strength=0.5), "RealESRGAN_x4plus")

    assert adjusted.denoise_strength == 1.0


def test_settings_for_model_keeps_settings_for_general_model() -> None:
    base = JobSettings(denoise_strength=0.5)

    assert settings_for_model(base, "realesr-general-x4v3") is base


def test_options_for_job_maps_settings_and_fixed_defaults(tmp_path: Path) -> None:
    job = Job(
        id=1,
        input_path=tmp_path / "a.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / "out.jpg",
        settings=JobSettings(
            scale=2,
            face_enhance=True,
            denoise_strength=0.5,
            alpha_mode="bicubic",
            device="cpu",
            output_format=OutputFormat.JPG,
            quality=80,
            tile=128,
            strip_metadata=True,
            target_profile="srgb",
        ),
        auto_download=False,
    )

    options = options_for_job(job)

    assert options.input_path == job.input_path
    assert options.output_arg == str(job.output_path)
    assert options.model == "realesr-general-x4v3"
    assert options.scale == 2
    assert options.tile == 128
    assert options.quality == 80
    assert options.output_format == OutputFormat.JPG
    assert options.face_enhance is True
    assert options.alpha_mode == "bicubic"
    assert options.target_profile == "srgb"
    assert options.auto_download is False
    # Fixed, non-configurable defaults the GUI never exposes.
    assert options.tile_pad == 10
    assert options.pre_pad == 0
    assert options.background == "white"
    assert options.overwrite is False
    assert options.download_timeout == 600
    assert options.lock_timeout == 600


def test_bare_job_settings_defaults_to_a_safe_tile() -> None:
    # The whole point of the constant: tiling is ON by default, so peak memory scales
    # with the tile rather than the image. tile=0 (a whole-image pass) can exhaust
    # GPU/MPS memory and hard-crash on a large input, and JobSettings() is what the
    # panel's reset hands the user, so the bare dataclass must never carry 0.
    assert JobSettings().tile == DEFAULT_TILE
    assert JobSettings().tile != MIN_TILE
    assert DEFAULT_TILE == 256


def test_bare_job_settings_are_the_built_in_defaults() -> None:
    # JobSettings() *is* the single source of the built-ins — there is no second
    # defaults layer to drift against — so every field must stand on its own as the
    # value the app ships with, not a placeholder some other layer overwrites.
    defaults = JobSettings()

    assert defaults.scale == DEFAULT_SCALE
    assert defaults.face_enhance is False
    assert defaults.denoise_strength == 0.5
    assert defaults.alpha_mode == "realesrgan"
    assert defaults.device == DEFAULT_DEVICE
    assert defaults.output_format == OutputFormat.PNG
    assert defaults.quality == 95
    assert defaults.tile == DEFAULT_TILE
    assert defaults.strip_metadata is False
    assert defaults.target_profile is None


def test_bare_job_settings_defaults_to_the_scale_the_panel_always_opened_on() -> None:
    # Scale folded into JobSettings, and the fold must not have moved the value: 4x is
    # what the panel checked before it was persisted or resettable, so it is what the
    # built-in has to be. Pinned against the constant *and* the literal, because the
    # constant alone would happily follow a typo.
    assert JobSettings().scale == DEFAULT_SCALE
    assert DEFAULT_SCALE == 4


def test_scale_choices_are_the_two_selectable_scales() -> None:
    # The same 2x/4x the panel has always offered — enumerated once, beside
    # JobSettings, for the panel and the config loader both.
    assert SCALE_VALUES == (2, 4)
    assert SCALE_CHOICES == (("2x", 2), ("4x", 4))
    assert DEFAULT_SCALE in SCALE_VALUES


def test_zero_tile_stays_selectable() -> None:
    # 0 stops being the default; it does not stop being a choice a power user can make.
    assert MIN_TILE == 0
    assert JobSettings(tile=MIN_TILE).tile == 0


def test_job_log_payload_includes_settings_and_paths(tmp_path: Path) -> None:
    job = Job(
        id=7,
        input_path=tmp_path / "a.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / "o.png",
        settings=JobSettings(scale=2),
        auto_download=True,
    )

    payload = job_log_payload(job)

    assert payload["input_path"] == str(tmp_path / "a.png")
    assert payload["output_path"] == str(tmp_path / "o.png")
    assert payload["model"] == "realesr-general-x4v3"
    assert payload["auto_download"] is True
    assert payload["settings"]["output_format"] == "png"  # type: ignore[index]
    # Scale is logged as part of the settings snapshot, and only there — a second
    # top-level copy would be a place for the two to disagree.
    assert payload["settings"]["scale"] == 2  # type: ignore[index]
    assert "scale" not in payload


def test_job_status_summary_empty() -> None:
    assert job_status_summary([]) == "No jobs"


def test_job_status_summary_omits_zero_counts_and_groups_queued() -> None:
    statuses = [
        "succeeded",
        "succeeded",
        "failed",
        "cancelled",
        "pending",
        "running",
        "cancelling",
    ]

    assert job_status_summary(statuses) == "2 done, 1 failed, 1 cancelled, 3 queued"


def test_job_status_summary_orders_done_then_queued() -> None:
    assert job_status_summary(["pending", "succeeded"]) == "1 done, 1 queued"


def test_job_settings_log_payload_accepts_string_backed_output_format() -> None:
    payload = job_settings_log_payload(JobSettings(output_format="webp"))  # type: ignore[arg-type]

    assert payload["output_format"] == "webp"


def test_create_jobs_uses_all_inputs_and_models(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"")
    second.write_bytes(b"")

    jobs = create_jobs(
        input_paths=[first, second],
        models=["realesr-general-x4v3", "RealESRGAN_x4plus"],
        settings=JobSettings(output_format=OutputFormat.PNG),
        existing_jobs=[],
        auto_download=True,
        job_ids=count(1),
    )

    assert [(job.id, job.input_path, job.model) for job in jobs] == [
        (1, first, "realesr-general-x4v3"),
        (2, first, "RealESRGAN_x4plus"),
        (3, second, "realesr-general-x4v3"),
        (4, second, "RealESRGAN_x4plus"),
    ]
    assert all(job.auto_download for job in jobs)
    assert {job.output_path.name for job in jobs} == {
        "a-realesr-general-x4v3-4x.png",
        "a-realesrgan-x4plus-4x.png",
        "b-realesr-general-x4v3-4x.png",
        "b-realesrgan-x4plus-4x.png",
    }


def test_create_jobs_takes_scale_from_the_settings_snapshot(tmp_path: Path) -> None:
    # Scale arrives inside the settings snapshot, not beside it: the job freezes it,
    # the output name is planned from it, and options_for_job hands that same value to
    # the upscaler. One value, one carrier, all the way down.
    source = tmp_path / "a.png"
    source.write_bytes(b"")

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(scale=2),
        existing_jobs=[],
        auto_download=True,
        job_ids=count(1),
    )

    assert jobs[0].settings.scale == 2
    assert jobs[0].output_path.name == "a-realesr-general-x4v3-2x.png"
    assert options_for_job(jobs[0]).scale == 2


def test_create_jobs_snapshot_is_frozen_against_later_panel_edits(tmp_path: Path) -> None:
    # The enqueue-snapshot semantic, at the planner: the JobSettings a job holds is a
    # frozen value, so nothing the panel does afterwards can reach back into queued
    # work. dataclasses.replace models the panel moving on — it builds a new settings
    # object and must leave the queued job's own untouched.
    source = tmp_path / "a.png"
    source.write_bytes(b"")
    panel = JobSettings(scale=2)

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=panel,
        existing_jobs=[],
        auto_download=True,
        job_ids=count(1),
    )
    queued = jobs[0]

    moved_on = replace(panel, scale=4)

    assert moved_on.scale == 4
    assert queued.settings.scale == 2
    assert options_for_job(queued).scale == 2
    # And the output it already planned still names the scale it was queued with.
    assert queued.output_path.name == "a-realesr-general-x4v3-2x.png"


def test_create_jobs_disambiguates_case_only_sibling_inputs(tmp_path: Path) -> None:
    # Photo.png and photo.png live in one directory; their default output stems
    # differ only in case and would clobber each other on macOS/Windows. The
    # batch-wide reservation set must give them distinct output filenames.
    upper = tmp_path / "Photo.png"
    lower = tmp_path / "photo.png"
    upper.write_bytes(b"")
    lower.write_bytes(b"")

    jobs = create_jobs(
        input_paths=[upper, lower],
        models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.PNG),
        existing_jobs=[],
        auto_download=True,
        job_ids=count(1),
    )

    assert len(jobs) == 2
    # Human-derived names are preserved (not lowercased); only the disambiguation
    # suffix separates the two, and their names differ case-insensitively.
    names = [job.output_path.name for job in jobs]
    assert names == [
        "Photo-realesr-general-x4v3-4x.png",
        "photo-realesr-general-x4v3-4x-2.png",
    ]
    assert len({name.casefold() for name in names}) == 2


def test_create_jobs_separates_sidecars_across_output_formats(tmp_path: Path) -> None:
    # Same image + model + scale, different output formats — the two jobs share
    # the default stem, so a single sidecar JSON would overwrite. The planner
    # must give them distinct stems.
    source = tmp_path / "a.png"
    source.write_bytes(b"")
    png_job = Job(
        id=1,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(output_format=OutputFormat.PNG),
        auto_download=True,
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.JPG),
        existing_jobs=[png_job],
        auto_download=True,
        job_ids=count(2),
    )

    assert len(jobs) == 1
    assert jobs[0].output_path.with_suffix(".json") != png_job.output_path.with_suffix(".json")
    assert jobs[0].output_path.name == "a-realesr-general-x4v3-4x-2.jpg"


def test_create_jobs_reserves_existing_output_paths(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    source.write_bytes(b"")
    existing = Job(
        id=1,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(),
        auto_download=True,
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.PNG),
        existing_jobs=[existing],
        auto_download=True,
        job_ids=count(2),
    )

    assert jobs[0].output_path.name == "a-realesr-general-x4v3-4x-2.png"


def test_retry_failed_jobs_replans_outputs(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    source.write_bytes(b"")
    succeeded = Job(
        id=1,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(),
        auto_download=True,
        status="succeeded",
    )
    failed = Job(
        id=2,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "old.png",
        settings=JobSettings(),
        auto_download=True,
        status="failed",
        message="bad",
        warnings=["warning"],
    )

    assert retry_failed_jobs([succeeded, failed]) == [2]
    assert failed.status == "pending"
    assert failed.message == ""
    assert failed.warnings == []
    assert failed.output_path.name == "a-realesr-general-x4v3-4x-2.png"
