from __future__ import annotations

from itertools import count
from pathlib import Path

import pytest

from pixelup.app_config import AppConfig
from pixelup.jobs import (
    Job,
    JobSettings,
    coerce_output_format,
    config_log_payload,
    create_jobs,
    job_log_payload,
    job_settings_defaults,
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
        scale=2,
        output_path=tmp_path / "out.jpg",
        settings=JobSettings(
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


def test_config_log_payload_shape() -> None:
    payload = config_log_payload(
        AppConfig(
            max_concurrent_jobs=2,
            output_format=OutputFormat.JPG,
            quality=70,
            tile=128,
            device="cpu",
            auto_download=False,
        )
    )

    assert payload == {
        "max_concurrent_jobs": 2,
        "output_format": "jpg",
        "quality": 70,
        "tile": 128,
        "device": "cpu",
        "auto_download": False,
        "font_family": AppConfig().font_family,
    }


def test_job_log_payload_includes_settings_and_paths(tmp_path: Path) -> None:
    job = Job(
        id=7,
        input_path=tmp_path / "a.png",
        model="realesr-general-x4v3",
        scale=4,
        output_path=tmp_path / "o.png",
        settings=JobSettings(),
        auto_download=True,
    )

    payload = job_log_payload(job)

    assert payload["input_path"] == str(tmp_path / "a.png")
    assert payload["output_path"] == str(tmp_path / "o.png")
    assert payload["model"] == "realesr-general-x4v3"
    assert payload["scale"] == 4
    assert payload["auto_download"] is True
    assert payload["settings"]["output_format"] == "png"  # type: ignore[index]


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


def test_job_settings_defaults_come_from_app_config() -> None:
    settings = job_settings_defaults(
        AppConfig(
            output_format=OutputFormat.WEBP,
            quality=82,
            tile=256,
            device="cpu",
        )
    )

    assert settings.output_format == OutputFormat.WEBP
    assert settings.quality == 82
    assert settings.tile == 256
    assert settings.device == "cpu"


def test_create_jobs_uses_all_inputs_and_models(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"")
    second.write_bytes(b"")

    jobs = create_jobs(
        input_paths=[first, second],
        models=["realesr-general-x4v3", "RealESRGAN_x4plus"],
        scale=4,
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
        scale=4,
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
        scale=4,
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(output_format=OutputFormat.PNG),
        auto_download=True,
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        scale=4,
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
        scale=4,
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(),
        auto_download=True,
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        scale=4,
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
        scale=4,
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(),
        auto_download=True,
        status="succeeded",
    )
    failed = Job(
        id=2,
        input_path=source,
        model="realesr-general-x4v3",
        scale=4,
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
