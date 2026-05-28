from __future__ import annotations

from itertools import count
from pathlib import Path

from pixelup.app_config import AppConfig
from pixelup.jobs import (
    Job,
    JobSettings,
    coerce_output_format,
    create_jobs,
    job_settings_defaults,
    job_settings_log_payload,
    retry_failed_jobs,
)
from pixelup.paths import OutputFormat


def test_coerce_output_format_accepts_string_values() -> None:
    assert coerce_output_format("png") == OutputFormat.PNG
    assert coerce_output_format("jpg") == OutputFormat.JPG


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
