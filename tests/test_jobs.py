from __future__ import annotations

from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest
from PIL import Image

from pixelup import upscale as upscale_module
from pixelup.config import RuntimeDirs
from pixelup.devices import DEFAULT_DEVICE
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.jobs import (
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
from pixelup.output_reservation import BUNDLE_SUFFIXES, assert_output_bundle_available
from pixelup.parameters import (
    DEFAULT_SCALE,
    DEFAULT_TILE,
    SCALE_CHOICES,
    SCALE_VALUES,
    TILE_CHOICES,
    TILE_VALUES,
)
from pixelup.paths import NamePlanCache, OutputFormat
from pixelup.upscale import build_plan


@pytest.mark.parametrize(
    ("directory_name", "filename"),
    [
        ("album{2026}", "photo.png"),
        ("album", "photo{edited}.png"),
        ("album{2026}", "photo{edited}.png"),
    ],
)
def test_planned_output_preserves_literal_braces_through_the_processing_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory_name: str,
    filename: str,
) -> None:
    directory = tmp_path / directory_name
    directory.mkdir()
    source = directory / filename
    with Image.new("RGB", (2, 3), "white") as image:
        image.save(source)
    monkeypatch.setattr(upscale_module, "resolve_device", lambda _device, _gpu: "cpu")
    job = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(scale=2, output_format=OutputFormat.WEBP),
        existing_jobs=[],
        job_ids=count(1),
    )[0]

    plan = build_plan(
        options_for_job(job),
        RuntimeDirs(tmp_path / "models", tmp_path / "temp"),
        check_model=False,
    )

    assert plan.input_path == source
    assert plan.output_path == job.output_path
    assert plan.output_path == directory / f"{source.stem}-realesr-general-x4v3-2x.webp"
    assert plan.output_format == OutputFormat.WEBP
    assert plan.input_size == (2, 3)
    assert plan.output_size == (4, 6)


@pytest.mark.parametrize("planning_pass", ["create", "retry"])
def test_large_batch_reservation_work_grows_with_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planning_pass: str
) -> None:
    # 1,000 images compared with six models produce 6,000 jobs. Count collision
    # identity work, rather than elapsed time: revisiting all earlier reservations
    # would block the window with quadratic work even if each key were cached.
    models = [
        "realesr-general-x4v3",
        "RealESRGAN_x4plus",
        "RealESRNet_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x2plus",
        "realesr-animevideov3",
    ]
    inputs = [tmp_path / f"image-{index:04d}.png" for index in range(1000)]
    job_count = len(inputs) * len(models)
    identity_operations = 0
    real_key = NamePlanCache.key

    def counted_key(self: NamePlanCache, path: Path) -> str:
        nonlocal identity_operations
        identity_operations += 1
        return real_key(self, path)

    monkeypatch.setattr(NamePlanCache, "key", counted_key)
    if planning_pass == "create":
        jobs = create_jobs(
            input_paths=inputs,
            models=models,
            settings=JobSettings(),
            existing_jobs=[],
            job_ids=count(1),
        )
    else:
        jobs = [
            Job(
                id=index,
                input_path=input_path,
                model=model,
                output_path=tmp_path / f"old-{index}.png",
                settings=JobSettings(),
                status="failed",
            )
            for index, (input_path, model) in enumerate(
                ((path, model) for path in inputs for model in models), start=1
            )
        ]
        assert retry_failed_jobs(jobs) == list(range(1, job_count + 1))

    assert len(jobs) == job_count
    assert jobs[0].output_path.name == "image-0000-realesr-general-x4v3-4x.png"
    assert jobs[-1].output_path.name == "image-0999-realesr-animevideov3-4x.png"
    assert len({job.output_path for job in jobs}) == job_count
    # At most one reservation and one candidate bundle per job; allow the
    # candidate itself as well as every image/sidecar suffix in each bundle.
    assert identity_operations <= 2 * (len(BUNDLE_SUFFIXES) + 1) * job_count


@pytest.mark.parametrize("extension", ["png", "JPG", "jpeg", "webp", "json"])
def test_create_and_retry_avoid_every_normalized_output_bundle_member(tmp_path, extension):
    # An image may legitimately survive without its sidecar. Canonically
    # equivalent Unicode names must agree with publication on Windows and Mac.
    source = tmp_path / "cafe\u0301.png"
    occupied = tmp_path / f"caf\u00e9-realesr-general-x4v3-4x.{extension}"
    occupied.write_bytes(b"keep")
    jobs = create_jobs(
        input_paths=[source], models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.WEBP),
        existing_jobs=[], job_ids=count(1),
    )
    job = jobs[0]
    assert job.output_path.stem.endswith("-2")
    assert_output_bundle_available(job.output_path)
    job.output_path.with_suffix(".PNG").write_bytes(b"partial bundle")
    job.status = "failed"
    assert retry_failed_jobs(jobs) == [job.id]
    assert job.output_path.stem.endswith("-3")
    assert_output_bundle_available(job.output_path)
    assert occupied.read_bytes() == b"keep"


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
            denoise_strength=0.5,
            alpha_mode="bicubic",
            device="cpu",
            output_format=OutputFormat.JPG,
            quality=80,
            tile=128,
            strip_metadata=True,
            target_profile="srgb",
        ),
    )

    options = options_for_job(job)

    assert options.input_path == job.input_path
    assert options.output_path is job.output_path
    assert options.model == "realesr-general-x4v3"
    assert options.scale == 2
    assert options.tile == 128
    assert options.quality == 80
    assert options.output_format == OutputFormat.JPG
    assert options.alpha_mode == "bicubic"
    assert options.target_profile == "srgb"
    # Fixed, non-configurable defaults the GUI never exposes.
    assert options.tile_pad == 10
    assert options.pre_pad == 0
    assert options.background == "white"


def test_bare_job_settings_defaults_to_a_safe_tile() -> None:
    # JobSettings() is what the panel's reset hands the user.
    assert JobSettings().tile == DEFAULT_TILE
    assert DEFAULT_TILE == 256


def test_bare_job_settings_are_the_built_in_defaults() -> None:
    # JobSettings() *is* the single source of the built-ins — there is no second
    # defaults layer to drift against — so every field must stand on its own as the
    # value the app ships with, not a placeholder some other layer overwrites.
    defaults = JobSettings()

    assert defaults.scale == DEFAULT_SCALE
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


def test_tile_choices_are_doublings_without_an_untiled_pass() -> None:
    # Peak memory scales with the tile's AREA, so only doublings are meaningful. There
    # is no untiled "Whole image" choice: a tile larger than the image is one pass.
    assert TILE_VALUES == (128, 256, 512, 1024, 2048)
    assert all(isinstance(label, str) for label, _value in TILE_CHOICES)
    assert DEFAULT_TILE in TILE_VALUES
    # 128 must be one rung below the default: the machine that cannot fit 256 has
    # somewhere to go.
    assert TILE_VALUES[TILE_VALUES.index(DEFAULT_TILE) - 1] == 128


def test_job_log_payload_includes_settings_and_paths(tmp_path: Path) -> None:
    job = Job(
        id=7,
        input_path=tmp_path / "a.png",
        model="realesr-general-x4v3",
        output_path=tmp_path / "o.png",
        settings=JobSettings(scale=2),
    )

    payload = job_log_payload(job)

    assert payload["input_path"] == str(tmp_path / "a.png")
    assert payload["output_path"] == str(tmp_path / "o.png")
    assert payload["model"] == "realesr-general-x4v3"
    assert payload["settings"]["output_format"] == "png"  # type: ignore[index]
    # Scale is logged as part of the settings snapshot, and only there — a second
    # top-level copy would be a place for the two to disagree.
    assert payload["settings"]["scale"] == 2  # type: ignore[index]
    assert "scale" not in payload


def test_job_status_summary_empty() -> None:
    assert english().of(job_status_summary([])) == "No jobs"


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

    assert english().of(job_status_summary(statuses)) == "2 done, 1 failed, 1 cancelled, 3 queued"


def test_job_status_summary_orders_done_then_queued() -> None:
    assert english().of(job_status_summary(["pending", "succeeded"])) == "1 done, 1 queued"


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
        job_ids=count(1),
    )

    assert [(job.id, job.input_path, job.model) for job in jobs] == [
        (1, first, "realesr-general-x4v3"),
        (2, first, "RealESRGAN_x4plus"),
        (3, second, "realesr-general-x4v3"),
        (4, second, "RealESRGAN_x4plus"),
    ]
    assert {job.output_path.name for job in jobs} == {
        "a-realesr-general-x4v3-4x.png",
        "a-realesrgan-x4plus-4x.png",
        "b-realesr-general-x4v3-4x.png",
        "b-realesrgan-x4plus-4x.png",
    }


def test_a_batch_reads_each_output_folder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Planning runs on the window thread: a batch reads its output folder once,
    # not once per candidate name, so a large or slow folder costs one listing.
    inputs = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"")
        inputs.append(path)
    (tmp_path / "a-realesr-general-x4v3-4x.png").write_bytes(b"taken")
    listed: list[Path] = []
    real_iterdir = Path.iterdir

    def counted_iterdir(self: Path):
        listed.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counted_iterdir)
    jobs = create_jobs(
        input_paths=inputs,
        models=["realesr-general-x4v3", "RealESRGAN_x4plus"],
        settings=JobSettings(output_format=OutputFormat.PNG),
        existing_jobs=[],
        job_ids=count(1),
    )

    assert len(jobs) == 6
    assert jobs[0].output_path.name == "a-realesr-general-x4v3-4x-2.png"
    assert listed == [tmp_path.resolve()] or listed == [tmp_path]


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
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.JPG),
        existing_jobs=[png_job],
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
    )

    jobs = create_jobs(
        input_paths=[source],
        models=["realesr-general-x4v3"],
        settings=JobSettings(output_format=OutputFormat.PNG),
        existing_jobs=[existing],
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
        status="succeeded",
    )
    failed = Job(
        id=2,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "old.png",
        settings=JobSettings(),
        status="failed",
        message=Message("jobs.failedFallback"),
        warnings=[Message("warning.extensionMismatch")],
    )

    assert retry_failed_jobs([succeeded, failed]) == [2]
    assert failed.status == "pending"
    assert failed.message is None
    assert failed.warnings == []
    assert failed.output_path.name == "a-realesr-general-x4v3-4x-2.png"


def test_retry_failed_jobs_limits_work_to_the_disclosed_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    source.write_bytes(b"")
    disclosed = Job(
        id=1,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "old-disclosed.png",
        settings=JobSettings(),
        status="failed",
        message="first",
    )
    later_failure = Job(
        id=2,
        input_path=source,
        model="realesr-general-x4v3",
        output_path=tmp_path / "a-realesr-general-x4v3-4x.png",
        settings=JobSettings(),
        status="failed",
        message="later",
    )

    assert retry_failed_jobs(
        [disclosed, later_failure], only_job_ids=frozenset({disclosed.id})
    ) == [disclosed.id]
    assert disclosed.status == "pending"
    assert disclosed.output_path.name == "a-realesr-general-x4v3-4x-2.png"
    assert later_failure.status == "failed"
    assert later_failure.message == "later"
    assert later_failure.output_path.name == "a-realesr-general-x4v3-4x.png"
