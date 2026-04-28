from __future__ import annotations

import importlib.metadata
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from pixelup import __version__
from pixelup.config import ensure_models_dir, resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError, exit_code_for
from pixelup.models import (
    KNOWN_MODELS,
    all_model_names,
    download_models,
    list_model_records,
    model_file,
    verify_present_models,
)
from pixelup.paths import OutputFormat
from pixelup.reporting import Reporter, ReportMode
from pixelup.signals import CANCELLED_EXIT_CODE, OperationCancelled, cancellation_context
from pixelup.upscale import UpscaleOptions, run_upscale

HELP_OPTIONS = {"help_option_names": ["--help", "-h"]}


class AlphaMode(StrEnum):
    REALESRGAN = "realesrgan"
    BICUBIC = "bicubic"


class DeviceMode(StrEnum):
    AUTO = "auto"
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class TargetProfile(StrEnum):
    SRGB = "srgb"
    P3 = "p3"
    ADOBERGB = "adobergb"

root_app = typer.Typer(
    add_completion=True,
    context_settings=HELP_OPTIONS,
    help=(
        "PixelUp upscales one image file to one output file.\n\n"
        "Bare upscale: pixelup INPUT OUTPUT [OPTIONS]\n"
        "Model commands: pixelup models COMMAND [OPTIONS]\n"
        "Version: pixelup --version"
    ),
)
upscale_app = typer.Typer(add_completion=False, context_settings=HELP_OPTIONS)
models_app = typer.Typer(add_completion=False, context_settings=HELP_OPTIONS, help="Manage models.")
root_app.add_typer(models_app, name="models")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--version":
        try:
            _version_command(args[1:])
        except typer.Exit as exc:
            raise SystemExit(exc.exit_code) from None
        return
    if not args or args[0] in {"--help", "-h", "--install-completion", "--show-completion"}:
        root_app(args=args, prog_name="pixelup")
        return
    if args[0] == "models":
        models_app(args=args[1:], prog_name="pixelup models")
        return
    upscale_app(args=args, prog_name="pixelup")


@upscale_app.command(help="Upscale one image file to one output file.")
def upscale(
    input_path: Annotated[Path, typer.Argument(help="Input image path.")],
    output: Annotated[str, typer.Argument(help="Output file, directory, or placeholder path.")],
    model: Annotated[str, typer.Option("--model")] = "RealESRGAN_x4plus",
    scale: Annotated[int, typer.Option("--scale")] = 4,
    tile: Annotated[int, typer.Option("--tile")] = 0,
    tile_pad: Annotated[int, typer.Option("--tile-pad")] = 10,
    pre_pad: Annotated[int, typer.Option("--pre-pad")] = 0,
    fp32: Annotated[bool, typer.Option("--fp32")] = False,
    face_enhance: Annotated[bool, typer.Option("--face-enhance")] = False,
    denoise_strength: Annotated[float, typer.Option("--denoise-strength")] = 1.0,
    alpha_mode: Annotated[AlphaMode, typer.Option("--alpha-mode")] = AlphaMode.REALESRGAN,
    gpu_id: Annotated[int | None, typer.Option("--gpu-id")] = None,
    device: Annotated[DeviceMode, typer.Option("--device")] = DeviceMode.AUTO,
    output_format: Annotated[OutputFormat | None, typer.Option("--format")] = None,
    quality: Annotated[int, typer.Option("--quality")] = 95,
    background: Annotated[str, typer.Option("--background")] = "white",
    strip_metadata: Annotated[bool, typer.Option("--strip-metadata")] = False,
    target_profile: Annotated[TargetProfile | None, typer.Option("--target-profile")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    auto_download: Annotated[bool, typer.Option("--auto-download")] = False,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    temp_dir: Annotated[Path | None, typer.Option("--temp-dir")] = None,
    download_timeout: Annotated[int, typer.Option("--download-timeout")] = 600,
    lock_timeout: Annotated[int, typer.Option("--lock-timeout")] = 600,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    reporter = Reporter(report, quiet=quiet, verbose=verbose)
    try:
        if quiet and verbose:
            raise PixelupError(
                ErrorCode.INVALID_ARGUMENT,
                "--quiet and --verbose are mutually exclusive.",
            )
        with cancellation_context():
            runtime_dirs = resolve_runtime_dirs(models_dir=models_dir, temp_dir=temp_dir)
            options = UpscaleOptions(
                input_path=input_path,
                output_arg=output,
                model=model,
                scale=scale,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                fp32=fp32,
                face_enhance=face_enhance,
                denoise_strength=denoise_strength,
                alpha_mode=alpha_mode.value,
                gpu_id=gpu_id,
                device=device.value,
                output_format=output_format,
                quality=quality,
                background=background,
                strip_metadata=strip_metadata,
                target_profile=target_profile.value if target_profile else None,
                overwrite=overwrite,
                auto_download=auto_download,
                download_timeout=download_timeout,
                lock_timeout=lock_timeout,
                dry_run=dry_run,
            )
            reporter.result(run_upscale(options, runtime_dirs, **_run_callbacks(reporter)))
    except OperationCancelled as exc:
        reporter.error(exc)
        raise typer.Exit(CANCELLED_EXIT_CODE) from exc
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("list", help="List known model files and presence status.")
def models_list(
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        ensure_models_dir(runtime_dirs.models_dir)
        records = list_model_records(runtime_dirs.models_dir)
        _emit_models_records(reporter, runtime_dirs.models_dir, records)
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("check", help="Check model presence, optionally downloading missing models.")
def models_check(
    models: Annotated[list[str] | None, typer.Argument()] = None,
    download_missing: Annotated[bool, typer.Option("--download-missing")] = False,
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    download_timeout: Annotated[int, typer.Option("--download-timeout")] = 600,
    lock_timeout: Annotated[int, typer.Option("--lock-timeout")] = 600,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        if download_missing:
            requested = models or [info.name for info in KNOWN_MODELS]
            runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
            download_models(
                runtime_dirs.models_dir,
                requested,
                download_timeout=download_timeout,
                lock_timeout=lock_timeout,
                **_download_callbacks(reporter),
            )
            records = list_model_records(runtime_dirs.models_dir, requested)
            _emit_models_records(reporter, runtime_dirs.models_dir, records)
            return
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        ensure_models_dir(runtime_dirs.models_dir)
        records = list_model_records(runtime_dirs.models_dir, models)
        _emit_models_records(reporter, runtime_dirs.models_dir, records)
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("download", help="Download known model files.")
def models_download(
    models: Annotated[list[str], typer.Argument()],
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    download_timeout: Annotated[int, typer.Option("--download-timeout")] = 600,
    lock_timeout: Annotated[int, typer.Option("--lock-timeout")] = 600,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        if not models:
            raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Specify at least one model.")
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        results = download_models(
            runtime_dirs.models_dir,
            models,
            download_timeout=download_timeout,
            lock_timeout=lock_timeout,
            **_download_callbacks(reporter),
        )
        reporter.result(
            {
                "ok": True,
                "models_dir": str(runtime_dirs.models_dir),
                "models": results,
            }
        )
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("remove", help="Remove model files from the models directory.")
def models_remove(
    models: Annotated[list[str] | None, typer.Argument()] = None,
    all_models: Annotated[bool, typer.Option("--all")] = False,
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        if not all_models and not models:
            raise PixelupError(
                ErrorCode.INVALID_ARGUMENT,
                "Specify at least one model or use --all.",
        )
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        names = all_model_names(include_unlisted=True) if all_models else list(models or [])
        removed: list[str] = []
        if runtime_dirs.models_dir.is_dir():
            for name in names:
                path = model_file(runtime_dirs.models_dir, name)
                if path.is_file():
                    path.unlink()
                    removed.append(name)
        payload = {"ok": True, "models_dir": str(runtime_dirs.models_dir), "removed": removed}
        reporter.result(payload)
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("verify", help="Verify present model file sizes and checksums.")
def models_verify(
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        ensure_models_dir(runtime_dirs.models_dir)
        payload = {
            "ok": True,
            "models_dir": str(runtime_dirs.models_dir),
            "models": verify_present_models(runtime_dirs.models_dir),
        }
        reporter.result(payload)
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


@models_app.command("dir", help="Print the resolved models directory.")
def models_dir_command(
    report: Annotated[ReportMode, typer.Option("--report")] = ReportMode.AUTO,
    models_dir: Annotated[Path | None, typer.Option("--models-dir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    del verbose
    reporter = Reporter(report, quiet=quiet)
    try:
        runtime_dirs = resolve_runtime_dirs(models_dir=models_dir)
        if reporter.is_human:
            reporter.info(str(runtime_dirs.models_dir))
        else:
            reporter.success({"ok": True, "models_dir": str(runtime_dirs.models_dir)})
    except PixelupError as exc:
        reporter.error(exc)
        raise typer.Exit(exit_code_for(exc.code)) from exc


def _emit_models_records(
    reporter: Reporter,
    models_dir: Path,
    records: list[dict[str, object]],
) -> None:
    if reporter.is_human:
        table = Table("Name", "Alias", "Status", "Size")
        for record in records:
            size = record["size_bytes"]
            table.add_row(
                str(record["name"]),
                str(record["alias"] or ""),
                "present" if record["present"] else "missing",
                str(size) if size is not None else "",
            )
        reporter.table(table)
        return
    reporter.result({"models_dir": str(models_dir), "models": records})


def _download_callbacks(reporter: Reporter) -> dict[str, object]:
    return {
        "on_download": lambda model, done, total: reporter.download(
            model=model,
            bytes_done=done,
            bytes_total=total,
        ),
        "on_waiting": lambda model, waited: reporter.waiting(
            reason="model_download_in_progress",
            model=model,
            seconds_waited=waited,
        ),
    }


def _run_callbacks(reporter: Reporter) -> dict[str, object]:
    return {
        **_download_callbacks(reporter),
        "on_start": lambda plan, tiles: reporter.start(
            input_path=str(plan.input_path),
            output_path=str(plan.output_path),
            model=plan.model,
            scale=plan.scale,
            tiles=tiles,
        ),
        "on_progress": lambda phase: reporter.progress(phase=phase),
        "on_tile": lambda tile, tiles: reporter.progress(
            phase="upscale", tile=tile, tiles=tiles
        ),
        "on_warning": reporter.warning,
    }


def _version_command(args: list[str]) -> None:
    report = ReportMode.AUTO
    quiet = False
    while args:
        token = args.pop(0)
        if token == "--report":
            if not args:
                _version_error(report, quiet, "Missing value for --report.", {"option": token})
            raw_report = args.pop(0)
            try:
                report = ReportMode(raw_report)
            except ValueError as exc:
                _version_error(
                    report,
                    quiet,
                    "--report must be one of 'auto', 'human', 'single', or 'stream'.",
                    {"report": raw_report},
                    exc,
                )
        elif token == "--quiet":
            quiet = True
        else:
            _version_error(
                report,
                quiet,
                "Unsupported --version option.",
                {"option": token},
            )
    reporter = Reporter(report, quiet=quiet)
    payload = {
        "ok": True,
        "pixelup": __version__,
        "realesrgan": _metadata_version("realesrgan"),
        "torch": _metadata_version("torch"),
        "basicsr_fork": "basicsr-fixed",
        "basicsr_fork_version": _metadata_version("basicsr-fixed"),
    }
    if reporter.is_human:
        reporter.info(
            "pixelup {pixelup} (realesrgan {realesrgan}, torch {torch}, "
            "basicsr-fixed {basicsr_fork_version})".format(**payload)
        )
    else:
        reporter.success(payload)


def _version_error(
    report: ReportMode,
    quiet: bool,
    message: str,
    details: dict[str, object],
    cause: Exception | None = None,
) -> None:
    error = PixelupError(
        ErrorCode.INVALID_ARGUMENT,
        message,
        details=details,
    )
    reporter = Reporter(report, quiet=quiet)
    reporter.error(error)
    raise typer.Exit(exit_code_for(error.code)) from cause or error


def _metadata_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


if __name__ == "__main__":
    main()
