from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError


class OutputFormat(StrEnum):
    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"


def absolute_user_path(path: Path) -> Path:
    """Make a user-supplied path absolute without resolving symlinks or aliases."""
    return Path(os.path.abspath(path.expanduser()))


@dataclass(frozen=True, slots=True)
class OutputContext:
    input_path: Path
    output_arg: str
    model: str
    scale: int
    output_format: OutputFormat
    input_size: tuple[int, int]

    @property
    def output_size(self) -> tuple[int, int]:
        width, height = self.input_size
        return width * self.scale, height * self.scale


def resolve_output_path(context: OutputContext) -> Path:
    output = Path(context.output_arg).expanduser()
    if _is_directory_output(context.output_arg, output):
        return default_output_path(
            context.input_path,
            model=context.model,
            scale=context.scale,
            output_format=context.output_format,
            output_dir=output,
        )
    if "{" in context.output_arg or "}" in context.output_arg:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Output filename templates are not supported by the GUI app.",
            details={"output": context.output_arg},
        )
    return absolute_user_path(output)


def infer_output_format(output_arg: str, forced: OutputFormat | None) -> OutputFormat:
    if forced is not None:
        return forced
    output = Path(output_arg).expanduser()
    if _is_directory_output(output_arg, output):
        return OutputFormat.PNG
    suffix = output.suffix.lower().lstrip(".")
    if suffix == "jpeg":
        suffix = "jpg"
    try:
        return OutputFormat(suffix)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Output format could not be inferred from the output path.",
            hint="Choose png, jpg, or webp.",
            details={"output": output_arg},
        ) from exc


def default_output_path(
    input_path: Path,
    *,
    model: str,
    scale: int,
    output_format: OutputFormat,
    output_dir: Path | None = None,
    reserved: set[Path] | None = None,
) -> Path:
    directory = absolute_user_path(output_dir or input_path.parent)
    stem = f"{input_path.stem}-{model_filename_token(model)}-{scale}x"
    suffix = "." + ("jpg" if output_format == OutputFormat.JPG else output_format.value)
    return collision_safe_path(
        directory / f"{stem}{suffix}",
        reserved=reserved,
        companion_suffixes=(".json",),
    )


def collision_safe_path(
    path: Path,
    *,
    reserved: set[Path] | None = None,
    companion_suffixes: tuple[str, ...] = (),
) -> Path:
    # Keys are casefolded so a candidate collides with any file or reservation
    # that differs only in case — on macOS/Windows the two would be one file.
    used = {_collision_key(item) for item in (reserved or set())}
    candidate = absolute_user_path(path)
    if _bundle_is_free(candidate, companion_suffixes, used):
        return candidate
    for index in range(2, 10000):
        numbered = candidate.with_name(f"{candidate.stem}-{index}{candidate.suffix}")
        if _bundle_is_free(numbered, companion_suffixes, used):
            return numbered
    raise PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        "Could not find an unused output filename.",
        details={"output": str(path)},
    )


def _collision_key(path: Path) -> str:
    return str(path.expanduser().resolve()).casefold()


def _exists_case_insensitively(path: Path) -> bool:
    # os-level exists() already answers this on case-insensitive filesystems;
    # the directory scan is what catches a case-only sibling on Linux.
    if path.exists():
        return True
    parent = path.parent
    if not parent.is_dir():
        return False
    target = path.name.casefold()
    return any(entry.name.casefold() == target for entry in parent.iterdir())


def _bundle_is_free(
    candidate: Path,
    companion_suffixes: tuple[str, ...],
    used: set[str],
) -> bool:
    paths = [candidate, *(candidate.with_suffix(suffix) for suffix in companion_suffixes)]
    return all(
        not _exists_case_insensitively(path) and _collision_key(path) not in used
        for path in paths
    )


def model_filename_token(model: str) -> str:
    token = model.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", token).strip("-")


def _is_directory_output(raw: str, path: Path) -> bool:
    return raw.endswith(("/", "\\")) or path.is_dir()
