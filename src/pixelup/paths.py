from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message


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
            Message("error.outputTemplate"),
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
            Message("error.outputFormatUnknown"),
            hint=Message("error.hintChooseFormat"),
            details={"output": output_arg},
        ) from exc


@dataclass(slots=True)
class NamePlanCache:
    """What one planning pass has read: each output folder's entry names and each
    reserved path's collision key. A batch of jobs scans a folder once instead of
    once per candidate name; publication still claims every name no-clobber."""

    names: dict[Path, frozenset[str]] = field(default_factory=dict)
    keys: dict[Path, str] = field(default_factory=dict)

    def key(self, path: Path) -> str:
        found = self.keys.get(path)
        if found is None:
            found = self.keys[path] = _collision_key(path)
        return found

    def exists_case_insensitively(self, path: Path) -> bool:
        # The directory listing catches a case-only sibling on a case-sensitive
        # volume and a broken symlink, which exists() would miss.
        parent = path.parent
        names = self.names.get(parent)
        if names is None:
            try:
                names = frozenset(entry.name.casefold() for entry in parent.iterdir())
            except (FileNotFoundError, NotADirectoryError):
                names = frozenset()
            self.names[parent] = names
        return path.name.casefold() in names


def default_output_path(
    input_path: Path,
    *,
    model: str,
    scale: int,
    output_format: OutputFormat,
    output_dir: Path | None = None,
    reserved: set[Path] | None = None,
    cache: NamePlanCache | None = None,
) -> Path:
    directory = absolute_user_path(output_dir or input_path.parent)
    stem = f"{input_path.stem}-{model_filename_token(model)}-{scale}x"
    suffix = "." + ("jpg" if output_format == OutputFormat.JPG else output_format.value)
    return collision_safe_path(
        directory / f"{stem}{suffix}",
        reserved=reserved,
        companion_suffixes=(".json",),
        cache=cache,
    )


def collision_safe_path(
    path: Path,
    *,
    reserved: set[Path] | None = None,
    companion_suffixes: tuple[str, ...] = (),
    cache: NamePlanCache | None = None,
) -> Path:
    # Keys are casefolded so a candidate collides with any file or reservation
    # that differs only in case — on macOS/Windows the two would be one file.
    cache = cache if cache is not None else NamePlanCache()
    used = {cache.key(item) for item in (reserved or set())}
    candidate = absolute_user_path(path)
    if _bundle_is_free(candidate, companion_suffixes, used, cache):
        return candidate
    for index in range(2, 10000):
        numbered = candidate.with_name(f"{candidate.stem}-{index}{candidate.suffix}")
        if _bundle_is_free(numbered, companion_suffixes, used, cache):
            return numbered
    raise PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        Message("error.noUnusedName"),
        details={"output": str(path)},
    )


def _collision_key(path: Path) -> str:
    return str(path.expanduser().resolve()).casefold()


def _bundle_is_free(
    candidate: Path,
    companion_suffixes: tuple[str, ...],
    used: set[str],
    cache: NamePlanCache,
) -> bool:
    paths = [candidate, *(candidate.with_suffix(suffix) for suffix in companion_suffixes)]
    return all(
        not cache.exists_case_insensitively(path) and _collision_key(path) not in used
        for path in paths
    )


def model_filename_token(model: str) -> str:
    token = model.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", token).strip("-")


def _is_directory_output(raw: str, path: Path) -> bool:
    return raw.endswith(("/", "\\")) or path.is_dir()
