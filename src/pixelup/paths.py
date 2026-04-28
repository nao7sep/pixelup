from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.models import model_short_name


class OutputFormat(StrEnum):
    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"


@dataclass(frozen=True, slots=True)
class RunTimestamp:
    date: str
    time: str
    datetime: str

    @classmethod
    def now(cls) -> RunTimestamp:
        value = datetime.now(UTC)
        return cls(
            date=value.strftime("%Y%m%d"),
            time=value.strftime("%H%M%S"),
            datetime=value.strftime("%Y%m%d-%H%M%S-utc"),
        )


@dataclass(frozen=True, slots=True)
class OutputContext:
    input_path: Path
    output_arg: str
    model: str
    scale: int
    output_format: OutputFormat
    input_size: tuple[int, int]
    face_enhance: bool
    denoise_strength: float
    timestamp: RunTimestamp

    @property
    def output_size(self) -> tuple[int, int]:
        width, height = self.input_size
        return width * self.scale, height * self.scale


_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
_SEPARATORS = "_-"


def resolve_output_path(context: OutputContext) -> Path:
    output_arg = context.output_arg
    output = Path(output_arg).expanduser()
    template = output_arg
    if _is_directory_output(output_arg, output):
        template = str(output / default_filename_pattern())
    if "{" in template or "}" in template:
        template = substitute_placeholders(template, context)
    return Path(template).expanduser().resolve()


def infer_output_format(output_arg: str, forced: OutputFormat | None) -> OutputFormat:
    if forced is not None:
        return forced
    output = Path(output_arg).expanduser()
    if _is_directory_output(output_arg, output):
        return OutputFormat.PNG
    suffix = Path(output_arg).suffix.lower().lstrip(".")
    if suffix == "{ext}":
        return OutputFormat.PNG
    if suffix == "jpeg":
        suffix = "jpg"
    try:
        return OutputFormat(suffix)
    except ValueError as exc:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Output format could not be inferred from the output path.",
            hint="Use --format png, --format jpg, or --format webp.",
            details={"output": output_arg},
        ) from exc


def default_filename_pattern() -> str:
    return "{stem}__{model_short}_{scale}x__{width}px.{ext}"


def substitute_placeholders(template: str, context: OutputContext) -> str:
    values = _placeholder_values(context)
    result = template
    search_start = 0
    while match := _PLACEHOLDER_RE.search(result, search_start):
        name = match.group(1)
        if name not in values:
            raise PixelupError(
                ErrorCode.INVALID_ARGUMENT,
                f"Unknown output placeholder '{{{name}}}'.",
                details={"placeholder": name},
            )
        value = values[name]
        if value:
            result = result[: match.start()] + value + result[match.end() :]
            search_start = match.start() + len(value)
            continue
        result, search_start = _remove_empty_placeholder(result, match.start(), match.end())
    if "{" in result or "}" in result:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Output path contains an invalid placeholder.",
            details={"output": template},
        )
    return result


def _placeholder_values(context: OutputContext) -> dict[str, str]:
    width, height = context.output_size
    denoise = ""
    if context.model == "realesr-general-x4v3" and context.denoise_strength != 1.0:
        denoise = f"{context.denoise_strength:g}"
    return {
        "stem": context.input_path.stem,
        "ext": context.output_format.value,
        "model": context.model,
        "model_short": model_short_name(context.model),
        "scale": str(context.scale),
        "width": str(width),
        "height": str(height),
        "denoise": denoise,
        "face": "face" if context.face_enhance else "",
        "date": context.timestamp.date,
        "time": context.timestamp.time,
        "datetime": context.timestamp.datetime,
    }


def _remove_empty_placeholder(text: str, start: int, end: int) -> tuple[str, int]:
    right = end
    while right < len(text) and text[right] in _SEPARATORS:
        right += 1
    if right > end:
        return text[:start] + text[right:], start

    left = start
    while left > 0 and text[left - 1] in _SEPARATORS:
        left -= 1
    return text[:left] + text[end:], left


def _is_directory_output(raw: str, path: Path) -> bool:
    return raw.endswith(("/", "\\")) or path.is_dir()
