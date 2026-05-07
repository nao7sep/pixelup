from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError


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
    return output.resolve()


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
    directory = (output_dir or input_path.parent).expanduser().resolve()
    stem = f"{input_path.stem}-{model_filename_token(model)}-{scale}x"
    suffix = "." + ("jpg" if output_format == OutputFormat.JPG else output_format.value)
    return collision_safe_path(directory / f"{stem}{suffix}", reserved=reserved)


def collision_safe_path(path: Path, *, reserved: set[Path] | None = None) -> Path:
    used = {item.expanduser().resolve() for item in (reserved or set())}
    candidate = path.expanduser().resolve()
    if not candidate.exists() and candidate not in used:
        return candidate
    for index in range(2, 10000):
        numbered = candidate.with_name(f"{candidate.stem}-{index}{candidate.suffix}")
        if not numbered.exists() and numbered not in used:
            return numbered
    raise PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        "Could not find an unused output filename.",
        details={"output": str(path)},
    )


def model_filename_token(model: str) -> str:
    token = model.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", token).strip("-")


def _is_directory_output(raw: str, path: Path) -> bool:
    return raw.endswith(("/", "\\")) or path.is_dir()
