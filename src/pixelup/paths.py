from __future__ import annotations

import errno
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.output_reservation import BUNDLE_SUFFIXES, normalized_name


class OutputFormat(StrEnum):
    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"


def absolute_user_path(path: Path) -> Path:
    """Make a user-supplied path absolute without resolving symlinks or aliases."""
    return Path(os.path.abspath(path.expanduser()))


@dataclass(slots=True)
class NamePlanCache:
    """One planning pass's folder snapshots and normalized bundle reservations.

    Reservations enter once when a job is accepted, so checking another name does
    not revisit every earlier job. Publication checks every name again just before it.
    """

    names: dict[Path, frozenset[str]] = field(default_factory=dict)
    reserved: set[str] = field(default_factory=set)

    def key(self, path: Path) -> str:
        return _collision_key(path)

    def reserve_output_bundle(self, output_path: Path) -> None:
        self.reserved.add(self.key(output_path))
        for suffix in BUNDLE_SUFFIXES:
            if suffix != output_path.suffix:
                self.reserved.add(self.key(output_path.with_suffix(suffix)))

    def exists_case_insensitively(self, path: Path) -> bool:
        # The directory listing catches a case-only sibling on a case-sensitive
        # volume and a broken symlink, which exists() would miss.
        parent = path.parent
        names = self.names.get(parent)
        if names is None:
            try:
                names = frozenset(normalized_name(entry.name) for entry in parent.iterdir())
            except (FileNotFoundError, NotADirectoryError):
                names = frozenset()
            self.names[parent] = names
        return normalized_name(path.name) in names


def default_output_path(
    input_path: Path,
    *,
    model: str,
    scale: int,
    output_format: OutputFormat,
    cache: NamePlanCache | None = None,
) -> Path:
    directory = absolute_user_path(input_path.parent)
    suffix = "." + ("jpg" if output_format == OutputFormat.JPG else output_format.value)
    stem = _bounded_stem(
        input_path.stem,
        directory,
        (suffix, *BUNDLE_SUFFIXES),
        tail=f"-{model_filename_token(model)}-{scale}x",
    )
    return collision_safe_path(
        directory / f"{stem}{suffix}",
        companion_suffixes=BUNDLE_SUFFIXES,
        cache=cache,
    )


def collision_safe_path(
    path: Path,
    *,
    companion_suffixes: tuple[str, ...] = (),
    cache: NamePlanCache | None = None,
) -> Path:
    # Keys are casefolded so a candidate collides with any file or reservation
    # that differs only in case — on macOS/Windows the two would be one file.
    cache = cache if cache is not None else NamePlanCache()
    candidate = absolute_user_path(path)
    stem = _bounded_stem(candidate.stem, candidate.parent, (candidate.suffix, *companion_suffixes))
    candidate = candidate.with_name(f"{stem}{candidate.suffix}")
    if _bundle_is_free(candidate, companion_suffixes, cache):
        return candidate
    for index in range(2, 10000):
        numbered = candidate.with_name(f"{candidate.stem}-{index}{candidate.suffix}")
        if _bundle_is_free(numbered, companion_suffixes, cache):
            return numbered
    raise PixelupError(
        ErrorCode.OUTPUT_EXISTS,
        Message("error.noUnusedName"),
        details={"output": str(path)},
    )


def _component_limit(directory: Path) -> int:
    if os.name == "nt":
        return 255
    # A destination may not exist yet. Its nearest existing ancestor supplies
    # the volume's advertised component limit without writing a probe file.
    while True:
        try:
            return os.pathconf(directory, "PC_NAME_MAX")
        except FileNotFoundError:
            directory = directory.parent


def _name_units(value: str) -> int:
    if os.name == "nt":
        return len(value.encode("utf-16-le", errors="surrogatepass")) // 2
    return len(value.encode("utf-8", errors="surrogateescape"))


def _truncate_name(value: str, budget: int) -> str:
    units = 0
    for index, character in enumerate(value):
        units += _name_units(character)
        if units > budget:
            return value[:index]
    return value


def _bounded_stem(stem: str, directory: Path, suffixes: tuple[str, ...], *, tail: str = "") -> str:
    limit = _component_limit(directory)
    if limit < 0:
        return stem + tail
    # Reserve numbering now, so collisions keep the same base instead of
    # successively shortening it. Every companion shares that base.
    budget = limit - max(_name_units(suffix) for suffix in suffixes) - _name_units("-9999")
    if budget < 1:
        raise OSError(
            errno.ENAMETOOLONG, "Output suffix exceeds the filename limit", str(directory)
        )
    if _name_units(tail) <= budget:
        return _truncate_name(stem, budget - _name_units(tail)) + tail
    return _truncate_name(stem + tail, budget)


def _collision_key(path: Path) -> str:
    return normalized_name(str(path.expanduser().resolve()))


def _bundle_is_free(
    candidate: Path,
    companion_suffixes: tuple[str, ...],
    cache: NamePlanCache,
) -> bool:
    paths = [candidate, *(candidate.with_suffix(suffix) for suffix in companion_suffixes)]
    return all(
        not cache.exists_case_insensitively(path) and cache.key(path) not in cache.reserved
        for path in paths
    )


def model_filename_token(model: str) -> str:
    token = model.lower().replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", token).strip("-")
