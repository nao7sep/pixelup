from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INPUT_NOT_FOUND = "input_not_found"
    INPUT_UNREADABLE = "input_unreadable"
    INPUT_INVALID_FORMAT = "input_invalid_format"
    OUTPUT_EXISTS = "output_exists"
    OUTPUT_UNWRITABLE = "output_unwritable"
    OUTPUT_DIR_MISSING = "output_dir_missing"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    MODEL_CORRUPT = "model_corrupt"
    FACE_ENHANCE_UNAVAILABLE = "face_enhance_unavailable"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_ARGUMENT = "invalid_argument"
    INTERNAL_ERROR = "internal_error"
    JOB_CANCELLED = "job_cancelled"


@dataclass(slots=True)
class PixelupError(Exception):
    """A structured failure with explicitly trusted user-facing fields."""

    code: ErrorCode
    user_message: str
    user_hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.user_message


def user_text(error: PixelupError, *, internal_fallback: str) -> str:
    """Map a structured failure to copy safe for the operation's UI owner."""
    if error.code == ErrorCode.INTERNAL_ERROR:
        return internal_fallback
    return f"{error.user_message} {error.user_hint}" if error.user_hint else error.user_message
