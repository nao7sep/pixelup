from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pixelup.i18n.message import Message


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
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_ARGUMENT = "invalid_argument"
    INTERNAL_ERROR = "internal_error"
    JOB_CANCELLED = "job_cancelled"


# The two codes whose message never reaches a reader: an internal failure is
# shown through its owner's own fallback, and a cancellation as the plain
# Cancelled status. Their message is English diagnostic text for the log.
_DIAGNOSTIC_CODES = frozenset({ErrorCode.INTERNAL_ERROR, ErrorCode.JOB_CANCELLED})


@dataclass(slots=True)
class PixelupError(Exception):
    """A structured failure with explicitly trusted user-facing fields.

    ``message`` and ``hint`` are what a reader is told, held as catalogue
    messages so the interface renders them in its own language at the boundary
    (localization-conventions). Only an internal or cancelled failure carries a
    plain English string instead, because no reader ever sees it.
    """

    code: ErrorCode
    message: Message | str
    hint: Message | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.message, str) and self.code not in _DIAGNOSTIC_CODES:
            raise TypeError(f"a {self.code.value} failure is shown to the reader: pass a Message")

    def __str__(self) -> str:
        # The log keeps English whatever the interface speaks.
        if isinstance(self.message, str):
            return self.message
        from pixelup.i18n.localizer import english

        text = english().of(self.message)
        return f"{text} {english().of(self.hint)}" if self.hint else text


def user_text(error: PixelupError, *, internal_fallback: Message) -> Message:
    """Map a structured failure to what the operation's UI owner shows."""
    if isinstance(error.message, str):
        return internal_fallback
    if error.hint is None:
        return error.message
    return Message.of("error.withHint", message=error.message, hint=error.hint)
