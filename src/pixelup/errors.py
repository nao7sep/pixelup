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
    AUTO_DOWNLOAD_DISABLED = "auto_download_disabled"
    DENOISE_STRENGTH_UNSUPPORTED = "denoise_strength_unsupported"
    FACE_ENHANCE_UNAVAILABLE = "face_enhance_unavailable"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_ARGUMENT = "invalid_argument"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"


EXIT_CODE_BY_ERROR: dict[ErrorCode, int] = {
    ErrorCode.INPUT_NOT_FOUND: 3,
    ErrorCode.INPUT_UNREADABLE: 3,
    ErrorCode.INPUT_INVALID_FORMAT: 3,
    ErrorCode.OUTPUT_EXISTS: 4,
    ErrorCode.OUTPUT_UNWRITABLE: 4,
    ErrorCode.OUTPUT_DIR_MISSING: 4,
    ErrorCode.MODEL_NOT_FOUND: 5,
    ErrorCode.MODEL_DOWNLOAD_FAILED: 5,
    ErrorCode.MODEL_CORRUPT: 5,
    ErrorCode.AUTO_DOWNLOAD_DISABLED: 5,
    ErrorCode.FACE_ENHANCE_UNAVAILABLE: 5,
    ErrorCode.DENOISE_STRENGTH_UNSUPPORTED: 2,
    ErrorCode.INVALID_ARGUMENT: 2,
    ErrorCode.OUT_OF_MEMORY: 6,
    ErrorCode.INTERNAL_ERROR: 1,
    ErrorCode.CANCELLED: 7,
}


@dataclass(slots=True)
class PixelupError(Exception):
    code: ErrorCode
    message: str
    hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def exit_code_for(code: ErrorCode) -> int:
    return EXIT_CODE_BY_ERROR[code]


def error_payload(error: PixelupError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "code": error.code.value,
        "message": error.message,
    }
    if error.hint:
        payload["hint"] = error.hint
    if error.details:
        payload["details"] = error.details
    return payload

