from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from pixelup.errors import ErrorCode, PixelupError


class OperationCancelled(PixelupError):
    def __init__(self, signum: int | None = None) -> None:
        details: dict[str, Any] = {}
        if signum is not None:
            details["signal"] = _signal_name(signum)
        super().__init__(
            ErrorCode.CANCELLED,
            "Operation cancelled.",
            details=details,
        )


@dataclass(slots=True)
class _CancellationState:
    previous_handlers: dict[signal.Signals, Any] = field(default_factory=dict)
    temp_paths: set[Path] = field(default_factory=set)
    cancelled: OperationCancelled | None = None


_STATE = _CancellationState()
_HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@contextmanager
def cancellation_context() -> Iterator[None]:
    _STATE.cancelled = None
    _STATE.previous_handlers = {}
    for signum in _HANDLED_SIGNALS:
        _STATE.previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        for signum, previous in _STATE.previous_handlers.items():
            signal.signal(signum, previous)
        _STATE.previous_handlers = {}


@contextmanager
def temp_file_guard(path: Path) -> Iterator[None]:
    register_temp_file(path)
    try:
        check_cancelled()
        yield
        check_cancelled()
    except OperationCancelled:
        _unlink(path)
        raise
    finally:
        unregister_temp_file(path)


def register_temp_file(path: Path) -> None:
    _STATE.temp_paths.add(path)


def unregister_temp_file(path: Path) -> None:
    _STATE.temp_paths.discard(path)


def check_cancelled() -> None:
    if _STATE.cancelled is not None:
        raise _STATE.cancelled


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    error = OperationCancelled(signum)
    _STATE.cancelled = error
    for path in list(_STATE.temp_paths):
        _unlink(path)
    raise error


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)
