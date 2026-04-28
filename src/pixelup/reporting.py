from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Any, TextIO

from rich.console import Console
from rich.table import Table

from pixelup.errors import PixelupError, error_payload


class ReportMode(StrEnum):
    AUTO = "auto"
    HUMAN = "human"
    SINGLE = "single"
    STREAM = "stream"


class Reporter:
    def __init__(
        self,
        mode: ReportMode,
        *,
        quiet: bool = False,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.mode = resolve_report_mode(mode, stdout=stdout)
        self.quiet = quiet
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.console = Console(file=self.stderr, stderr=True, highlight=False)

    @property
    def is_human(self) -> bool:
        return self.mode == ReportMode.HUMAN

    def info(self, message: str) -> None:
        if self.mode == ReportMode.HUMAN and not self.quiet:
            self.console.print(message)

    def table(self, table: Table) -> None:
        if self.mode == ReportMode.HUMAN and not self.quiet:
            self.console.print(table)

    def success(self, payload: dict[str, Any]) -> None:
        if self.mode == ReportMode.HUMAN:
            if not self.quiet:
                self.console.print(payload.get("message", "OK"))
            return
        self._json_line(payload)

    def error(self, error: PixelupError) -> None:
        payload = error_payload(error)
        if self.mode == ReportMode.STREAM:
            payload = {"event": "result", **payload}
        if self.mode == ReportMode.HUMAN:
            self.console.print(f"Error: {error.message}")
            self.console.print(f"Code: {error.code.value}")
            if error.hint:
                self.console.print(f"Hint: {error.hint}")
            return
        self._json_line(payload)

    def result(self, payload: dict[str, Any]) -> None:
        if self.mode == ReportMode.STREAM:
            payload = {"event": "result", **payload}
        self.success(payload)

    def _json_line(self, payload: dict[str, Any]) -> None:
        self.stdout.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        self.stdout.write("\n")
        self.stdout.flush()


def resolve_report_mode(mode: ReportMode, *, stdout: TextIO | None = None) -> ReportMode:
    if mode != ReportMode.AUTO:
        return mode
    stream = stdout or sys.stdout
    return ReportMode.HUMAN if stream.isatty() else ReportMode.SINGLE

