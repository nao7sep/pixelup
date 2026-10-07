"""Quit owns only PixelUp's active output staging and exclusive publication claims."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pixelup.errors import ErrorCode, PixelupError


def open_output_handle(path: Path, flags: int, mode: int = 0o666) -> int:
    """Keep owned output handles deletable while encode/copy is still running.

    Windows' CRT opener excludes FILE_SHARE_DELETE. CreateFile's sharing option
    lets quit mark a partial file for deletion; process exit closes held handles.
    """
    if os.name != "nt":
        return os.open(path, flags, mode)
    return _open_windows_output_handle(path, flags)


def _open_windows_output_handle(path: Path, flags: int) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    absolute = str(path.absolute())
    native_path = (
        absolute
        if absolute.startswith("\\\\?\\")
        else "\\\\?\\UNC\\" + absolute[2:]
        if absolute.startswith("\\\\")
        else "\\\\?\\" + absolute
    )
    access = 0x40000000 if flags & os.O_WRONLY else 0x80000000
    disposition = 1 if flags & os.O_CREAT else 3  # CREATE_NEW / OPEN_EXISTING
    handle = create_file(native_path, access, 0x7, None, disposition, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_NOINHERIT)
    except BaseException as exc:
        if not close_handle(handle):
            exc.add_note(
                f"Output handle close failed: {ctypes.WinError(ctypes.get_last_error())!r}"
            )
        if flags & os.O_CREAT:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup:
                exc.add_note(f"Output claim cleanup failed: {cleanup!r}")
        raise


class OutputCleanup(QObject):
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._gate = threading.Lock()
        self._active: dict[int, tuple[Path, bool]] = {}
        self._closing = False
        self._settled = threading.Event()
        self._settled.set()

    def reset(self) -> None:
        with self._gate:
            if self._active or not self._settled.is_set():
                raise RuntimeError("output cleanup still owns work")
            self._closing = False

    def admit(self, descriptor: int, path: Path, *, staging: bool) -> None:
        with self._gate:
            if self._closing:
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Output publication stopped for quit")
            self._active[descriptor] = (path, staging)

    def release(self, descriptor: int) -> None:
        with self._gate:
            self._active.pop(descriptor, None)

    def check_open(self) -> None:
        with self._gate:
            if self._closing:
                raise PixelupError(ErrorCode.JOB_CANCELLED, "Output publication stopped for quit")

    @property
    def settled(self) -> bool:
        return self._settled.is_set()

    def begin_shutdown(self) -> None:
        with self._gate:
            if self._closing:
                return
            self._closing = True
            active = tuple(self._active.items())
            if not active:
                return
            self._settled.clear()
        threading.Thread(
            target=self._clean, args=(active,), name="pixelup-output-cleanup", daemon=True
        ).start()

    def _clean(self, active: tuple[tuple[int, tuple[Path, bool]], ...]) -> None:
        try:
            for descriptor, (path, staging) in active:
                try:
                    if staging:
                        path.unlink(missing_ok=True)
                    else:
                        owned = os.fstat(descriptor)
                        current = path.stat()
                        if (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino):
                            path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    from pixelup.session_log import log

                    log.warning("quit.output_cleanup_failed", path=str(path), reason=str(exc))
        finally:
            self._settled.set()
            self.finished.emit()


output_cleanup = OutputCleanup()
