"""Publish a complete staged member without replacing another writer's file."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

from pixelup.output_cleanup import output_cleanup
from pixelup.output_reservation import PublishedFile


def publish_complete(stage: Path, target: Path) -> PublishedFile:
    output_cleanup.check_open()
    try:
        identity = stage.stat()
        try:
            os.link(stage, target, follow_symlinks=False)
        except OSError as exc:
            if exc.errno not in {
                errno.EACCES, errno.EINVAL, errno.ENOSYS, errno.ENOTSUP,
                errno.EOPNOTSUPP, errno.EPERM,
            }:
                raise
            rename_no_replace(stage, target)
    except OSError:
        # Quit can remove an owned stage while a filesystem call is blocked.
        # That is cancellation; unrelated publication failures keep their cause.
        output_cleanup.check_open()
        raise
    return PublishedFile(target, identity.st_dev, identity.st_ino)


def rename_no_replace(source: Path, target: Path) -> None:
    """Native no-replace rename also works on destinations without hard links.

    Never substitute an existence check plus overwriting rename, or copy into a
    final name: those respectively lose a competing file or expose partial bytes.
    """
    if os.name == "nt":
        # Windows rename refuses an existing destination, including on exFAT.
        os.rename(source, target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(target), 0x4)  # RENAME_EXCL
    else:
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "No complete no-replace publication available", target)
        rename.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), target)
