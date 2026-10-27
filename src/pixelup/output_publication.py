"""Publish a complete staged member under its checked final name."""

from __future__ import annotations

import errno
import os
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
            # Without hard links the stage takes the name checked just before
            # publication by an ordinary rename.
            os.rename(stage, target)
    except OSError:
        # Quit can remove an owned stage while a filesystem call is blocked.
        # That is cancellation; unrelated publication failures keep their cause.
        output_cleanup.check_open()
        raise
    return PublishedFile(target, identity.st_dev, identity.st_ino)

