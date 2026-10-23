from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image
from PySide6.QtGui import QImageReader

from pixelup.gui import ImagePreview
from pixelup.i18n import localizer


def test_preview_coalesces_selections_until_active_decode_physically_returns(
    qapp, tmp_path: Path, monkeypatch, process_until,
) -> None:
    paths = [tmp_path / f"image-{number}.png" for number in range(5)]
    for path in paths:
        Image.new("RGB", (1000, 800), "red").save(path)
    preview = ImagePreview()
    preview.resize(320, 240)
    entered, release = threading.Event(), threading.Event()
    reads: list[str] = []
    original = QImageReader.read

    def held_read(reader):
        reads.append(reader.fileName())
        if len(reads) == 1:
            entered.set()
            assert release.wait(10), "preview test did not release held decode"
        return original(reader)

    monkeypatch.setattr(QImageReader, "read", held_read)
    try:
        preview.set_image(paths[0])
        process_until(entered.is_set, timeout_s=5, what="first preview read")
        for path in paths[1:]:
            preview.set_image(path)
        assert reads == [str(paths[0])]
        assert preview._pixmap is None
        assert preview.text() == localizer.t("preview.loading")
        release.set()
        process_until(
            lambda: not preview.loading and preview.reads.settled,
            timeout_s=10, what="latest preview",
        )
        assert reads == [str(paths[0]), str(paths[-1])]
        assert preview._pixmap is not None
        assert max(preview._pixmap.width(), preview._pixmap.height()) <= round(
            320 * preview.devicePixelRatioF()
        )
    finally:
        release.set()
        preview.reads.closing = True
        process_until(lambda: preview.reads.settled, timeout_s=10, what="preview settlement")
        preview.deleteLater()
        qapp.processEvents()
