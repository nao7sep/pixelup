"""Complete output visibility and no-replace behavior on the host filesystem."""

import errno
from pathlib import Path

import pytest

from pixelup import output_publication
from pixelup.output_reservation import published_file_is_current


def _without_hard_links(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr(output_publication.os, "link", unsupported)


def test_native_fallback_publishes_complete_bytes_and_preserves_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_hard_links(monkeypatch)
    stage = tmp_path / "stage.tmp"
    destination = tmp_path / "complete.json"
    data = b'{"content": "' + b"complete bytes " * 10000 + b'"}\n'
    stage.write_bytes(data)

    published = output_publication.publish_complete(stage, destination)

    assert destination.read_bytes() == data
    assert published_file_is_current(published)
    assert not stage.exists()


def test_native_fallback_never_replaces_an_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_hard_links(monkeypatch)
    stage = tmp_path / "stage.tmp"
    destination = tmp_path / "existing.json"
    stage.write_bytes(b"our complete bytes")
    destination.write_bytes(b"other writer's complete bytes")
    original = destination.stat()

    with pytest.raises(FileExistsError):
        output_publication.publish_complete(stage, destination)

    assert destination.read_bytes() == b"other writer's complete bytes"
    assert destination.stat().st_ino == original.st_ino
    assert stage.read_bytes() == b"our complete bytes"
