from pathlib import Path

import pytest

from pixelup.signals import OperationCancelled, temp_file_guard


def test_temp_file_guard_removes_file_on_cancel(tmp_path: Path) -> None:
    temp_path = tmp_path / "pixelup-temp.png"

    with pytest.raises(OperationCancelled):
        with temp_file_guard(temp_path):
            temp_path.write_bytes(b"partial")
            raise OperationCancelled()

    assert not temp_path.exists()
