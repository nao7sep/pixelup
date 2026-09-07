from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def _distribution_file(distribution_name: str, suffix: str) -> Path:
    distribution = importlib.metadata.distribution(distribution_name)
    matches = [file for file in distribution.files or () if str(file).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {suffix} in {distribution_name}, found {len(matches)}."
        )
    return Path(distribution.locate_file(matches[0]))


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build-third-party-notices.py OUTPUT")

    repository = Path(__file__).resolve().parents[1]
    sections = [
        ("PixelUp adapted source", repository / "THIRD_PARTY_NOTICES"),
        ("OpenCV", _distribution_file("opencv-python", "cv2/LICENSE.txt")),
        (
            "OpenCV bundled third-party software",
            _distribution_file("opencv-python", "cv2/LICENSE-3RD-PARTY.txt"),
        ),
        (
            "pillow-heif",
            _distribution_file("pillow-heif", "licenses/LICENSE.txt"),
        ),
        (
            "pillow-heif bundled third-party software",
            _distribution_file("pillow-heif", "licenses/LICENSES_bundled.txt"),
        ),
    ]
    rendered: list[str] = []
    for title, source in sections:
        rendered.extend(
            (
                "=" * 78,
                title,
                f"Source package file: {source.name}",
                "=" * 78,
                "",
                source.read_text(encoding="utf-8").rstrip(),
                "",
            )
        )

    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rendered), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
