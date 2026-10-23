from pathlib import Path

import pytest

from pixelup.output_reservation import BUNDLE_SUFFIXES
from pixelup.paths import (
    NamePlanCache,
    OutputFormat,
    collision_safe_path,
    default_output_path,
    model_filename_token,
)


def test_default_output_path_uses_model_and_scale_filename(tmp_path: Path) -> None:
    resolved = default_output_path(
        tmp_path / "girl.png",
        model="RealESRGAN_x4plus_anime_6B",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "girl-realesrgan-x4plus-anime-6b-4x.png"


def test_default_output_path_adds_collision_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    first = tmp_path / "girl-realesr-general-x4v3-4x.png"
    first.write_bytes(b"existing")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_preserves_literal_symlinked_source_directory(
    tmp_path: Path,
    make_directory_alias,
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "chosen-alias"
    make_directory_alias(alias, physical)

    resolved = default_output_path(
        alias / "girl.png",
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == alias / "girl-realesr-general-x4v3-4x.png"


def test_reserved_paths_are_treated_as_collisions(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    cache = NamePlanCache()
    cache.reserve_output_bundle(tmp_path / "girl-realesr-general-x4v3-4x.png")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        cache=cache,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_avoids_sidecar_json_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    # An existing sidecar JSON at the default stem must push the new bundle to "-2".
    (tmp_path / "girl-realesr-general-x4v3-4x.json").write_bytes(b"{}")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_reserved_sidecar_as_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    cache = NamePlanCache()
    cache.reserve_output_bundle(tmp_path / "girl-realesr-general-x4v3-4x.json")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        cache=cache,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_case_only_existing_file_as_collision(tmp_path: Path) -> None:
    # A sibling that differs only in case is one file on macOS/Windows, so the
    # planner must disambiguate even on a case-sensitive filesystem.
    input_path = tmp_path / "Girl.png"
    (tmp_path / "girl-realesr-general-x4v3-4x.png").write_bytes(b"existing")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "Girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_case_only_reserved_name_as_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "Girl.png"
    cache = NamePlanCache()
    cache.reserve_output_bundle(tmp_path / "girl-realesr-general-x4v3-4x.png")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        cache=cache,
    )

    assert resolved == tmp_path / "Girl-realesr-general-x4v3-4x-2.png"


def test_model_filename_token_is_lowercase_with_hyphens() -> None:
    assert model_filename_token("RealESRGAN_x4plus_anime_6B") == "realesrgan-x4plus-anime-6b"


@pytest.mark.parametrize("suffix", BUNDLE_SUFFIXES)
def test_reservation_covers_every_normalized_bundle_member(tmp_path: Path, suffix: str) -> None:
    cache = NamePlanCache()
    cache.reserve_output_bundle(tmp_path / "CAF\u00c9.png")
    candidate = tmp_path / f"cafe\u0301{suffix}"

    resolved = collision_safe_path(candidate, cache=cache)

    assert resolved == candidate.with_name(f"{candidate.stem}-2{candidate.suffix}")


def test_reserved_directory_alias_keeps_the_literal_output_location(
    tmp_path: Path, make_directory_alias
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "chosen-alias"
    make_directory_alias(alias, physical)
    cache = NamePlanCache()
    cache.reserve_output_bundle(physical / "Girl-realesr-general-x4v3-4x.png")

    resolved = default_output_path(
        alias / "girl.png",
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        cache=cache,
    )

    assert resolved == alias / "girl-realesr-general-x4v3-4x-2.png"


@pytest.mark.parametrize("character", ["a", "猫", "\U0001f63a"])
def test_long_default_name_preserves_model_scale_and_fits_real_bundle(
    tmp_path: Path, character: str
) -> None:
    import os

    result = default_output_path(
        tmp_path / f"{character * 230}.png",
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )
    assert result.stem.endswith("-realesr-general-x4v3-4x")
    assert result.stem.startswith(character)
    for suffix in BUNDLE_SUFFIXES:
        member = result.with_suffix(suffix)
        if os.name != "nt":
            assert len(member.name.encode("utf-8")) + len("-9999") <= os.pathconf(
                tmp_path, "PC_NAME_MAX"
            )
        member.write_bytes(b"bundle member")


def test_long_candidate_reserves_companion_and_all_collision_digits(tmp_path: Path) -> None:
    source = tmp_path / f"{'猫' * 200}.png"
    companion = ".metadata.json"
    first = collision_safe_path(source, companion_suffixes=(companion,))
    first.with_suffix(companion).write_bytes(b"existing sidecar")
    cache = NamePlanCache()
    for index in range(2, 9999):
        cache.reserved.add(cache.key(first.with_name(f"{first.stem}-{index}.png")))

    result = collision_safe_path(source, companion_suffixes=(companion,), cache=cache)

    assert result.name == f"{first.stem}-9999.png"
    result.write_bytes(b"image")
    result.with_suffix(companion).write_bytes(b"sidecar")


def test_windows_budget_counts_utf16_units_without_splitting_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from types import SimpleNamespace

    from pixelup import paths

    monkeypatch.setattr(paths, "os", SimpleNamespace(name="nt", path=os.path))
    source = tmp_path / f"{'\U0001f63a' * 230}.png"
    result = collision_safe_path(source, companion_suffixes=(".metadata.json",))

    budget = 255 - len(".metadata.json") - len("-9999")
    assert result.stem == "\U0001f63a" * (budget // 2)
    assert len(result.with_suffix(".metadata.json").name.encode("utf-16-le")) // 2 + 5 <= 255


def test_missing_directory_uses_nearest_existing_ancestor_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from pixelup import paths

    if os.name == "nt":
        pytest.skip("POSIX pathconf limit discovery")
    actual = os.pathconf
    queried = []

    def pathconf(directory, name):
        queried.append(directory)
        return actual(directory, name)

    monkeypatch.setattr(paths.os, "pathconf", pathconf)
    parent = tmp_path / "missing" / "nested"
    result = collision_safe_path(parent / f"{'a' * 400}.png")
    assert queried == [parent, parent.parent, tmp_path]
    assert result.parent == parent
    assert len(result.name.encode()) + 5 <= actual(tmp_path, "PC_NAME_MAX")


def test_pathconf_access_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from pixelup import paths

    if os.name == "nt":
        pytest.skip("POSIX pathconf limit discovery")

    def denied(*args):
        raise PermissionError("volume inaccessible")

    monkeypatch.setattr(paths.os, "pathconf", denied)
    with pytest.raises(PermissionError):
        collision_safe_path(tmp_path / "image.png")


def test_reported_volume_limit_controls_the_entire_output_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from types import SimpleNamespace

    from pixelup import paths

    monkeypatch.setattr(
        paths, "os", SimpleNamespace(name="posix", path=os.path, pathconf=lambda *_: 80)
    )
    result = default_output_path(
        tmp_path / f"{'猫' * 100}.png",
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )
    assert result.stem.endswith("-realesr-general-x4v3-4x")
    assert len(result.with_suffix(".json").name.encode("utf-8")) + len("-9999") <= 80
    assert len(result.with_suffix(".json").name.encode("utf-8")) + 5 > 77


def test_indeterminate_volume_limit_does_not_invent_a_filename_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from types import SimpleNamespace

    from pixelup import paths

    monkeypatch.setattr(
        paths, "os", SimpleNamespace(name="posix", path=os.path, pathconf=lambda *_: -1)
    )
    proposed = tmp_path / f"{'a' * 400}.png"
    assert collision_safe_path(proposed) == proposed
