from pixelup.inference import model_architecture_spec
from pixelup.upscale import count_tiles


def test_model_architecture_specs_match_builtin_models() -> None:
    x4 = model_architecture_spec("RealESRGAN_x4plus")
    assert x4.kind == "rrdb"
    assert x4.netscale == 4
    assert x4.params["num_block"] == 23

    anime = model_architecture_spec("RealESRGAN_x4plus_anime_6B")
    assert anime.kind == "rrdb"
    assert anime.params["num_block"] == 6

    general = model_architecture_spec("realesr-general-x4v3")
    assert general.kind == "srvgg"
    assert general.params["num_conv"] == 32


def test_unknown_model_uses_requested_scale_rrdb_default() -> None:
    spec = model_architecture_spec("custom-model", requested_scale=2)

    assert spec.kind == "rrdb"
    assert spec.netscale == 2
    assert spec.params["scale"] == 2


def test_count_tiles_matches_input_grid() -> None:
    assert count_tiles((400, 300), 0) == 1
    assert count_tiles((400, 300), 256) == 4
    assert count_tiles((1, 1), 512) == 1
