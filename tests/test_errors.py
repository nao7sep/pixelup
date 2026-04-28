from pixelup.errors import ErrorCode, exit_code_for


def test_exit_code_mapping_matches_public_contract() -> None:
    assert exit_code_for(ErrorCode.INPUT_NOT_FOUND) == 3
    assert exit_code_for(ErrorCode.OUTPUT_EXISTS) == 4
    assert exit_code_for(ErrorCode.MODEL_NOT_FOUND) == 5
    assert exit_code_for(ErrorCode.DENOISE_STRENGTH_UNSUPPORTED) == 2
    assert exit_code_for(ErrorCode.OUT_OF_MEMORY) == 6
    assert exit_code_for(ErrorCode.INTERNAL_ERROR) == 1

