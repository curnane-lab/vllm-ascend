"""Guard-decision tests for hybrid-pool block-size overrides.

Pins the empirically measured danger map (Qwen3.5-4B, bf16, Atlas 910B3;
see the linked root-cause analysis) onto `_classify_hybrid_block_size`:
measured silent-corruption at {512, 640, 768} (concurrency >= 2) and at
user-set 2048 (single request suffices), empirically safe {128, 256}
(pad-shifted, concurrency cliff), structurally safe {1024} (natural).
"""

import pytest

from vllm_ascend.patch.platform.patch_mamba_config import (
    _classify_hybrid_block_size,
)

# Qwen3.5-4B hybrid-pool byte constants (bf16):
# K page per token 2048 B; ssm state page per block 2 MiB (natural
# alignment => 1024 tokens); conv page ~48 KiB.
K_PAGE_PER_TOKEN = 2048
SSM_PAGE = 2 * 1024 * 1024
CONV_PAGE = 48 * 1024


def classify(block_size):
    return _classify_hybrid_block_size(
        block_size, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)[0]


@pytest.mark.parametrize("block_size", [512, 640, 768, 2048, 4096])
def test_dangerous_block_sizes_are_rejected(block_size):
    assert classify(block_size) == "unsafe"


@pytest.mark.parametrize("block_size", [128, 256])
def test_shifted_block_sizes_are_flagged_not_structural(block_size):
    assert classify(block_size) == "shifted"


def test_natural_alignment_is_perfect_interlock():
    level, k_page, pad = _classify_hybrid_block_size(
        1024, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    assert level == "perfect"
    assert k_page == SSM_PAGE
    assert pad == CONV_PAGE


def test_pad_grows_monotonically_as_block_shrinks_below_alignment():
    _, _, pad_256 = _classify_hybrid_block_size(
        256, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    _, _, pad_128 = _classify_hybrid_block_size(
        128, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    assert pad_128 > pad_256 > CONV_PAGE
