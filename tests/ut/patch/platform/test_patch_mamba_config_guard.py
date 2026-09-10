"""Guard-decision tests for hybrid-pool block-size overrides.

Two layers are pinned here (Qwen3.5-4B, bf16, Atlas 910B3 byte constants):

* `_classify_hybrid_block_size` — static documentation of the *legacy*
  tensor-level overlay geometry: the empirically measured danger map
  (silent corruption at {512, 640, 768} with concurrency >= 2 and at
  user-set 2048 with a single request; pad-shifted empirical safety at
  {128, 256}; perfect interlock at 1024).

* `_hybrid_pool_layout` / `_strided_layout_infeasibility` — the post-fix
  policy. The runner now derives the FA views from the layout plan, so the
  legacy "unsafe"/"shifted" classes no longer gate admission: 512 uses the
  in-page strided split (FA K/V tile the ssm slot exactly; the only
  zero-capacity-cost strided geometry — smaller sizes pay the measured
  ~2.9x FIA strided slow path for no offsetting benefit), 128/256/640/768/
  2048 use disjoint spans behind an enlarged padded page (contiguous views,
  fast FIA), and only sizes the strided split cannot express (e.g. 384)
  are rejected.
"""

import pytest

from vllm_ascend.patch.platform.patch_mamba_config import (
    _classify_hybrid_block_size,
    _hybrid_pool_layout,
    _strided_layout_infeasibility,
)

# Qwen3.5-4B hybrid-pool byte constants (bf16):
# K page per token 2048 B; ssm state page per block 2 MiB (natural
# alignment => 1024 tokens); conv page ~48 KiB.
K_PAGE_PER_TOKEN = 2048
KV_PAGE_PER_TOKEN = 2 * K_PAGE_PER_TOKEN
SSM_PAGE = 2 * 1024 * 1024
CONV_PAGE = 48 * 1024


def classify(block_size):
    return _classify_hybrid_block_size(block_size, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)[0]


def plan(block_size, use_mla=False):
    return _hybrid_pool_layout(block_size, K_PAGE_PER_TOKEN, KV_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE, use_mla)


@pytest.mark.parametrize("block_size", [512, 640, 768, 2048, 4096])
def test_legacy_geometry_danger_map(block_size):
    assert classify(block_size) == "unsafe"


@pytest.mark.parametrize("block_size", [128, 256])
def test_legacy_geometry_shifted_map(block_size):
    assert classify(block_size) == "shifted"


def test_natural_alignment_is_perfect_interlock():
    level, k_page, pad = _classify_hybrid_block_size(1024, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    assert level == "perfect"
    assert k_page == SSM_PAGE
    assert pad == CONV_PAGE


def test_pad_grows_monotonically_as_block_shrinks_below_alignment():
    _, _, pad_256 = _classify_hybrid_block_size(256, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    _, _, pad_128 = _classify_hybrid_block_size(128, K_PAGE_PER_TOKEN, SSM_PAGE, CONV_PAGE)
    assert pad_128 > pad_256 > CONV_PAGE


def test_layout_plan_natural_alignment_keeps_legacy_page():
    layout, page = plan(1024)
    assert layout == "legacy"
    # identical to the pre-fix formula: max(attn_page, ssm_page) + conv
    assert page == 1024 * KV_PAGE_PER_TOKEN + CONV_PAGE


def test_layout_plan_512_uses_strided_in_page_split():
    # 512 is the only zero-capacity-cost strided geometry (2*k == ssm page):
    # K+V kernel blocks tile the ssm slot exactly. Smaller sizes would pay
    # the measured ~2.9x FIA strided slow path for nothing, so they go
    # disjoint instead.
    layout, page = plan(512)
    assert layout == "strided"
    assert page == SSM_PAGE + CONV_PAGE


@pytest.mark.parametrize("block_size", [128, 256, 640, 768, 2048, 4096])
def test_layout_plan_non_perfect_blocks_use_disjoint_spans(block_size):
    layout, page = plan(block_size)
    assert layout == "disjoint"
    # enlarged page: attention K+V spans sit above the [conv|ssm] spans
    assert page == block_size * KV_PAGE_PER_TOKEN + SSM_PAGE + CONV_PAGE


def test_layout_plan_mla_never_strided():
    # MLA hybrid pools keep contiguous views; non-perfect geometries fall
    # back to disjoint spans (the strided split is only validated for the
    # non-MLA branch).
    layout, page = plan(512, use_mla=True)
    assert layout == "disjoint"
    assert page == 512 * KV_PAGE_PER_TOKEN + SSM_PAGE + CONV_PAGE
    layout, page = plan(1024, use_mla=True)
    assert layout == "legacy"


@pytest.mark.parametrize("block_size", [128, 256, 512])
def test_strided_layout_feasible_for_small_power_of_two_blocks(block_size):
    assert _strided_layout_infeasibility(block_size, K_PAGE_PER_TOKEN, SSM_PAGE) is None


def test_strided_layout_rejects_non_kernel_multiple():
    reason = _strided_layout_infeasibility(100, K_PAGE_PER_TOKEN, SSM_PAGE)
    assert reason is not None and "multiple" in reason


def test_strided_layout_rejects_uneven_ssm_split():
    # 384 = 3 * 128 kernel blocks; the 2 MiB ssm page is not divisible by 3.
    reason = _strided_layout_infeasibility(384, K_PAGE_PER_TOKEN, SSM_PAGE)
    assert reason is not None and "sub-slots" in reason


def test_strided_layout_rejects_unaligned_sub_slot():
    # ssm page splits evenly into 4 chunks but the sub-slot is not
    # 512B-aligned (2097664 / 4 = 524416 = 512 * 1024 + 128).
    reason = _strided_layout_infeasibility(512, K_PAGE_PER_TOKEN, 2097664)
    assert reason is not None and "sub-slots" in reason


def test_strided_layout_rejects_oversized_kv_pair():
    # hypothetical 1 MiB ssm page: bs512 sub-slots (256 KiB) cannot host
    # one K plus one V kernel block (512 KiB).
    reason = _strided_layout_infeasibility(512, K_PAGE_PER_TOKEN, 1048576)
    assert reason is not None and "budget" in reason
