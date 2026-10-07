# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import lcm

import os
import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

    Args:
        vllm_config: vLLM Config
    """
    # Save the user input before it gets modified by MambaModelConfig
    mamba_block_size = vllm_config.cache_config.mamba_block_size
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    # get attention page size (for 1 token)
    if model_config.use_mla:
        raise RuntimeError("MLA is not supported on 310P currently.")
    kernel_block_alignment_size = 128
    attn_page_size_1_token = FullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
    ).page_size_bytes

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba page size
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    if int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")):
        # M5 compact pool: under LeapQuant the GDN temporal state is the
        # composite head-block layout (HEAD_BYTES per head, 1 MB/slot),
        # not dense fp32 [H, V, K] (2 MB). The attention module's
        # get_state_shape returns the composite shape for the actual
        # spec; keep this page math honest (see patch_mamba_config).
        from vllm_ascend.ops.gdn_leapquant import HEAD_BYTES

        mamba_shapes = tuple(
            (sh[0], HEAD_BYTES // 4)
            if len(sh) == 3 and sh[1] == sh[2]
            else sh
            for sh in mamba_shapes
        )
    mamba_page_size = MambaSpec(
        shapes=mamba_shapes,
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=-1,
    ).page_size_bytes

    # Model may be marked as is_hybrid
    #  but mamba is skipped via config,
    #  return directly
    if mamba_page_size == 0:
        return
    if cache_config.mamba_cache_mode == "all":
        base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
        attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
        chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
        attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
        cache_config.mamba_block_size = attn_block_size
    else:
        attn_block_size = kernel_block_alignment_size * cdiv(
            mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token
        )
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )
    if cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size
    attn_page_size = cache_config.block_size * attn_page_size_1_token
    assert attn_page_size >= mamba_page_size
    if attn_page_size == mamba_page_size:
        # don't need to pad mamba page size
        return
    # pad mamba page size to exactly match attention
    if cache_config.mamba_page_size_padded is None or cache_config.mamba_page_size_padded != attn_page_size:
        cache_config.mamba_page_size_padded = attn_page_size
        mamba_padding_pct = 100 * (attn_page_size - mamba_page_size) / mamba_page_size
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.",
            mamba_padding_pct,
        )


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
