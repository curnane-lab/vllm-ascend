# mypy: ignore-errors

import math
import os

import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


def _get_sparse_index_kpool(model_config) -> int | None:
    """Return the active sparse index-kpool ratio, if configured."""
    for config_name in ("hf_text_config", "hf_config"):
        config = getattr(model_config, config_name, None)
        if config is None or getattr(config, "index_topk", None) is None:
            continue
        if not hasattr(config, "index_kpool"):
            continue
        index_kpool = config.index_kpool
        if not isinstance(index_kpool, int) or index_kpool <= 1:
            raise ValueError("Sparse index-kpool models require index_kpool to be an integer greater than 1.")
        return index_kpool
    return None


def _using_sparse_sfa_c8(vllm_config, model_config) -> bool:
    """Mirror AscendConfig.enable_sparse_sfa_c8 gating at config time.

    This runs from ModelConfig.__post_init__, which precedes
    init_ascend_config(), so the derivation is repeated from vllm_config
    primitives instead of reading the not-yet-existing AscendConfig.
    """
    if vllm_config.cache_config.cache_dtype not in ("fp8", "int8"):
        return False
    from vllm_ascend.utils import model_uses_kpool_indexer, model_uses_sfa_sparse

    return model_uses_sfa_sparse(model_config) or model_uses_kpool_indexer(model_config)


def _using_kv_store(vllm_config) -> bool:
    """
    Check whether AscendStoreConnector is used.
    In the scenario where only PD separation is used, mamba_cache_mode is not automatically set to align.
    """
    if not vllm_config.kv_transfer_config:
        return False
    if vllm_config.kv_transfer_config.kv_connector == "AscendStoreConnector":
        return True
    if vllm_config.kv_transfer_config.kv_connector == "MultiConnector":
        kv_connector_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if not kv_connector_extra_config:
            return False
        if connectors := kv_connector_extra_config.get("connectors"):
            return any(connector.get("kv_connector") == "AscendStoreConnector" for connector in connectors)
    return False


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Update Hybrid Attention/Mamba cache configuration without forcing
    attention and Mamba cache page sizes to be equal.

    Args:
        vllm_config: vLLM configuration.
    """
    using_kv_store_with_hybrid = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and _using_kv_store(
        vllm_config
    )
    logger.debug("Using kv store: %s", using_kv_store_with_hybrid)
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    index_kpool = _get_sparse_index_kpool(model_config)
    if index_kpool is not None:
        parallel_config = vllm_config.parallel_config
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        kernel_block_size = 128
        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture,
            model_config=model_config,
        )
        mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
        mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
        if int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")):
            # M5 compact pool: under LeapQuant the GDN temporal state is
            # the composite head-block layout (HEAD_BYTES per head, 1
            # MB/slot), not dense fp32 [H, V, K] (2 MB). The attention
            # module's get_state_shape returns the composite shape for
            # the actual spec; keep the page math here honest so the
            # block size / padding derive from the smaller page.
            from vllm_ascend.ops.gdn_leapquant import HEAD_BYTES

            mamba_shapes = tuple(
                (sh[0], HEAD_BYTES // 4)
                if len(sh) == 3 and sh[1] == sh[2]
                else sh
                for sh in mamba_shapes
            )
        mamba_raw_page_size = sum(
            math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(mamba_shapes, mamba_dtypes)
        )

        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        if model_config.use_mla:
            kv_lora_rank = model_config.hf_text_config.kv_lora_rank
            qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
            attn_token_page_size = (
                (kv_lora_rank + qk_rope_head_dim) * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
            )
        else:
            attn_head_size = model_config.get_head_size()
            attn_token_page_size = 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

        sfa_c8_packed = model_config.use_mla and _using_sparse_sfa_c8(vllm_config, model_config)
        if sfa_c8_packed:
            # A C8-packed SFA page holds one token in kv_lora_rank int8 bytes
            # plus fp32 scale metadata, roughly half the bf16 page. Size the
            # shared mamba page from the packed page, otherwise it stays at
            # bf16 geometry and the C8 capacity gain is lost to padding.
            from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim

            attn_token_page_size = get_sfa_qsfa_packed_head_dim(
                model_config.hf_text_config.kv_lora_rank,
                model_config.hf_text_config.qk_rope_head_dim,
            )

        # The compressed indexer storage block is consumed by a CANN kernel
        # whose block size must be a multiple of 16. Keep the scheduler block
        # C128-aligned while making block_size / index_kpool C16-aligned too.
        alignment_tokens = math.lcm(kernel_block_size, index_kpool * 16)
        min_block_size = cdiv(mamba_raw_page_size, attn_token_page_size)
        requested_block_size = cache_config.block_size or kernel_block_size
        attn_block_size = alignment_tokens * cdiv(max(requested_block_size, min_block_size), alignment_tokens)
        if sfa_c8_packed:
            # The packed C8 page must cover the mamba state page, otherwise
            # the page-strided KV view gains a per-block hole and the CANN
            # operator's AutoContiguous copies the whole KV pool once per
            # layer per decode step. The covering size is what the alignment
            # formula derives from min_block_size alone (544 -> 640 for
            # GLM-5.3-Flash); cap larger requests at it, since they only
            # coarsen scheduling granularity without adding coverage.
            c8_block_size = alignment_tokens * cdiv(min_block_size, alignment_tokens)
            attn_block_size = min(attn_block_size, c8_block_size)
        if cache_config.block_size != attn_block_size:
            cache_config.block_size = attn_block_size
            logger.info(
                "Setting attention block size to %d tokens to align MLA, "
                "recurrent-state, and compressed indexer cache pages.",
                attn_block_size,
            )

        attn_page_size = cache_config.block_size * attn_token_page_size
        target_mamba_page_size = max(attn_page_size, mamba_raw_page_size)
        if cache_config.mamba_page_size_padded != target_mamba_page_size:
            cache_config.mamba_page_size_padded = target_mamba_page_size
            padding_bytes = target_mamba_page_size - mamba_raw_page_size
            mamba_padding_pct = 100 * padding_bytes / target_mamba_page_size
            logger.info(
                "Padding mamba page size by %.2f%% to align the sparse indexer and recurrent-state cache pages.",
                mamba_padding_pct,
            )
    # The extract_hidden_states connector (ExampleHiddenStatesConnector) only
    # manages the dedicated hidden-state cache-only layer; it does not migrate
    # mamba KV blocks across instances, so it does not require the block-aligned
    # mamba cache mode. Forcing "align" for it would route hybrid models onto
    # vLLM's fused GPU postprocess Triton kernel (introduced in vLLM #40172),
    # which the Ascend Triton backend cannot compile. Leave the mode as vLLM
    # derived it (e.g. "none" when prefix caching is off) for this case.
    spec_config = vllm_config.speculative_config
    is_extract_hidden_states = (
        spec_config is not None and getattr(spec_config, "method", None) == "extract_hidden_states"
    )
    if using_kv_store_with_hybrid and not is_extract_hidden_states:
        if cache_config.mamba_cache_mode == "none":
            cache_config.mamba_cache_mode = "align"
        else:
            assert cache_config.mamba_cache_mode == "align", (
                "mamba_cache_mode only support 'align' when kv_transfer enabled now!"
            )
    if cache_config.enable_prefix_caching and cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size
    else:
        cache_config.mamba_block_size = model_config.max_model_len


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config

def _patch_m5_mamba_state_shape():
    """M5: under VLLM_ASCEND_GDN_LEAPQUANT the Qwen3.5 GDN temporal state is
    the composite head-block layout (HEAD_BYTES per head, 1 MB/slot), not
    dense fp32 [H, V, K] (2 MB). The hybrid page math (upstream
    PlatformInterface.verify_and_update_config reads
    model_cls.get_mamba_state_shape_from_config) must see the compact size
    or the block size / padding stay derived from the dense 2 MB page.
    The attention module's own get_state_shape returns the same compact
    shape for the actual spec, so both agree."""
    try:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration as _Q35,
        )
    except ImportError:
        return
    orig = _Q35.get_mamba_state_shape_from_config.__func__

    @classmethod
    def wrapped(cls, vllm_config):
        shapes = orig(cls, vllm_config)
        if int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")):
            from vllm_ascend.ops.gdn_leapquant import HEAD_BYTES

            shapes = tuple(
                (sh[0], HEAD_BYTES // 4)
                if len(sh) == 3 and sh[1] == sh[2]
                else sh
                for sh in shapes
            )
        return shapes

    _Q35.get_mamba_state_shape_from_config = wrapped


_patch_m5_mamba_state_shape()

def _patch_m5_page_padding():
    """M5: keep the mamba page at the compact raw size. Upstream
    PlatformInterface.verify_and_update_config pads
    cache_config.mamba_page_size_padded up to the attention page; the
    mamba spec reads that field (mamba/abstract.py:get_kv_cache_spec),
    so the padding inflates every mamba layer's page. The KV-cache
    groups carry per-page byte sizes and the align-mode state copies are
    per-group, so the equality is not required - under LeapQuant the
    compact page saves 0.5 MB x 24 GDN layers per block."""
    from vllm.platforms.interface import Platform as _Platform

    orig = _Platform._align_hybrid_block_size.__func__

    @classmethod
    def wrapped(cls, vllm_config, backend_cls=None):
        orig(cls, vllm_config, backend_cls)
        cache_config = vllm_config.cache_config
        if not int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")):
            return
        if getattr(cache_config, "mamba_cache_mode", None) != "align":
            return
        if cache_config.mamba_page_size_padded is None:
            return
        model_cls, _ = ModelRegistry.resolve_model_cls(
            vllm_config.model_config.architecture,
            model_config=vllm_config.model_config,
        )
        shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
        dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
        raw = sum(
            math.prod(sh) * get_dtype_size(dt) for sh, dt in zip(shapes, dtypes)
        )
        if raw < cache_config.mamba_page_size_padded:
            logger.info(
                "M5: keeping the compact mamba page (%d bytes) instead of "
                "padding to the attention page (%d bytes)",
                raw,
                cache_config.mamba_page_size_padded,
            )
            cache_config.mamba_page_size_padded = raw

    _Platform._align_hybrid_block_size = wrapped


_patch_m5_page_padding()
