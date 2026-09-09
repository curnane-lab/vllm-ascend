# mypy: ignore-errors
import math
import os

import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


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


def _classify_hybrid_block_size(
    block_size: int,
    attn_single_token_k_page_size: int,
    ssm_block_page_size: int,
    conv_block_page_size: int,
) -> tuple[str, int, int]:
    """Classify a hybrid-pool block size by byte-level layout safety.

    The unified pool overlays FA [pad|K|V] and mamba [conv|ssm] views on
    the same physical tensors, so cross-group aliasing is impossible only
    when a block id maps to identical byte ranges in both views.

    Returns (level, k_page, pad) where level is one of:
      "perfect": K page == ssm page and pad == conv (per-id interlock);
      "shifted": pad > conv (FA region shifted past the mamba block ids a
                 realistic load can reach — empirical, has a concurrency
                 cliff, not structural);
      "unsafe":  pad <= conv and K page != ssm page (cross-group byte
                 collision; GDN in-place ssm writes overwrite other
                 requests' FA K cache — silent NaN corruption).
    """
    k_page = attn_single_token_k_page_size * block_size
    attn_page = 2 * k_page
    page = max(attn_page, ssm_block_page_size) + conv_block_page_size
    pad = page - attn_page
    if k_page == ssm_block_page_size and pad == conv_block_page_size:
        return "perfect", k_page, pad
    if pad > conv_block_page_size:
        return "shifted", k_page, pad
    return "unsafe", k_page, pad


def _hybrid_pool_layout(
    block_size: int,
    attn_single_token_k_page_size: int,
    attn_token_page_size: int,
    ssm_block_page_size: int,
    conv_block_page_size: int,
    use_mla: bool,
) -> tuple[str, int]:
    """Decide the physical layout of the shared hybrid attn+mamba pool.

    The pool overlays the full-attention (FA) K/V views and the mamba
    conv/ssm views on the same physical tensors. Returns
    (layout, mamba_page_size_padded) where layout is one of:

      "legacy":   tensor-level [pad|K|V] spans over [conv|ssm] spans. Safe
                  iff per-block K page == ssm page (natural alignment, e.g.
                  1024 tokens on Qwen3.5-4B): block id i then maps to the
                  same bytes in both groups (per-id interlock).
      "strided":  FA K/V become 4-D as_strided views whose kernel-block
                  stride is ssm_page/chunk, so block i's K and V pages live
                  inside mamba ssm slot i. Same-id containment holds for any
                  block-id assignment (structural safety). Requires
                  2*k_page <= ssm page; the padded page is unchanged.
      "disjoint": the padded page is enlarged to attn_page + ssm + conv so
                  the [conv|ssm] spans and the FA [pad|K|V] spans occupy
                  disjoint byte ranges; every view stays contiguous. Costs
                  ssm_block_page_size extra pool bytes per block.

    "legacy" is byte-identical to the pre-fix behavior and is kept for the
    natural alignment so the default path does not change at all.
    """
    k_page = attn_single_token_k_page_size * block_size
    attn_page = attn_token_page_size * block_size
    padded_page = max(attn_page, ssm_block_page_size) + conv_block_page_size
    if k_page == ssm_block_page_size:
        return "legacy", padded_page
    if not use_mla and 2 * k_page <= ssm_block_page_size:
        return "strided", padded_page
    return "disjoint", attn_page + ssm_block_page_size + conv_block_page_size


def _strided_layout_infeasibility(
    block_size: int,
    attn_single_token_k_page_size: int,
    ssm_block_page_size: int,
    kernel_block_size: int = 128,
) -> str | None:
    """Feasibility check for the "strided" in-page FA layout.

    Returns None if the layout is expressible, otherwise a reason string.
    The ssm page is split into ``block_size // kernel_block_size`` sub-slots
    that each host one K and one V kernel block; sub-slot starts must stay
    512B-aligned (the envelope the FIA/scatter kernels were validated with).
    """
    if block_size % kernel_block_size != 0:
        return (f"block size {block_size} is not a multiple of the attention "
                f"kernel block size {kernel_block_size}")
    chunk = block_size // kernel_block_size
    if ssm_block_page_size % chunk != 0 or (ssm_block_page_size // chunk) % 512 != 0:
        return (f"ssm page ({ssm_block_page_size} B) cannot be split into "
                f"{chunk} 512B-aligned kernel-block sub-slots")
    sub_slot = ssm_block_page_size // chunk
    kv_kernel_block_bytes = 2 * kernel_block_size * attn_single_token_k_page_size
    if kv_kernel_block_bytes > sub_slot:
        return (f"K+V kernel blocks ({kv_kernel_block_bytes} B) exceed the "
                f"per-sub-slot budget ({sub_slot} B)")
    return None


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
    using_kv_store_with_hybrid = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and _using_kv_store(
        vllm_config
    )
    logger.debug("Using kv store: %s", using_kv_store_with_hybrid)
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
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

    # get mamba block size
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    mamba_sizes = []
    for shape, dtype in zip(mamba_shapes, mamba_dtypes):
        mamba_sizes.append(math.prod(shape) * get_dtype_size(dtype))
    ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)

    # Pure linear attention models (e.g. bailing 2.5) have only SSM state,
    # no conv block. Detected by a single 3-D mamba shape (ssm only, no conv).
    # Example shape: MambaSpec(shapes=((8, 128, 128),), mamba_type='linear_attention')
    if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
        conv_block_page_size = 0

    # NOTE(zxr): because of the limit of Ascend Hardware, we need to keep
    # all cache tensors contiguous, so we align the page size of ssm_block
    # and single attn_block
    if model_config.use_mla:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        kv_lora_rank = model_config.hf_text_config.kv_lora_rank
        qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
        attn_single_token_k_page_size = kv_lora_rank * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_rope_token_page_size = qk_rope_head_dim * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = attn_single_token_k_page_size + attn_rope_token_page_size
    else:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    attn_block_size = kernel_block_size * cdiv(ssm_block_page_size, kernel_block_size * attn_single_token_k_page_size)
    # B-2 experiment: allow forcing a smaller hybrid block size via env var.
    # The default path is byte-identical to upstream behavior; the override
    # deliberately breaks the "K page == ssm page" equality to shrink the
    # EAGLE/MTP drop-one-block recompute tax.
    _force_bs = int(os.environ.get("VLLM_ASCEND_HYBRID_BLOCK_SIZE", "0"))
    if _force_bs > 0:
        # PR-1 structural fix is in effect: _reshape_kv_cache_tensors now
        # derives the FA view layout from _hybrid_pool_layout (in-page strided
        # split or disjoint padded spans), so the cross-group byte collisions
        # behind the D8 silent corruption (see results/bs512_rootcause.md) can
        # no longer occur for any override value. _classify_hybrid_block_size
        # stays as the static geometry documentation of the *legacy* overlay;
        # here we only reject sizes the strided layout cannot express.
        _layout, _ = _hybrid_pool_layout(
            _force_bs, attn_single_token_k_page_size, attn_token_page_size,
            ssm_block_page_size, conv_block_page_size, model_config.use_mla)
        if _layout == "strided":
            _infeasible = _strided_layout_infeasibility(
                _force_bs, attn_single_token_k_page_size, ssm_block_page_size)
            if _infeasible is not None:
                raise ValueError(
                    f"VLLM_ASCEND_HYBRID_BLOCK_SIZE={_force_bs} cannot be "
                    f"expressed with the in-page strided hybrid layout: "
                    f"{_infeasible}. Use the natural aligned block size "
                    f"({attn_block_size}).")
        logger.warning(
            "VLLM_ASCEND_HYBRID_BLOCK_SIZE=%d: overriding hybrid page alignment "
            "(natural attn_block_size=%d, ssm_block_page_size=%d B). Hybrid pool "
            "uses the '%s' page layout (structural fix; legacy geometry class: "
            "'%s'). Experimental.",
            _force_bs,
            attn_block_size,
            ssm_block_page_size,
            _layout,
            _classify_hybrid_block_size(
                _force_bs, attn_single_token_k_page_size, ssm_block_page_size,
                conv_block_page_size)[0],
        )
        attn_block_size = _force_bs
    else:
        assert attn_single_token_k_page_size * attn_block_size == ssm_block_page_size, (
            "Cannot align ssm_page_size and attn_page_size."
        )

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_token_page_size

    # pad mamba page size for conv_blocks. The ssm state must always fit in
    # the padded page, hence the max() with ssm_block_page_size (only binding
    # when the B-2 env override shrinks attn_page_size below the ssm page;
    # the default path is unchanged because attn_page_size >= ssm page there).
    # For geometries without the natural per-id interlock and without enough
    # ssm-page room for the in-page strided split, the padded page is enlarged
    # so the FA [pad|K|V] spans sit fully above the mamba [conv|ssm] spans
    # ("disjoint" layout) — this is what makes e.g. a user-set --block-size
    # 2048 safe (stock previously corrupted it silently, see
    # results/d9_bs2048_verdict.md).
    _layout, _mamba_page_target = _hybrid_pool_layout(
        cache_config.block_size, attn_single_token_k_page_size,
        attn_token_page_size, ssm_block_page_size, conv_block_page_size,
        model_config.use_mla)
    if _layout == "disjoint":
        logger.warning(
            "Hybrid pool block size %d cannot interlock attention and mamba "
            "pages in place; enlarging the padded page to %d B so attention "
            "and mamba spans are disjoint (fewer pool blocks as a result).",
            cache_config.block_size, _mamba_page_target)
    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != _mamba_page_target
    ):
        cache_config.mamba_page_size_padded = _mamba_page_target
        mamba_padding_pct = 100 * conv_block_page_size / cache_config.mamba_page_size_padded
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.",
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
