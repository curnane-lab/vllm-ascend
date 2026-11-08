#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Indexed state-pool I/O for GDN prefill.

The indexed operator reads initial states from source slots and writes final
states directly to independent destination slots. Single-sequence batches can
also use the experimental AscendC binding or the compact gather/scatter adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend import envs
from vllm_ascend.ops.gdn_pool_layout import (
    GDNPoolPrefillMetadata,
    prepare_gdn_pool_prefill_metadata,
    prepare_pool_qkv,
)
from vllm_ascend.ops.gdn_pool_prefill_ascendc import _capturing, gdn_pool_prefill_ascendc
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

_POOL_ALIGNMENT_BYTES = 64
_FUSED_CHUNK_SIZE = 64


@lru_cache(maxsize=1)
def _get_pool_prefill_op():
    try:
        import sgl_kernel_npu  # noqa: F401
    except (ImportError, OSError):
        return None
    return getattr(torch.ops.npu, "chunk_gated_delta_rule_pool", None)


@lru_cache(maxsize=1)
def _get_phase6_pool_prefill_op():
    try:
        from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd_pool
    except (ImportError, AttributeError, OSError):
        return None
    return npu_chunk_gated_delta_rule_fwd_pool


def _get_ctypes_fused_fwd():
    """Return the ctypes fused GDN prefill wrapper that accepts BTH g/beta."""
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_gated_delta_rule_fwd

    return npu_chunk_gated_delta_rule_fwd


def single_sequence_pool_batch(
    query: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    """Whether this prefill is one sequence the compact adapter can represent."""
    return (
        query.ndim == 4
        and query.shape[0] == 1
        and cu_seqlens.ndim == 1
        and cu_seqlens.shape[0] == 2
        and state_indices.ndim == 1
        and state_indices.shape[0] == 1
    )


@torch.no_grad()
def indexed_pool_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, None]:
    """Run one fused prefill with independent source and destination slots.

    Args:
        q: [B, Hk, T, Dk] BNSD layout.
        k: [B, Hk, T, Dk] BNSD layout.
        v: [B, Hv, T, Dv] BNSD layout.
        g: [B, T, Hv] BTH layout (decay gate).
        beta: [B, T, Hv] BTH layout.
        ssm_state: [Pool, Hv, Dv, Dk] state pool.
        src_slots: [B] int32 source pool slot indices.
        dst_slots: [B] int32 destination pool slot indices.
        has_initial_state: [B] bool mask; False rows read zero.
        scale: attention scale factor.

    Returns:
        (output, None) where output is [B, T, Hv, Dv] BSND.
    """
    native_out = gdn_pool_prefill_ascendc(
        src_pool=ssm_state,
        dst_pool=ssm_state,
        src_slots=src_slots,
        dst_slots=dst_slots,
        has_initial_state=has_initial_state,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
    )
    if native_out is not None:
        return native_out, None

    fused_fwd = _get_ctypes_fused_fwd()
    initial_state = gather_ssm_states(ssm_state, src_slots, has_initial_state)
    initial_state = initial_state.transpose(-1, -2).contiguous()

    output, final_state, *_ = fused_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=_FUSED_CHUNK_SIZE,
        scale=scale,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=False,
    )

    written_state = final_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
    scatter_ssm_states_(ssm_state, dst_slots, written_state)
    return output, None


def supports_pool_layout(ssm_state: torch.Tensor, device: torch.device) -> bool:
    if ssm_state.dtype != torch.bfloat16 or ssm_state.ndim != 4:
        return False
    if ssm_state.device != device:
        return False
    if ssm_state.data_ptr() % _POOL_ALIGNMENT_BYTES != 0:
        return False
    if (ssm_state.stride(0) * ssm_state.element_size()) % _POOL_ALIGNMENT_BYTES != 0:
        return False
    return (
        ssm_state.stride(-3) == ssm_state.size(-2) * ssm_state.size(-1)
        and ssm_state.stride(-2) == ssm_state.size(-1)
        and ssm_state.stride(-1) == 1
    )


def supports_pool_io(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    device: torch.device,
) -> bool:
    """Check if the indexed pool I/O path is eligible for this batch."""
    return (
        supports_pool_layout(ssm_state, device)
        and state_indices.dtype in (torch.int32, torch.int64)
        and has_initial_state.dtype == torch.bool
    )


def map_prefill_source_indices(
    state_indices: torch.Tensor,
    copies: Sequence[KVCacheBlockCopy],
) -> torch.Tensor:
    source_indices = state_indices
    for copy in copies:
        source_indices = torch.where(
            state_indices == copy.dst_block_id,
            copy.src_block_id,
            source_indices,
        )
    return source_indices.to(dtype=torch.int32).contiguous()


def prepare_gdn_pool_prefill_cow(
    attn_groups: list,
    static_forward_context: dict,
    copies: Sequence[KVCacheBlockCopy] | None,
) -> None:
    """Keep every cache copy and use prepared running slots for native pool I/O."""
    for groups in attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(0)
            set_copies = getattr(builder, "set_gdn_pool_prefill_cow", None)
            if set_copies is not None:
                set_copies(())


@torch.no_grad()
def run_indexed_pool_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
    *,
    dst_slots: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    pool_metadata: GDNPoolPrefillMetadata | None = None,
    chunk_metadata: object | None = None,
) -> torch.Tensor | None:
    """Run packed BSND prefill through indexed I/O or a legacy adapter."""
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    pool_op = (
        _get_pool_prefill_op() if chunk_metadata is None or getattr(chunk_metadata, "keep_meta", None) is None else None
    )
    phase6_pool_op = (
        _get_phase6_pool_prefill_op()
        if chunk_metadata is not None and getattr(chunk_metadata, "keep_meta", None) is None
        else None
    )
    use_phase6_pool = phase6_pool_op is not None and cu_seqlens is not None
    use_layout_fusion = (
        envs.VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION == 1
        and (pool_op is not None or use_phase6_pool)
        and cu_seqlens is not None
        and not _capturing()
    )
    if use_layout_fusion:
        if pool_metadata is None:
            pool_metadata = prepare_gdn_pool_prefill_metadata(
                state_indices,
                dst_slots,
                has_initial_state,
                cu_seqlens,
                device=ssm_state.device,
            )
        src_slots = pool_metadata.src_slots
        dst_slots = pool_metadata.dst_slots
        has_initial_state = pool_metadata.has_initial_state
    else:
        src_slots = state_indices.to(device=ssm_state.device, dtype=torch.int32).contiguous()
        if dst_slots is None:
            dst_slots = src_slots
        else:
            dst_slots = dst_slots.to(device=ssm_state.device, dtype=torch.int32).contiguous()
        has_initial_state = has_initial_state.to(device=ssm_state.device, dtype=torch.bool).contiguous()

    if (pool_op is not None or use_phase6_pool) and cu_seqlens is not None:
        if use_layout_fusion:
            query, key, value = prepare_pool_qkv(q, k, v, src_slots.shape[0])
        else:
            query = l2norm_fwd(q.contiguous()).squeeze(0).contiguous()
            key = l2norm_fwd(k.contiguous()).squeeze(0).contiguous()
            value = v.squeeze(0).contiguous()
        if use_phase6_pool:
            return phase6_pool_op(
                query.unsqueeze(0).transpose(1, 2),
                key.unsqueeze(0).transpose(1, 2),
                value.unsqueeze(0).transpose(1, 2),
                g.squeeze(0).to(torch.float32).contiguous().unsqueeze(0),
                beta.squeeze(0).to(value.dtype).contiguous().unsqueeze(0),
                ssm_state,
                src_slots,
                dst_slots,
                has_initial_state,
                cu_seqlens=chunk_metadata.cu_seqlens_host,
                chunk_indices=chunk_metadata.chunk_indices_chunk64_host,
                scale=scale,
                chunk_size=_FUSED_CHUNK_SIZE,
            )
        actual_seq_lengths = (
            pool_metadata.actual_seq_lengths
            if use_layout_fusion
            else torch.diff(cu_seqlens).to(torch.int32).contiguous()
        )
        output = pool_op(
            query,
            key,
            value,
            ssm_state,
            src_slots,
            dst_slots,
            beta=beta.squeeze(0).to(value.dtype).contiguous(),
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            g=g.squeeze(0).to(torch.float32).contiguous(),
            has_initial_state=has_initial_state,
        )
        return output.unsqueeze(0)

    if cu_seqlens is None or not single_sequence_pool_batch(q, cu_seqlens, src_slots):
        return None

    q_bnsd = l2norm_fwd(q.contiguous()).transpose(1, 2).contiguous()
    k_bnsd = l2norm_fwd(k.contiguous()).transpose(1, 2).contiguous()
    v_bnsd = v.transpose(1, 2).contiguous()
    try:
        output, _ = indexed_pool_prefill(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            g.to(torch.float32).contiguous(),
            beta.to(v.dtype).contiguous(),
            ssm_state,
            src_slots,
            dst_slots,
            has_initial_state,
            scale,
        )
    except ImportError:
        return None
    return output
