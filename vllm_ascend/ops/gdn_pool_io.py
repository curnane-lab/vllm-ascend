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

Prototype adapter. The fused kernel still consumes a compact state block, so
this path gathers selected pool rows, transposes them into the kernel layout,
and scatters the final state back. Direct slot addressing needs a native
operator. Padded pool rows go through the same gather/scatter kernels as the
compact prefill path, which read only the selected slots.
"""

from __future__ import annotations

import torch

from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

_POOL_ALIGNMENT_BYTES = 64
_FUSED_CHUNK_SIZE = 64


def _get_ctypes_fused_fwd():
    """Return the ctypes fused GDN prefill wrapper that accepts BTH g/beta."""
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_gated_delta_rule_fwd

    return npu_chunk_gated_delta_rule_fwd


def single_sequence_pool_batch(
    query: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    """Whether this prefill is one sequence the pool adapter can represent.

    Production prefill packs variable-length sequences into ``[1, T, ...]`` and
    passes ``cu_seqlens`` into the fused kernel. This adapter has no sequence
    offsets, so only a single packed sequence is eligible.
    """
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
    """Run fused GDN prefill reading and writing selected state-pool slots.

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
    fused_fwd = _get_ctypes_fused_fwd()

    # The operator reads a contiguous [B, Hv, Dk, Dv] block. Gather only the
    # selected rows, including padded pool strides, then match state_v_first=False.
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

    # final_state is [B, Hv, Dk, Dv]; the pool stores [Pool, Hv, Dv, Dk].
    written_state = final_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
    scatter_ssm_states_(ssm_state, dst_slots, written_state)
    return output, None


def supports_pool_io(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    device: torch.device,
) -> bool:
    """Check if the indexed pool I/O path is eligible for this batch."""
    if ssm_state.dtype != torch.bfloat16:
        return False
    if state_indices.dtype not in (torch.int32, torch.int64):
        return False
    if has_initial_state.dtype != torch.bool:
        return False
    if ssm_state.device != device:
        return False
    slot_stride_bytes = ssm_state.stride(0) * ssm_state.element_size()
    if slot_stride_bytes % _POOL_ALIGNMENT_BYTES != 0:
        return False
    if ssm_state.data_ptr() % _POOL_ALIGNMENT_BYTES != 0:
        return False
    return True


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
) -> torch.Tensor | None:
    """Adapt one production BSND prefill sequence onto ``indexed_pool_prefill``.

    ``q``/``k``/``v`` arrive as ``[1, T, H, D]``. The ctypes kernel does not
    apply Q/K L2 norm, matching ``use_qk_l2norm_in_kernel=False``, so norm is
    applied here. Returns ``None`` when the ctypes wrapper is unavailable so
    the caller can keep the compact prefill path.
    """
    try:
        _get_ctypes_fused_fwd()
    except ImportError:
        return None

    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    q_bnsd = l2norm_fwd(q.contiguous()).transpose(1, 2).contiguous()
    k_bnsd = l2norm_fwd(k.contiguous()).transpose(1, 2).contiguous()
    v_bnsd = v.transpose(1, 2).contiguous()
    output, _ = indexed_pool_prefill(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        g.to(torch.float32).contiguous(),
        beta.to(v.dtype).contiguous(),
        ssm_state,
        state_indices,
        state_indices,
        has_initial_state,
        scale,
    )
    return output
