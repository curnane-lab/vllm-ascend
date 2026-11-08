#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from dataclasses import dataclass

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.triton_utils import tl, triton

_FUSED_LAYOUT_MAX_TOKENS = 512
_FUSED_QK_BLOCK_ROWS = 64
_FUSED_LAYOUT_HEAD_DIM = 128


@dataclass(frozen=True)
class GDNPoolPrefillMetadata:
    src_slots: torch.Tensor
    dst_slots: torch.Tensor
    has_initial_state: torch.Tensor
    actual_seq_lengths: torch.Tensor


def prepare_gdn_pool_prefill_metadata(
    state_indices: torch.Tensor,
    dst_slots: torch.Tensor | None,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    device: torch.device | None = None,
) -> GDNPoolPrefillMetadata:
    device = state_indices.device if device is None else device
    src_slots = state_indices.to(device=device, dtype=torch.int32).contiguous()
    if dst_slots is None:
        dst_slots = src_slots
    else:
        dst_slots = dst_slots.to(device=device, dtype=torch.int32).contiguous()
    return GDNPoolPrefillMetadata(
        src_slots=src_slots,
        dst_slots=dst_slots,
        has_initial_state=has_initial_state.to(device=device, dtype=torch.bool).contiguous(),
        actual_seq_lengths=torch.diff(cu_seqlens).to(device=device, dtype=torch.int32).contiguous(),
    )


def supports_fused_layout(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[0] == k.shape[0] == v.shape[0] == 1
        and q.shape[1] == k.shape[1] == v.shape[1]
        and q.shape[2] == k.shape[2]
        and q.shape[2] > 0
        and v.shape[2] % q.shape[2] == 0
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == _FUSED_LAYOUT_HEAD_DIM
        and q.stride(1) == k.stride(1) == v.stride(1)
        and q.stride(2) == k.stride(2) == v.stride(2)
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device
        and q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    )


@triton.jit
def _fused_head_first_qkv_kernel(
    QKV,
    OUT,
    eps,
    M,
    QK_ROWS,
    HK,
    HV,
    D: tl.constexpr,
    token_stride,
    head_stride,
    key_offset,
    value_offset,
    BD: tl.constexpr,
    MBLOCK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * MBLOCK + tl.arange(0, MBLOCK)
    cols = tl.arange(0, BD)[None, :]
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < D)
    is_key = (rows >= QK_ROWS) & (rows < 2 * QK_ROWS)
    is_value = rows >= 2 * QK_ROWS
    local_rows = rows - is_key.to(tl.int64) * QK_ROWS - is_value.to(tl.int64) * 2 * QK_ROWS
    component_heads = HK + is_value.to(tl.int64) * (HV - HK)
    tokens = local_rows // component_heads
    heads = local_rows % component_heads
    source_base = is_key.to(tl.int64) * key_offset + is_value.to(tl.int64) * value_offset
    source_offsets = (
        tokens[:, None] * token_stride + heads[:, None] * head_stride + source_base[:, None] + cols
    )
    destination_base = (is_key.to(tl.int64) + 2 * is_value.to(tl.int64)) * QK_ROWS
    destination_offsets = (local_rows[:, None] + destination_base[:, None]) * D + cols
    x = tl.load(QKV + source_offsets, mask=mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(x * x, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(row_mask[:, None], square, 0), 1)[:, None]
    normalized = x * tl.rsqrt(square_sum + eps)
    result = tl.where(is_value[:, None], x, normalized)
    tl.store(OUT + destination_offsets, result.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _fused_head_first_qk_kernel(
    QK,
    OUT,
    eps,
    M,
    QK_ROWS,
    HK,
    D: tl.constexpr,
    token_stride,
    head_stride,
    key_offset,
    BD: tl.constexpr,
    MBLOCK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * MBLOCK + tl.arange(0, MBLOCK)
    cols = tl.arange(0, BD)[None, :]
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < D)
    is_key = rows >= QK_ROWS
    local_rows = rows - is_key.to(tl.int64) * QK_ROWS
    tokens = local_rows // HK
    heads = local_rows % HK
    source_offsets = (
        tokens[:, None] * token_stride
        + heads[:, None] * head_stride
        + is_key.to(tl.int64)[:, None] * key_offset
        + cols
    )
    destination_offsets = rows[:, None] * D + cols
    x = tl.load(QK + source_offsets, mask=mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(x * x, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(row_mask[:, None], square, 0), 1)[:, None]
    normalized = x * tl.rsqrt(square_sum + eps)
    tl.store(OUT + destination_offsets, normalized.to(OUT.dtype.element_ty), mask=mask)


def fused_head_first_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    tokens, num_k_heads, num_v_heads = q.shape[1], q.shape[2], v.shape[2]
    head_dim = q.shape[-1]
    qk_rows = tokens * num_k_heads
    rows = qk_rows * 2 + tokens * num_v_heads
    packed = torch.empty(rows * head_dim, dtype=q.dtype, device=q.device)
    qk_elements = qk_rows * head_dim
    query = packed[:qk_elements].view(tokens, num_k_heads, head_dim)
    key = packed[qk_elements : 2 * qk_elements].view(tokens, num_k_heads, head_dim)
    value = packed[2 * qk_elements :].view(tokens, num_v_heads, head_dim)
    block = 32
    _fused_head_first_qkv_kernel[(triton.cdiv(rows, block),)](
        q,
        packed,
        1e-6,
        rows,
        qk_rows,
        num_k_heads,
        num_v_heads,
        head_dim,
        q.stride(1),
        q.stride(2),
        (k.data_ptr() - q.data_ptr()) // q.element_size(),
        (v.data_ptr() - q.data_ptr()) // q.element_size(),
        BD=triton.next_power_of_2(head_dim),
        MBLOCK=block,
    )
    return query, key, value


def fused_head_first_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    tokens, num_k_heads, head_dim = q.shape[1], q.shape[2], q.shape[-1]
    qk_rows = tokens * num_k_heads
    packed = torch.empty(2 * qk_rows * head_dim, dtype=q.dtype, device=q.device)
    query = packed[: qk_rows * head_dim].view(tokens, num_k_heads, head_dim)
    key = packed[qk_rows * head_dim :].view(tokens, num_k_heads, head_dim)
    _fused_head_first_qk_kernel[(triton.cdiv(2 * qk_rows, _FUSED_QK_BLOCK_ROWS),)](
        q,
        packed,
        1e-6,
        2 * qk_rows,
        qk_rows,
        num_k_heads,
        head_dim,
        q.stride(1),
        q.stride(2),
        (k.data_ptr() - q.data_ptr()) // q.element_size(),
        BD=triton.next_power_of_2(head_dim),
        MBLOCK=_FUSED_QK_BLOCK_ROWS,
    )
    return query, key, v.squeeze(0).contiguous()


def _separate_l2norm_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    query = l2norm_fwd(q.contiguous()).squeeze(0).contiguous()
    key = l2norm_fwd(k.contiguous()).squeeze(0).contiguous()
    return query, key, v.squeeze(0).contiguous()


def prepare_pool_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, row_count: int):
    tokens = q.shape[1]
    if tokens <= _FUSED_LAYOUT_MAX_TOKENS and row_count <= 2 and supports_fused_layout(q, k, v):
        return fused_head_first_qkv(q, k, v)
    if tokens <= _FUSED_LAYOUT_MAX_TOKENS and row_count in (3, 4) and supports_fused_layout(q, k, k):
        return fused_head_first_qk(q, k, v)
    if tokens <= _FUSED_LAYOUT_MAX_TOKENS:
        query_key = l2norm_fwd(torch.cat((q.contiguous(), k.contiguous()), dim=2))
        query = query_key[:, :, : q.shape[2]].squeeze(0).contiguous()
        key = query_key[:, :, q.shape[2] :].squeeze(0).contiguous()
        return query, key, v.squeeze(0).contiguous()
    return _separate_l2norm_qkv(q, k, v)
