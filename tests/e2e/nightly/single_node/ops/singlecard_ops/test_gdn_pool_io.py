# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.gdn_pool_io import indexed_pool_prefill, supports_pool_io

_HK = 16
_HV = 32
_HEAD_DIM = 128
_SLOTS = 256
_CHUNK_SIZE = 64

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires NPU",
)


def _make_pool(device, dtype=torch.bfloat16):
    return torch.randn(_SLOTS, _HV, _HEAD_DIM, _HEAD_DIM, device=device, dtype=dtype)


def _make_inputs(batch: int, tokens: int, device: str):
    q = torch.randn(batch, _HK, tokens, _HEAD_DIM, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(batch, _HK, tokens, _HEAD_DIM, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(batch, _HV, tokens, _HEAD_DIM, device=device, dtype=torch.bfloat16).contiguous()
    beta = torch.rand(batch, tokens, _HV, device=device, dtype=torch.bfloat16)
    g = -torch.rand(batch, tokens, _HV, device=device, dtype=torch.float32)
    return q, k, v, g, beta


def test_pool_io_supports_bf16_aligned():
    device = "npu:0"
    pool = _make_pool(device)
    indices = torch.zeros(4, device=device, dtype=torch.int32)
    has_init = torch.ones(4, device=device, dtype=torch.bool)
    assert supports_pool_io(pool, indices, has_init, torch.device(device))


def test_pool_io_rejects_fp32():
    device = "npu:0"
    pool = _make_pool(device, dtype=torch.float32)
    indices = torch.zeros(4, device=device, dtype=torch.int32)
    has_init = torch.ones(4, device=device, dtype=torch.bool)
    assert not supports_pool_io(pool, indices, has_init, torch.device(device))


def test_pool_io_output_matches_compact(monkeypatch):
    # This checks the compact gather/scatter adapter against the same fused
    # kernel. The AscendC op, when installed, is a different implementation.
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gdn_pool_prefill_ascendc", lambda **kwargs: None)
    device = "npu:0"
    batch, tokens = 2, 64
    pool = _make_pool(device)
    src = torch.tensor([3, 7], device=device, dtype=torch.int32)
    has_init = torch.ones(batch, device=device, dtype=torch.bool)
    q, k, v, g, beta = _make_inputs(batch, tokens, device)
    scale = _HEAD_DIM**-0.5

    initial_compact = pool[src.long()].transpose(-1, -2).contiguous()
    from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_gated_delta_rule_fwd

    ref_out, ref_state, *_ = npu_chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_compact,
        output_final_state=True,
        chunk_size=_CHUNK_SIZE,
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
    torch.npu.synchronize()

    pool_copy = pool.clone()
    out, _ = indexed_pool_prefill(q, k, v, g, beta, pool_copy, src, src.clone(), has_init, scale=scale)
    torch.npu.synchronize()

    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        pool_copy[src.long()],
        ref_state.transpose(-1, -2).to(pool_copy.dtype),
        rtol=1e-2,
        atol=1e-2,
    )


def test_pool_io_fresh_row_zero():
    device = "npu:0"
    batch, tokens = 2, 64
    pool = _make_pool(device)
    src = torch.tensor([3, 7], device=device, dtype=torch.int32)
    has_init = torch.tensor([True, False], device=device, dtype=torch.bool)
    q, k, v, g, beta = _make_inputs(batch, tokens, device)

    pool_copy = pool.clone()
    out, _ = indexed_pool_prefill(
        q,
        k,
        v,
        g,
        beta,
        pool_copy,
        src,
        src.clone(),
        has_init,
        scale=_HEAD_DIM**-0.5,
    )
    torch.npu.synchronize()
    assert torch.isfinite(out.float()).all()
