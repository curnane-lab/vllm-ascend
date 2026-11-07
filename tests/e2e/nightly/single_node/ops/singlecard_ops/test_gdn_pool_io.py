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


def test_run_indexed_pool_prefill_matches_legacy_with_independent_slots():
    pytest.importorskip("sgl_kernel_npu")
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    from vllm_ascend.ops import gdn_pool_io

    gdn_pool_io._get_pool_prefill_op.cache_clear()
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(11)
    lengths = [65, 7]
    total = sum(lengths)
    heads = 4
    head_dim = 128
    q = torch.randn(1, total, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = 0.1 * torch.randn_like(q)
    g = -torch.rand(1, total, heads, device=device, dtype=torch.float32)
    beta = torch.rand(1, total, heads, device=device, dtype=torch.bfloat16)
    pool = 0.05 * torch.randn(32, heads, head_dim, head_dim, device=device, dtype=torch.bfloat16)
    src = torch.tensor([5, 5], dtype=torch.int32, device=device)
    dst = torch.tensor([17, 29], dtype=torch.int32, device=device)
    has_initial = torch.tensor([True, False], device=device)
    cu = torch.tensor([0, 65, 72], dtype=torch.int32, device=device)

    reference = pool.clone()
    initial = reference[src.long()].contiguous()
    initial[~has_initial] = 0
    ref_out, final = torch.ops.npu.chunk_gated_delta_rule(
        l2norm_fwd(q.contiguous()).squeeze(0).contiguous(),
        l2norm_fwd(k.contiguous()).squeeze(0).contiguous(),
        v.squeeze(0).contiguous(),
        beta=beta.squeeze(0),
        initial_state=initial,
        actual_seq_lengths=torch.diff(cu),
        scale=head_dim**-0.5,
        g=g.squeeze(0),
    )
    reference[dst.long()] = final

    actual = pool.clone()
    output = gdn_pool_io.run_indexed_pool_prefill(
        q,
        k,
        v,
        g,
        beta,
        actual,
        src,
        has_initial,
        head_dim**-0.5,
        dst_slots=dst,
        cu_seqlens=cu,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output, ref_out.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual[5], pool[5], rtol=0, atol=0)
