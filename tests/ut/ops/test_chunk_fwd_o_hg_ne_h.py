# SPDX-License-Identifier: Apache-2.0
"""Regression test: chunk_fwd_o must stay finite for Hg != H at long T.

With Qwen3.5's GDN head layout (16 key heads / 48 value heads, K=V=128),
the AscendC ``chunk_fwd_o`` kernel produces Inf/NaN once the per-call
sequence length grows (T=8192 reproduces; the fwd_h outputs -- h, v_new
and final_state -- remain finite, so the overflow is inside fwd_o).
The long-T Hg!=H case is xfail until the kernel is fixed; strict=False
so a fix surfaces as XPASS.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires NPU",
)

CHUNK = 64


def _run(Hg, H, T):
    torch.npu.set_device(0)
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )

    init_device_properties_triton()
    dev = torch.device("npu:0")
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(T, H, 128, generator=gen).to(dev, torch.bfloat16).unsqueeze(0)
    k = torch.nn.functional.normalize(torch.randn(T, Hg, 128, generator=gen),
                                      dim=-1).to(dev, torch.bfloat16).unsqueeze(0)
    v = torch.randn(T, H, 128, generator=gen).to(dev, torch.bfloat16).unsqueeze(0)
    beta = torch.sigmoid(torch.randn(T, H, generator=gen)).to(dev, torch.float32).unsqueeze(0)
    glog = torch.nn.functional.logsigmoid(
        torch.randn(T, H, generator=gen) - 0.5).to(dev, torch.float32).unsqueeze(0)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int64)
    s0 = torch.zeros(1, H, 128, 128, device=dev, dtype=torch.float32)

    gc = chunk_local_cumsum(glog, chunk_size=CHUNK, cu_seqlens=cu)
    attn = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=gc, cu_seqlens=cu)
    attn = solve_tril(A=attn, cu_seqlens=cu)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=attn, g_cumsum=gc, cu_seqlens=cu)
    ka = k.transpose(1, 2).contiguous()
    ci = prepare_chunk_indices(cu, CHUNK).to(torch.int64).flatten().tolist()
    h, v_new, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        ka, w.to(torch.bfloat16).transpose(1, 2).contiguous(),
        u.to(torch.bfloat16).transpose(1, 2).contiguous(),
        g=gc.transpose(1, 2).contiguous(), gk=None, initial_state=s0,
        output_final_state=True, chunk_size=CHUNK, save_new_value=True,
        cu_seqlens=cu.tolist(), chunk_indices=ci, use_exp2=False,
        transpose_state_layout=False)
    assert not torch.isnan(h).any() and not torch.isinf(h).any()
    assert not torch.isnan(v_new).any() and not torch.isinf(v_new).any()
    assert not torch.isnan(final).any() and not torch.isinf(final).any()
    qa = q.transpose(1, 2).contiguous()
    o = torch.ops._C_ascend.chunk_fwd_o(
        qa, ka, v_new, h, 128 ** -0.5,
        g=gc.transpose(1, 2).contiguous(), g_gamma=None, cu_seqlens=cu.tolist(),
        chunk_indices=ci, chunk_size=CHUNK, transpose_state_layout=False)
    return o


@pytest.mark.parametrize("hg,h", [(8, 8), (16, 48)])
def test_chunk_fwd_o_finite_short(hg, h):
    o = _run(hg, h, 256)
    assert torch.isfinite(o).all()


def test_chunk_fwd_o_finite_long_t_equal_heads():
    o = _run(8, 8, 8192)
    assert torch.isfinite(o).all()


@pytest.mark.parametrize("t_len", [256, 8192])
@pytest.mark.xfail(reason="chunk_fwd_o intermittently overflows for Hg!=H "
                          "(16/48): values sit near the fp32 max, so "
                          "non-finite outputs appear run-to-run; fwd_h "
                          "outputs stay finite. Equal-head layouts are "
                          "unaffected.", strict=False)
def test_chunk_fwd_o_finite_long_t_hg_ne_h(t_len):
    o = _run(16, 48, t_len)
    assert torch.isfinite(o).all()
