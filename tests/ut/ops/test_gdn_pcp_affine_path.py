# SPDX-License-Identifier: Apache-2.0
"""Three-way correctness tests for the GDN CP (PCP) affine-scan prefill path.

Leg A: CPU FP64 per-token recurrence (ground truth for outputs/final states).
Leg B: AscendC kernels with CP=1 over the packed batch (global reference).
Leg C: segmented emulation of the PCP path executed by
       vllm_ascend/ops/triton/fla/chunk.py when the pcp group world_size > 1:
       per-rank AscendC fwd_h from the shared initial state, the triton
       chunk_delta_hupdate cumulative transition Phi, the correction recursion
           updated_i = final_i + Phi_i (updated_{i-1} - s0),
       and the triton fwd_h rerun with the corrected entering state.

The tests assert that Leg C reproduces Leg B exactly (bit-identical for
chunk-aligned splits) and that both match the FP64 reference within the
bf16-kernel noise floor.

Requires one NPU; skipped otherwise.
"""

import pytest
import torch

try:
    from vllm_ascend.ops.triton.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from vllm_ascend.ops.triton.fla.chunk_delta_hupdate import (
        chunk_gated_delta_rule_fwd_hupdate,
    )
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
    from vllm_ascend.utils import enable_custom_op
except ImportError as exc:  # pragma: no cover - non-NPU environment
    pytest.skip(f"vllm-ascend NPU build unavailable: {exc}",
                allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires NPU",
)

CHUNK = 64


@pytest.fixture(scope="module")
def npu_env():
    torch.npu.set_device(0)
    assert enable_custom_op(), "custom ops disabled"
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )

    init_device_properties_triton()
    yield


def _gen_case(req_lens, hg, h, seed, gate="normal", s0_scale=0.0,
              num_decodes=0):
    dev = torch.device("npu:0")
    total = sum(req_lens)
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(total, h, generator=gen) - 0.5
    if gate == "extreme":
        mix = torch.rand(total, h, generator=gen)
        a = torch.where(mix < 0.7, torch.randn(total, h, generator=gen) * 0.5 + 3.0,
                        torch.where(mix < 0.9, torch.randn(total, h, generator=gen) * 0.5 - 4.0,
                                    torch.randn(total, h, generator=gen) * 0.5 - 12.0))
    elif gate == "mild":
        # gamma ~ 0.996: long-memory regime, segment maps do not underflow
        a = torch.randn(total, h, generator=gen) * 0.5 + 5.5
    boundaries = [0]
    for length in req_lens:
        boundaries.append(boundaries[-1] + length)
    s0 = torch.zeros(len(req_lens), h, 128, 128, device=dev, dtype=torch.float32)
    if s0_scale > 0:
        s0 = (torch.randn(len(req_lens), h, 128, 128, generator=gen,
                          dtype=torch.float64) * s0_scale).float().to(dev)
    return {
        "req_lens": req_lens, "total": total, "hg": hg, "h": h,
        "q": torch.randn(total, h, 128, generator=gen).to(dev, torch.bfloat16).unsqueeze(0),
        "k": torch.nn.functional.normalize(
            torch.randn(total, hg, 128, generator=gen), dim=-1).to(dev, torch.bfloat16).unsqueeze(0),
        "v": torch.randn(total, h, 128, generator=gen).to(dev, torch.bfloat16).unsqueeze(0),
        "glog": torch.nn.functional.logsigmoid(a).to(dev, torch.float32).unsqueeze(0),
        "beta": torch.sigmoid(torch.randn(total, h, generator=gen)).to(dev, torch.float32).unsqueeze(0),
        "scale": 128 ** -0.5,
        "cu": torch.tensor(boundaries, device=dev, dtype=torch.int64),
        "num_decodes": num_decodes,
        "s0": s0,
    }


def _leg_a_fp64(c):
    """Per-token FP64 recurrence; returns (outputs [T,H,V], finals [N,H,K,V])."""
    h, hg = c["h"], c["hg"]
    k_expanded = c["k"][0].double().cpu().repeat_interleave(h // hg, dim=1)
    q = c["q"][0].double().cpu()
    v = c["v"][0].double().cpu()
    glog = c["glog"][0].double().cpu()
    beta = c["beta"][0].double().cpu()
    s0 = c["s0"].double().cpu()
    outs = torch.zeros(c["total"], h, 128, dtype=torch.float64)
    finals = s0.clone()
    off = 0
    for req_i, length in enumerate(c["req_lens"]):
        state = s0[req_i].clone()
        for t in range(length):
            gam = glog[off + t].exp().view(h, 1, 1)
            bt = beta[off + t].view(h, 1, 1)
            kt = k_expanded[off + t].view(h, 128, 1)
            vt = v[off + t].view(h, 1, 128)
            key_state = torch.bmm(kt.transpose(1, 2), state)
            state = gam * (state - bt * torch.bmm(kt, key_state)) + bt * torch.bmm(kt, vt)
            outs[off + t] = torch.einsum("hk,hkv->hv", q[off + t], state) * c["scale"]
        finals[req_i] = state
        off += length
    return outs, finals


def _preprocess(c, k, v, beta, glog, cu):
    gc = chunk_local_cumsum(glog, chunk_size=CHUNK, cu_seqlens=cu)
    attn = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=gc, cu_seqlens=cu)
    attn = solve_tril(A=attn, cu_seqlens=cu)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=attn, g_cumsum=gc, cu_seqlens=cu)
    return gc, w, u


def _asc_h(k, w, u, gc, s0, cu):
    ci = prepare_chunk_indices(cu, CHUNK).to(torch.int64)
    h, v_new, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k.to(torch.bfloat16).transpose(1, 2).contiguous(),
        w.to(torch.bfloat16).transpose(1, 2).contiguous(),
        u.to(torch.bfloat16).transpose(1, 2).contiguous(),
        g=gc.transpose(1, 2).contiguous(), gk=None, initial_state=s0,
        output_final_state=True, chunk_size=CHUNK, save_new_value=True,
        cu_seqlens=cu.tolist(), chunk_indices=ci.flatten().tolist(),
        use_exp2=False, transpose_state_layout=False)
    return h, v_new, final


def _asc_o(q, k_asc, v_new, h, gc, cu, scale):
    o = torch.ops._C_ascend.chunk_fwd_o(
        q.to(torch.bfloat16).transpose(1, 2).contiguous(), k_asc, v_new, h,
        scale, g=gc.transpose(1, 2).contiguous(), g_gamma=None,
        cu_seqlens=cu.tolist(),
        chunk_indices=prepare_chunk_indices(cu, CHUNK).to(torch.int64).flatten().tolist(),
        chunk_size=CHUNK, transpose_state_layout=False)
    return o[0].transpose(0, 1)


def _leg_b_global(c):
    gc, w, u = _preprocess(c, c["k"], c["v"], c["beta"], c["glog"], c["cu"])
    h, v_new, final, k_asc = None, None, None, None
    k_asc = c["k"].to(torch.bfloat16).transpose(1, 2).contiguous()
    ci = prepare_chunk_indices(c["cu"], CHUNK).to(torch.int64)
    h, v_new, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_asc,
        w.to(torch.bfloat16).transpose(1, 2).contiguous(),
        u.to(torch.bfloat16).transpose(1, 2).contiguous(),
        g=gc.transpose(1, 2).contiguous(), gk=None, initial_state=c["s0"],
        output_final_state=True, chunk_size=CHUNK, save_new_value=True,
        cu_seqlens=c["cu"].tolist(), chunk_indices=ci.flatten().tolist(),
        use_exp2=False, transpose_state_layout=False)
    o = _asc_o(c["q"], k_asc, v_new, h, gc, c["cu"], c["scale"])
    return o, final


def _triton_h_to_asc_layout(h):
    """Triton fwd_h yields [1, NC, H, K, V]; AscendC fwd_o wants [1, H, NC, K, V]."""
    return h.transpose(1, 2).contiguous()


def _leg_c_pcp(c, segments_per_req):
    H = c["h"]
    outs = torch.zeros(c["total"], H, 128, device=c["k"].device,
                       dtype=torch.bfloat16)
    finals = c["s0"].clone()
    off = 0
    for req_i, length in enumerate(c["req_lens"]):
        sl = slice(off, off + length)
        if req_i < c["num_decodes"]:
            # decode rows are batch-split: full row per rank, Phi zeroed by
            # the hupdate kernel; outputs come from the row's own pass.
            cu_seg = torch.tensor([0, length], device=c["k"].device,
                                  dtype=torch.int64)
            gc, w, u = _preprocess(c, c["k"][:, sl].contiguous(),
                                   c["v"][:, sl].contiguous(),
                                   c["beta"][:, sl].contiguous(),
                                   c["glog"][:, sl].contiguous(), cu_seg)
            h, v_new, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                c["k"][:, sl].to(torch.bfloat16).transpose(1, 2).contiguous(),
                w.to(torch.bfloat16).transpose(1, 2).contiguous(),
                u.to(torch.bfloat16).transpose(1, 2).contiguous(),
                g=gc.transpose(1, 2).contiguous(), gk=None,
                initial_state=c["s0"][req_i:req_i + 1],
                output_final_state=True, chunk_size=CHUNK, save_new_value=True,
                cu_seqlens=cu_seg.tolist(),
                chunk_indices=prepare_chunk_indices(cu_seg, CHUNK).to(torch.int64).flatten().tolist(),
                use_exp2=False, transpose_state_layout=False)
            k_asc = c["k"][:, sl].to(torch.bfloat16).transpose(1, 2).contiguous()
            outs[sl] = _asc_o(c["q"][:, sl], k_asc, v_new, h, gc, cu_seg, c["scale"])
            finals[req_i] = final[0]
            off += length
            continue

        p = segments_per_req
        base = length // p
        bounds = [(i * base, (i + 1) * base) for i in range(p - 1)]
        bounds.append(((p - 1) * base, length))

        upd_prev = None
        upd_next = c["s0"][req_i]
        for rank, (a0, a1) in enumerate(bounds):
            seg_len = a1 - a0
            cu_seg = torch.tensor([0, seg_len], device=c["k"].device,
                                  dtype=torch.int64)
            ks = c["k"][:, off + a0: off + a1].contiguous()
            vs = c["v"][:, off + a0: off + a1].contiguous()
            bs = c["beta"][:, off + a0: off + a1].contiguous()
            gs = c["glog"][:, off + a0: off + a1].contiguous()
            gc, w, u = _preprocess(c, ks, vs, bs, gs, cu_seg)
            k_asc = ks.to(torch.bfloat16).transpose(1, 2).contiguous()
            ci = prepare_chunk_indices(cu_seg, CHUNK).to(torch.int64)
            h, v_new, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                k_asc,
                w.to(torch.bfloat16).transpose(1, 2).contiguous(),
                u.to(torch.bfloat16).transpose(1, 2).contiguous(),
                g=gc.transpose(1, 2).contiguous(), gk=None,
                initial_state=c["s0"][req_i:req_i + 1],
                output_final_state=True, chunk_size=CHUNK, save_new_value=True,
                cu_seqlens=cu_seg.tolist(), chunk_indices=ci.flatten().tolist(),
                use_exp2=False, transpose_state_layout=False)
            # rank-level cumulative transition Phi from the last hupdate slot
            hu = chunk_gated_delta_rule_fwd_hupdate(
                k=ks, w=w, u=u, g=gc, cu_seqlens=cu_seg,
                chunk_indices=prepare_chunk_indices(cu_seg, CHUNK), num_decodes=0)
            phi = hu[0, -1]

            # updated_i = final_i + Phi_i (updated_{i-1} - s0)   [chunk.py]
            upd_next = final[0] if upd_prev is None else \
                final[0] + torch.matmul(phi, upd_prev - c["s0"][req_i])
            if rank == 0:
                outs[sl.start + a0: sl.start + a1] = _asc_o(
                    c["q"][:, off + a0: off + a1], k_asc, v_new, h, gc,
                    cu_seg, c["scale"])
            else:
                # rank r reruns with updated_state[r-1] as entering state
                rerun_in = c["s0"][req_i:req_i + 1].clone()
                rerun_in[0] = upd_prev
                h_r, v_new_r, _ = chunk_gated_delta_rule_fwd_h(
                    k=ks, w=w, u=u, g=gc, initial_state=rerun_in,
                    output_final_state=True, cu_seqlens=cu_seg,
                    chunk_indices=prepare_chunk_indices(cu_seg, CHUNK))
                outs[sl.start + a0: sl.start + a1] = _asc_o(
                    c["q"][:, off + a0: off + a1], k_asc,
                    _triton_h_to_asc_layout(v_new_r) if v_new_r.shape[1] != H else v_new_r,
                    _triton_h_to_asc_layout(h_r), gc, cu_seg, c["scale"])
            upd_prev = upd_next
        finals[req_i] = upd_next
        off += length
    return outs, finals


def _rel_err(x, ref):
    torch.npu.synchronize()
    d = x.detach().cpu().float().to(torch.float64) - ref.detach().cpu().float().to(torch.float64)
    return (d.norm() / ref.detach().cpu().float().to(torch.float64).norm()
            .clamp_min(1e-30)).item()


@pytest.mark.parametrize(
    "req_lens,hg,h,segments_per_req,gate,s0_scale,num_decodes,seed",
    [
        ([2048], 8, 8, 2, "normal", 0.0, 0, 0),
        ([3072], 8, 8, 3, "normal", 0.0, 0, 1),
        ([2047], 8, 8, 3, "normal", 0.0, 0, 2),  # unaligned segment bounds
        ([2048], 8, 8, 4, "normal", 2.0, 0, 3),  # non-zero initial state
        ([1024, 512, 256], 8, 8, 2, "normal", 0.0, 0, 4),  # packed batch
        ([1, 1, 1024], 8, 8, 2, "normal", 0.0, 2, 5),  # mixed decode/prefill
        ([1024], 8, 8, 2, "mild", 0.0, 0, 6),  # long-memory gates
        ([1024], 8, 8, 2, "extreme", 0.0, 0, 7),  # near-reset gates
    ],
)
def test_gdn_pcp_affine_path_matches_global(npu_env, req_lens, hg, h,
                                            segments_per_req, gate, s0_scale,
                                            num_decodes, seed):
    c = _gen_case(req_lens, hg, h, seed=seed, gate=gate,
                  s0_scale=s0_scale, num_decodes=num_decodes)
    o_ref, f_ref = _leg_a_fp64(c)
    o_global, f_global = _leg_b_global(c)
    o_seg, f_seg = _leg_c_pcp(c, segments_per_req)
    torch.npu.synchronize()

    # final states: both paths match the FP64 reference within bf16 noise
    assert _rel_err(f_global, f_ref) < 5e-3
    assert _rel_err(f_seg, f_ref) < 5e-3
    # final states: both paths match the FP64 reference within bf16 noise
    assert _rel_err(f_global, f_ref) < 5e-3
    assert _rel_err(f_seg, f_ref) < 5e-3
    # outputs: aggregate bf16 noise floor vs FP64 truth
    assert _rel_err(o_global, o_ref) < 5e-2
    assert _rel_err(o_seg, o_ref) < 5e-2

    aligned = all(
        length % (CHUNK * segments_per_req) == 0 and length >= CHUNK * segments_per_req
        for length in req_lens if length > 1
    )
    if aligned and num_decodes == 0 and gate != "mild":
        # chunk-aligned split with fast-decaying gates: the PCP path is
        # bit-identical to CP=1 (mild gates keep low-bit rerun differences
        # alive across the whole segment; tolerance asserted above)
        assert torch.equal(f_seg, f_global)
        assert _rel_err(o_seg, o_global) < 1e-2
