# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the GDN PCP rerun-elimination path.

The PCP prefill previously re-ran a full fwd_h with the corrected entering
state on non-first ranks. chunk.py now applies an affine correction of the
already-computed h/v_new instead:

    h_c   += M_c @ delta_s            (M_c = per-chunk cumulative map,
                                       row = chunk_offsets[s] + s + c)
    v_new -= W_c @ (M_c @ delta_s)    (v_new = v - W @ h_chunk_start is
                                       linear in the chunk-start state)
    delta_s = updated_h_state - initial_state   (zero for decode rows)

The initial state enters the chunk-0 h/v_new through a kernel-specific
path that this affine correction does not cover, so chunk 0 of each rank is
recomputed by a minimal (<= 64-token) fwd_h rerun with the corrected
entering state.

These tests drive the real ``chunk_gated_delta_rule_fwd`` with a mocked PCP
group (single NPU, two or more logical ranks) and compare the reassembled
output against the CP=1 run of the same function plus the CPU FP64
per-token recurrence.

Requires one NPU; skipped otherwise.
"""

from types import SimpleNamespace

import pytest
import torch

try:
    from vllm_ascend.ops.triton.fla import chunk as chunk_mod
    from vllm_ascend.ops.triton.fla.chunk_delta_hupdate import (
        chunk_gated_delta_rule_fwd_hupdate,
    )
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.utils import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
        prepare_final_chunk_indices,
        prepare_update_chunk_offsets,
    )
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
    from vllm_ascend.utils import enable_custom_op
except ImportError as exc:  # pragma: no cover - non-NPU environment
    pytest.skip(f"vllm-ascend NPU build unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires NPU",
)

CHUNK = 64


class FakePcpGroup:
    """Minimal stand-in for the PCP communication group.

    Per forward, chunk.py performs two all_gather calls in a fixed order:
    first the per-rank final states, then the per-rank final h_update maps.
    ``final_states`` / ``h_updates`` are pre-filled by the test with every
    logical rank's F_r(s0) and Phi_r so that a sequential single-process run
    can emulate them.
    """

    def __init__(self, world_size, rank_in_group, final_states, h_updates):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.final_states = final_states
        self.h_updates = h_updates
        self._call = 0

    def all_gather(self, input_, dim=-1):
        self._call += 1
        if self._call % 2 == 1:
            return torch.stack(self.final_states)
        return torch.stack(self.h_updates)


def _gen_case(req_lens, hg, h, seed, gate="normal", s0_scale=0.0):
    dev = torch.device("npu:0")
    total = sum(req_lens)
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(total, h, generator=gen) - 0.5
    if gate == "extreme":
        mix = torch.rand(total, h, generator=gen)
        a = torch.where(
            mix < 0.7,
            torch.randn(total, h, generator=gen) * 0.5 + 3.0,
            torch.where(
                mix < 0.9,
                torch.randn(total, h, generator=gen) * 0.5 - 4.0,
                torch.randn(total, h, generator=gen) * 0.5 - 12.0,
            ),
        )
    k = torch.nn.functional.normalize(torch.randn(total, hg, 128, generator=gen), dim=-1)
    v = torch.randn(total, h, 128, generator=gen)
    q = torch.randn(total, h, 128, generator=gen)
    cu = [0]
    for length in req_lens:
        cu.append(cu[-1] + length)
    s0 = torch.zeros(len(req_lens), h, 128, 128, device=dev, dtype=torch.float32)
    if s0_scale > 0:
        s0 = (torch.randn(len(req_lens), h, 128, 128, generator=gen, dtype=torch.float64) * s0_scale).float().to(dev)
    return {
        "q": q.to(dev, torch.bfloat16).unsqueeze(0),
        "k": k.to(dev, torch.bfloat16).unsqueeze(0),
        "v": v.to(dev, torch.bfloat16).unsqueeze(0),
        "g": torch.nn.functional.logsigmoid(a).float().to(dev).unsqueeze(0),
        "beta": torch.sigmoid(torch.randn(total, h, generator=gen)).float().to(dev).unsqueeze(0),
        "cu": cu,
        "s0": s0,
        "scale": 128**-0.5,
    }


def _rank_slice(case, req_lens, rank, world):
    """Per-sequence contiguous token slices for one logical rank.

    Returns (gather-slice, rank-local cu_seqlens, per-sequence global starts).
    """
    pieces = []  # global (a, b) per sequence
    cu_loc = [0]  # rank-local cumulative lengths
    g = 0
    for length in req_lens:
        base = length // world
        start = rank * base
        end = length if rank == world - 1 else (rank + 1) * base
        pieces.append((g + start, g + end))
        cu_loc.append(cu_loc[-1] + (end - start))
        g += length
    idx = torch.cat([torch.arange(a, b) for a, b in pieces]).to(case["q"].device)
    cu_t = torch.tensor(cu_loc, device=case["q"].device, dtype=torch.int64)
    sl = lambda x: x[:, idx].contiguous()
    return sl, cu_t, [p[0] for p in pieces]


def _prebuilt_meta(cu_t, num_decodes=0):
    cu_host = tuple(cu_t.tolist())
    ci = prepare_chunk_indices(cu_t, CHUNK)
    return SimpleNamespace(
        block_indices_cumsum=None,
        cu_seqlens_host=cu_host,
        cu_seqlens_kern=None,
        chunk_indices_chunk64=ci,
        chunk_indices_chunk64_host=tuple(ci.flatten().tolist()),
        chunk_offsets_chunk64=prepare_chunk_offsets(cu_t, CHUNK),
        update_chunk_offsets_chunk64=prepare_update_chunk_offsets(cu_t, CHUNK),
        final_chunk_indices_chunk64=prepare_final_chunk_indices(cu_t, CHUNK),
        chunk_indices_large_block=None,
        keep_meta=None,
        num_decodes=num_decodes,
    )


def _rank_pieces(case, req_lens, rank, world):
    """F_r(s0) and Phi_r for one logical rank (mirrors chunk.py internals)."""
    sl, cu_t, _ = _rank_slice(case, req_lens, rank, world)
    k, v, g, beta = sl(case["k"]), sl(case["v"]), sl(case["g"]), sl(case["beta"])
    gc = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu_t)
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=gc, cu_seqlens=cu_t)
    A = solve_tril(A=A, cu_seqlens=cu_t)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=gc, cu_seqlens=cu_t)
    h, _, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k.to(torch.bfloat16).transpose(1, 2).contiguous(),
        w.to(torch.bfloat16).transpose(1, 2).contiguous(),
        u.to(torch.bfloat16).transpose(1, 2).contiguous(),
        g=gc.transpose(1, 2).contiguous(),
        gk=None,
        initial_state=case["s0"],
        output_final_state=True,
        chunk_size=CHUNK,
        save_new_value=True,
        cu_seqlens=cu_t.tolist(),
        chunk_indices=prepare_chunk_indices(cu_t, CHUNK).flatten().tolist(),
        use_exp2=False,
        transpose_state_layout=False,
    )
    h_update = chunk_gated_delta_rule_fwd_hupdate(
        k=k,
        w=w,
        u=u,
        g=gc,
        cu_seqlens=cu_t,
        chunk_indices=prepare_chunk_indices(cu_t, CHUNK),
        chunk_offsets=prepare_chunk_offsets(cu_t, CHUNK),
        update_chunk_offsets=prepare_update_chunk_offsets(cu_t, CHUNK),
        num_decodes=0,
    )
    fci = prepare_final_chunk_indices(cu_t, CHUNK)
    return final, h_update[:, fci, :, :, :]


def _run_fwd(case, req_lens, group, world, rank, monkeypatch):
    sl, cu_t, _ = _rank_slice(case, req_lens, rank, world)
    meta = _prebuilt_meta(cu_t)
    monkeypatch.setattr(chunk_mod, "get_pcp_group", lambda: group)
    monkeypatch.setattr(chunk_mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    return chunk_mod.chunk_gated_delta_rule_fwd(
        q=sl(case["q"]),
        k=sl(case["k"]),
        v=sl(case["v"]),
        g=sl(case["g"]),
        beta=sl(case["beta"]),
        scale=case["scale"],
        initial_state=case["s0"],
        output_final_state=True,
        cu_seqlens=cu_t,
        prebuilt_meta=meta,
    )


def _leg_a_fp64(case, req_lens):
    q = case["q"][0].float().cpu().double()
    k = case["k"][0].float().cpu().double()
    v = case["v"][0].float().cpu().double()
    g = case["g"][0].cpu().double()
    b = case["beta"][0].cpu().double()
    s0 = case["s0"].cpu().double()
    total = sum(req_lens)
    outs = torch.zeros(total, q.shape[1], 128, dtype=torch.float64)
    finals = s0.clone()
    off = 0
    for i, length in enumerate(req_lens):
        S = s0[i].clone()
        for t in range(length):
            gam = g[off + t].exp().view(-1, 1, 1)
            bt = b[off + t].view(-1, 1, 1)
            kt = k[off + t].view(-1, 128, 1)
            vt = v[off + t].view(-1, 1, 128)
            S = gam * (S - bt * torch.bmm(kt, torch.bmm(kt.transpose(1, 2), S))) + bt * torch.bmm(kt, vt)
            outs[off + t] = torch.einsum("hk,hkv->hv", q[off + t], S) * case["scale"]
        finals[i] = S
        off += length
    return outs, finals


@pytest.fixture(autouse=True)
def _npu_env():
    torch.npu.set_device(0)
    assert enable_custom_op()
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )

    init_device_properties_triton()
    yield


@pytest.mark.parametrize(
    "req_lens,world,seed,gate,s0_scale",
    [
        ([6144], 2, 0, "normal", 0.0),
        ([6144], 3, 1, "normal", 0.0),
        ([6143], 3, 2, "normal", 0.0),
        ([8192], 4, 3, "normal", 2.0),
        ([8192], 2, 4, "extreme", 0.0),
        ([4096, 2048, 2048], 2, 5, "normal", 0.0),
    ],
)
def test_pcp_rerun_elimination_matches_cp1(req_lens, world, seed, gate, s0_scale, monkeypatch):
    case = _gen_case(req_lens, 8, 8, seed, gate, s0_scale)
    pieces = [_rank_pieces(case, req_lens, r, world) for r in range(world)]

    # CP=1 reference through the same function
    ref_group = FakePcpGroup(1, 0, [case["s0"]], [case["s0"]])
    ref_out = _run_fwd(case, req_lens, ref_group, 1, 0, monkeypatch)
    o_ref, final_ref = ref_out[1], ref_out[3]

    o_parts, final_pcp = [], None
    for rank in range(world):
        fs = [pieces[r][0] for r in range(world)]
        fus = [pieces[r][1] for r in range(world)]
        grp = FakePcpGroup(world, rank, fs, fus)
        out = _run_fwd(case, req_lens, grp, world, rank, monkeypatch)
        o_parts.append(out[1])
        final_pcp = out[3]
    o_pcp = torch.empty_like(o_ref)
    for rank in range(world):
        _, cu_loc, gstarts = _rank_slice(case, req_lens, rank, world)
        off = 0
        for si in range(len(req_lens)):
            n = int(cu_loc[si + 1] - cu_loc[si])
            a = gstarts[si]
            if n:
                o_pcp[:, a : a + n] = o_parts[rank][:, off : off + n]
                off += n

    d = (o_pcp.float() - o_ref.float()).abs().norm(dim=-1)
    n = o_ref.float().norm(dim=-1).clamp_min(1e-30)
    assert (d / n).max().item() < 0.02, "output mismatch vs CP=1"
    assert torch.allclose(final_pcp.float(), final_ref.float(), atol=1e-2, rtol=1e-2)

    o64, f64 = _leg_a_fp64(case, req_lens)
    d64 = ((o_pcp.float().cpu().double() - o64).norm(dim=-1) / o64.norm(dim=-1).clamp_min(1e-30)).max().item()
    assert d64 < 0.05, f"output mismatch vs FP64: {d64}"
    f_rel = ((final_pcp.cpu().double() - f64).norm() / f64.norm()).item()
    assert f_rel < 0.05, f"final state mismatch vs FP64: {f_rel}"
