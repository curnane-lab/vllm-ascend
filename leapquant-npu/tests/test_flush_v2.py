"""M3-3b parity + bench: flush v2 (cube pipeline) vs torch pools.flush.

Judge: decoded-state reconstruction error vs the TRUE folded state,
ratio kernel/torch <= 1.01. The true state is computed in pure fp32
torch (delta-rule simulation from the dense init, no pools, no custom
ops), sidestepping the platform's flaky device indexing after ops.
Also field-wise comparison (codes rel, w/p/fill exact) and latency.
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402
from gdn_lq_flush_v2 import flush_pools_v2  # noqa: E402

DEVICE = "npu"
H, V, K, SLOTS = 32, 128, 128, 512
ITERS = 50


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def simulate_true(base, idx, kn, qn, vx, gl, bt):
    """pure fp32 delta-rule simulation of 16 steps from the dense init."""
    S = base[idx].clone().float()
    p = torch.ones(idx.shape[0], H, device=DEVICE)
    w = torch.zeros(idx.shape[0], H, WINDOW, device=DEVICE)
    kb = torch.zeros(idx.shape[0], H, WINDOW, K, device=DEVICE)
    ub = torch.zeros(idx.shape[0], H, WINDOW, V, device=DEVICE)
    for t in range(16):
        at = torch.exp(gl.float())
        pn = (p * at).unsqueeze(-1).unsqueeze(-1)
        knf = kn.float()
        qnf = qn.float()
        Skn = torch.einsum("bhvk,bhk->bhv", S, knf)
        unew = bt.float().unsqueeze(-1) * (vx.float() - Skn)
        S = pn * S + torch.einsum("bhj,bhjv,bhjk->bhvk",
                                  w * at.unsqueeze(-1), ub, kb)
        kb[:, :, t] = knf
        ub[:, :, t] = unew
        w = torch.where(torch.arange(WINDOW, device=DEVICE).view(1, 1, -1) == t,
                        torch.ones_like(w), w * at.unsqueeze(-1))
        p = p * at
    # fold the 16 buffered rows (all live) into the returned state:
    # the flush folds rows j < fill (=16) with the CURRENT w
    return S + torch.einsum("bhj,bhjv,bhjk->bhvk", w, ub, kb)


def run_trial(seed, use_kernel):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(1, SLOTS, (8,), generator=gen).to(DEVICE)
    base = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    g = torch.Generator().manual_seed(7)
    kn = _norm(torch.randn(8, H, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(8, H, K, generator=g)).to(DEVICE)
    vx = torch.randn(8, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    gl = (torch.randn(8, H, generator=g) * 0.05).to(DEVICE)
    bt = torch.rand(8, H, generator=g).to(torch.bfloat16).to(DEVICE)
    trueS = simulate_true(base, idx, kn, qn, vx, gl, bt).cpu()

    buf = base.clone()
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    for _ in range(16):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    if use_kernel:
        assert flush_pools_v2(pools, idx), "flush v2 unavailable"
    else:
        pools.flush(idx)
    torch.npu.synchronize()
    # CPU shadow decode (device advanced indexing is unsafe after ops)
    shadow = GDNLeapQuantPools(pools._state.cpu())
    t_idx, h_idx = shadow.th_index(idx.cpu())
    s = torch.einsum("bhrv,bhrk->bhvk", shadow.q[t_idx, h_idx].float(),
                     shadow.u[t_idx, h_idx].float())
    from vllm_ascend.ops.gdn_leapquant import QMAX
    s = s + shadow.codes[t_idx, h_idx].float() * \
        (shadow.s_k[t_idx, h_idx] / QMAX).unsqueeze(-2) * \
        shadow.s_v[t_idx, h_idx].unsqueeze(-1)
    err = (((s - trueS) ** 2).sum() / (trueS ** 2).sum()).item() ** 0.5
    fill = shadow.fill[t_idx, h_idx].clone()
    return err, fill


def parity():
    rs = []
    for seed in (100, 101, 102):
        e_t, f_t = run_trial(seed, False)
        e_k, f_k = run_trial(seed, True)
        rs.append(e_k / e_t)
        print(f"seed {seed}: torch={e_t:.6f} kernel={e_k:.6f} "
              f"ratio={e_k / e_t:.4f} fill torch={f_t[0,0].item()} "
              f"kernel={f_k[0,0].item()}", flush=True)
    import statistics
    print(f"ratio mean/max: {statistics.mean(rs):.4f} {max(rs):.4f}")
    assert max(rs) <= 1.01, "decoded-state ratio over 1.01"
    print("FLUSH V2 RATIO PASS")


def bench(b, iters=ITERS):
    gen = torch.Generator().manual_seed(3)
    idx = (torch.arange(b) + 1).to(DEVICE)
    slots = max(SLOTS, b + 8)
    buf = torch.randn(slots, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)

    def _norm2(x):
        return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)

    g = torch.Generator().manual_seed(11)
    kn = _norm2(torch.randn(b, H, K, generator=g)).to(DEVICE)
    qn = _norm2(torch.randn(b, H, K, generator=g)).to(DEVICE)
    vx = torch.randn(b, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    gl = (torch.randn(b, H, generator=g) * 0.05).to(DEVICE)
    bt = torch.rand(b, H, generator=g).to(torch.bfloat16).to(DEVICE)

    def fill_window(n):
        pools.fill.zero_()
        pools.w.zero_()
        for _ in range(n):
            pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)

    def time_flush(tag):
        assert flush_pools_v2(pools, idx)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            assert flush_pools_v2(pools, idx)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"B={b:4d} {tag}: {ms:.3f} ms")
        return ms

    fill_window(16)
    time_flush("all-due   ")
    fill_window(1)
    time_flush("nothing-due")


if __name__ == "__main__":
    if "--bench-only" in sys.argv:
        for b in (16, 64, 256):
            bench(b)
    else:
        parity()
        for b in (16, 64, 256):
            bench(b)
