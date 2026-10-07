"""M3-A/B measurement: step v2 pure-kernel timing (reset OUTSIDE the loop,
same caliber as the dequant/gemv benches) + triton step same caliber +
production-form wall (fresh kn/qn per step -> executor slow path).

A table per B: dequant | gemv | epilogue (measured) | 3-op kernel total |
triton step (same caliber) | total vs 4.238 (frozen M1 pure-kernel line)
| vs 5.8 (dense fla production line).
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step  # noqa: E402
from gdn_lq_step_v2 import (  # noqa: E402
    step_pools_v2, step_pools_v2_strict, warm_step_v2, _bufs,
)
from gdn_lq_dequant import gdn_lq_dequant  # noqa: E402
from gdn_lq_gemv import gdn_lq_gemv  # noqa: E402
from gdn_lq_epilogue import gdn_lq_epilogue  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128
ITERS = 50


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def make_inputs(b, seed=11):
    g = torch.Generator().manual_seed(seed)
    nk = H // 2
    kn = _norm(torch.randn(b, nk, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(b, nk, K, generator=g)).to(DEVICE)
    vx = torch.randn(b, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    g_log = (torch.randn(b, H, generator=g) * 0.05).to(DEVICE)
    beta = torch.rand(b, H, generator=g).to(torch.bfloat16).to(DEVICE)
    return kn, qn, vx, g_log, beta


def time_op(fn, iters=ITERS):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(b, rounds=2):
    slots = max(4096, b + 8)
    gen = torch.Generator().manual_seed(3)
    idx = (torch.arange(b) + 1).to(DEVICE)
    buf = torch.randn(slots, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    assert warm_step_v2(DEVICE)
    kn, qn, vx, g_log, beta = make_inputs(b)
    kn_e = kn.repeat_interleave(2, dim=1)
    qn_e = qn.repeat_interleave(2, dim=1)
    staging, aq, cq = _bufs(b, H, V, K, DEVICE)
    o = torch.empty(b, H, V, dtype=torch.bfloat16, device=DEVICE)
    nk = H // 2

    def reset():
        pools.fill.zero_()
        pools.w.zero_()

    def dequant():
        assert gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, H, nk, 1.0)

    def gemv():
        assert gdn_lq_gemv(staging, aq, cq)

    def epi():
        assert gdn_lq_epilogue(pools._state, idx, cq, kn, qn, vx, g_log, beta,
                               o, H, nk, 1.0)

    def triton():
        gdn_leapquant_step(pools, idx, kn_e, qn_e, vx, g_log, beta, 1.0)

    # pre-run once so slow paths are done; then each timing block resets
    # OUTSIDE the loop (pure kernel caliber)
    res = {}
    for name, fn in (("dequant", dequant), ("gemv", gemv), ("epilogue", epi),
                     ("triton", triton)):
        vals = []
        for _ in range(rounds):
            reset()
            torch.npu.synchronize()
            vals.append(time_op(fn))
        res[name] = (min(vals), max(vals))

    # production-form wall: fresh kn/qn CONTENT per step, produced on
    # device (as real projection outputs are) - the tensors are new views
    # each step, so without input pooling this exercises the executor
    # slow path; with pooling only the copy_ into fixed slots remains
    def prod_inputs():
        gg = torch.Generator(device=DEVICE).manual_seed(1000)
        nk = H // 2
        kn2 = _norm(torch.randn(b, nk, K, generator=gg, device=DEVICE))
        qn2 = _norm(torch.randn(b, nk, K, generator=gg, device=DEVICE))
        vx2 = torch.randn(b, H, V, generator=gg, device=DEVICE).to(torch.bfloat16)
        g2 = (torch.randn(b, H, generator=gg, device=DEVICE) * 0.05)
        b2 = torch.rand(b, H, generator=gg, device=DEVICE).to(torch.bfloat16)
        return kn2, qn2, vx2, g2, b2

    def prod_once():
        kn2, qn2, vx2, g2, b2 = prod_inputs()
        step_pools_v2_strict(pools, idx, kn2, qn2, vx2, g2, b2, 1.0)

    prod_once()
    reset()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        kn2, qn2, vx2, g2, b2 = prod_inputs()
        step_pools_v2_strict(pools, idx, kn2, qn2, vx2, g2, b2, 1.0)
    torch.npu.synchronize()
    res["prod_wall"] = ((time.perf_counter() - t0) / ITERS * 1e3,) * 2

    # cached-wall (inputs hoisted, executor hits)
    def cached_once():
        step_pools_v2_strict(pools, idx, kn, qn, vx, g_log, beta, 1.0)

    reset()
    torch.npu.synchronize()
    vals = []
    for _ in range(rounds):
        reset()
        torch.npu.synchronize()
        vals.append(time_op(cached_once))
    res["cached_wall"] = (min(vals), max(vals))

    print(f"B={b:4d} ({b*H:6d} inst):")
    print(f"  dequant   {res['dequant'][0]:7.3f}-{res['dequant'][1]:7.3f} ms")
    print(f"  gemv      {res['gemv'][0]:7.3f}-{res['gemv'][1]:7.3f} ms")
    print(f"  epilogue  {res['epilogue'][0]:7.3f}-{res['epilogue'][1]:7.3f} ms  (MEASURED)")
    tot = (res['dequant'][0] + res['gemv'][0] + res['epilogue'][0],
           res['dequant'][1] + res['gemv'][1] + res['epilogue'][1])
    print(f"  3op total {tot[0]:7.3f}-{tot[1]:7.3f} ms   vs 4.238: {tot[1]/4.238:.2f}x  vs 5.8: {tot[1]/5.8:.2f}x")
    print(f"  triton    {res['triton'][0]:7.3f}-{res['triton'][1]:7.3f} ms  (reset outside loop)")
    print(f"  cached wall {res['cached_wall'][0]:7.3f}-{res['cached_wall'][1]:7.3f} ms")
    print(f"  prod  wall {res['prod_wall'][0]:7.3f}-{res['prod_wall'][1]:7.3f} ms  (fresh kn/qn per step, slow path)")
    return res


if __name__ == "__main__":
    for b in (16, 64, 256):
        run(b)
