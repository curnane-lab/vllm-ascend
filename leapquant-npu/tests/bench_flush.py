"""M3 bench: AscendC flush latency, nothing-due vs all-due.

nothing-due: window at fill=1 (one step, no flush since) - measures the
due-scan fast path. With fill=L semantics a flush does NOT clear fill,
so after an all-due flush the head stays due; the nothing-due fixture
must therefore be re-created (1 step) rather than re-flushed.
Timing caliber: warm once (executor cache) -> sync -> perf_counter loop.
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc  # noqa: E402

DEVICE = "npu"
H, V, K, SLOTS = 32, 128, 128, 4096
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


def bench(b):
    gen = torch.Generator().manual_seed(3)
    idx = (torch.arange(b) + 1).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    kn, qn, vx, g_log, beta = make_inputs(b)
    kn_e = kn.repeat_interleave(2, dim=1)
    qn_e = qn.repeat_interleave(2, dim=1)

    def fill_window(steps):
        # reset then run `steps` steps without flush
        t_idx, h_idx = pools.th_index(idx)
        pools.fill[t_idx, h_idx] = 0
        pools.w[t_idx, h_idx] = 0.0
        for _ in range(steps):
            pools.decode_step(idx, kn_e, qn_e, vx, g_log, beta, 1.0)

    def time_flush(tag):
        assert flush_pools_ascendc(pools, idx)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            assert flush_pools_ascendc(pools, idx)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / ITERS * 1e3
        print(f"B={b:4d} {tag}: {ms:.3f} ms")
        return ms

    # all-due: window full (16 steps). NOTE fill=L semantics: the first
    # flush folds and leaves fill=L; repeated flush calls would re-fold
    # (still due), which IS the all-due steady-state cost.
    fill_window(16)
    time_flush("all-due ")
    # nothing-due: fresh window (1 step -> fill=1)
    fill_window(1)
    time_flush("nothing-due")


for b in (16, 64, 256):
    bench(b)
