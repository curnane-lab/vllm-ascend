"""Benchmark: AscendC LeapQuant flush vs eager torch pools.flush (all due)."""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128
SLOTS = 2048


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def bench(B, iters=30):
    gen = torch.Generator().manual_seed(0)
    idx = (torch.arange(B) + 1).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)

    # drive one full window so every (slot, head) is due, with real data
    for t in range(WINDOW):
        g = torch.Generator().manual_seed(5000 + t)
        kn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        qn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        vx = torch.randn(B, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
        g_log = (torch.randn(B, H, generator=g) * 0.05).to(DEVICE)
        beta = torch.rand(B, H, generator=g).to(torch.bfloat16).to(DEVICE)
        pools.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)

    def reset():
        t_idx, h_idx = pools.th_index(idx)
        pools.fill[t_idx, h_idx] = WINDOW  # everything due again

    res = {}
    runners = (("torch", lambda: pools.flush(idx)),
               ("ascendc", lambda: flush_pools_ascendc(pools, idx)))
    for name, fn in runners:
        fn()  # warm (also builds the executor cache for ascendc)
        reset()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
            reset()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        res[name] = dt
    print(f"B={B:4d} all-due ({B * H:6d} programs): "
          f"torch={res['torch']:7.3f} ms  ascendc={res['ascendc']:7.3f} ms  "
          f"speedup={res['torch'] / res['ascendc']:5.2f}x")


for b in (16, 64, 256):
    bench(b)
