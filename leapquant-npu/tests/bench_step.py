"""Micro-benchmark: Triton-Ascend LeapQuant step kernel vs eager torch pools."""

import sys
from pathlib import Path
import time

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step  # noqa: E402

DEVICE = "npu"
H, HK, V, K, SLOTS = 32, 16, 128, 128, 2048


def bench(B, iters=50):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(1, SLOTS, (B,), generator=g).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=g).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    kn = torch.randn(B, H, K, generator=g).to(torch.bfloat16).to(DEVICE)
    qn = torch.randn(B, H, K, generator=g).to(torch.bfloat16).to(DEVICE)
    v = torch.randn(B, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    g_log = (torch.randn(B, H, generator=g) * 0.05).to(DEVICE)
    beta = torch.rand(B, H, generator=g).to(torch.bfloat16).to(DEVICE)

    def reset():
        t_idx, h_idx = pools.th_index(idx)
        pools.fill[t_idx, h_idx] = 0
        pools.w[t_idx, h_idx] = 0.0

    def run_torch():
        pools.decode_step(idx, kn, qn, v, g_log, beta, 1.0)
        reset()

    def run_triton():
        gdn_leapquant_step(pools, idx, kn, qn, v, g_log, beta, 1.0)
        reset()

    for name, fn in [("torch", run_torch), ("triton", run_triton)]:
        fn()  # warm
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        print(f"B={B:4d} {name:6s}: {dt:8.3f} ms/step-layer")


for b in [16, 64, 256]:
    bench(b)
