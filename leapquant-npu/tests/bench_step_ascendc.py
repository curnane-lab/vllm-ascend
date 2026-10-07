"""Micro-benchmark: AscendC LeapQuant step vs triton vs eager torch pools."""

import sys
from pathlib import Path
import time

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step  # noqa: E402
from vllm_ascend.ops.gdn_lq_step_ascendc import step_pools_ascendc  # noqa: E402

DEVICE = "npu"
H, V, K, SLOTS = 32, 128, 128, 2048


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def bench(B, iters=50):
    g = torch.Generator().manual_seed(0)
    idx = (torch.arange(B) + 1).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=g).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    kn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
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

    def run_ascendc():
        r = step_pools_ascendc(pools, idx, kn, qn, v, g_log, beta, 1.0)
        assert r is not None
        reset()

    res = {}
    for name, fn in (("torch", run_torch), ("triton", run_triton),
                     ("ascendc", run_ascendc)):
        try:
            fn()  # warm (builds the executor cache for ascendc)
            reset()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
            res[name] = (time.perf_counter() - t0) / iters * 1e3
        except Exception as e:
            res[name] = None
            print(f"  {name} failed: {type(e).__name__}: {e}")
    line = f"B={B:4d} ({B * H:6d} inst): " + "  ".join(
        f"{n}={res[n]:7.3f} ms" if res[n] is not None else f"{n}=   FAIL"
        for n in ("torch", "triton", "ascendc"))
    if res["triton"] and res["ascendc"]:
        line += f"  ascendc-vs-triton={res['triton'] / res['ascendc']:5.2f}x"
    print(line)


for b in (16, 64, 256):
    bench(b)
